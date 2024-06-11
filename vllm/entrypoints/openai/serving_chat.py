import base64
import codecs
import time
from dataclasses import dataclass
import traceback
from typing import (AsyncGenerator, AsyncIterator, Dict, Iterable, List,
                    Optional)
from typing import Sequence as GenericSequence
from typing import TypedDict, Union, cast, final

from fastapi import Request
from openai.types.chat import ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionContentPartParam, ChatCompletionLogProb,
    ChatCompletionLogProbs, ChatCompletionLogProbsContent,
    ChatCompletionMessageParam, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse, OpenAIBaseModel, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (LoRAModulePath,
                                                    OpenAIServing)
from vllm.logger import init_logger
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor)
from vllm.outputs import RequestOutput
from vllm.sequence import Logprob, MultiModalData
from vllm.utils import random_uuid
from PIL import Image
import re
import io

from .make_prompt import make_prompt

logger = init_logger(__name__)

def time_ms():
    return time.time() * 1000

@final  # So that it should be compatible with Dict[str, str]
class ConversationMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class ChatMessageParseResult:
    messages: List[ConversationMessage]
    is_multimodal: bool = False

class CompletionPerformanceStatistics(OpenAIBaseModel):
    # store when this statistics class created
    statistics_created_at: float
    # store when the request is recieved by serving module
    request_created_at: float
    # store the time interval between current engine response and last engine response
    runner_last_latency: Optional[float] = None
    # store the last latency measure of model latency
    runner_last_model_latency: Optional[float] = None
    # store the last latency measure of cuda graph prepare latency
    runner_last_perpare_latency: Optional[float] = None
    # store the last latency measure of sampler latency
    runner_last_sampler_latency: Optional[float] = None
    # estimated current request throughput (tok / sec)
    request_throughput: Optional[float] = None
    

class ChatCompletionStreamResponseWithStatistics(ChatCompletionStreamResponse):
    performance: Optional[CompletionPerformanceStatistics] = None

class PerformanceTracker:
    def __init__(self):
        self.created_at = time_ms()
    
    def step(self, output: RequestOutput) -> CompletionPerformanceStatistics:
        if output.metrics.last_runner_latency is not None and len(output.metrics.last_runner_latency) > 0:
            return CompletionPerformanceStatistics(
                statistics_created_at=time_ms(),
                request_created_at=time_ms(),
                runner_last_latency=output.metrics.last_runner_latency[-1],
                runner_last_model_latency=output.metrics.last_runner_model_latency[-1],
                runner_last_perpare_latency=output.metrics.last_runner_prepare_latency[-1],
                runner_last_sampler_latency=output.metrics.last_runner_sampler_latency[-1],
                request_throughput=1.0 / (output.metrics.last_runner_latency[-1] / 1000),
            )
        else:
            return CompletionPerformanceStatistics(
                statistics_created_at=time_ms(),
                request_created_at=time_ms(),
            )

class OpenAIServingChat(OpenAIServing):

    def __init__(
        self,
        engine: AsyncLLMEngine,
        model_config: ModelConfig,
        served_model_names: List[str],
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None
    ):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=lora_modules)

        self.response_role = response_role
        self._load_chat_template(chat_template)

    def _load_chat_template(self, chat_template: Optional[str]):
        tokenizer = self.tokenizer

        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    tokenizer.chat_template = f.read()
            except OSError as e:
                JINJA_CHARS = "{}\n"
                if not any(c in chat_template for c in JINJA_CHARS):
                    msg = (f"The supplied chat template ({chat_template}) "
                           f"looks like a file path, but it failed to be "
                           f"opened. Reason: {e}")
                    raise ValueError(msg) from e

                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                tokenizer.chat_template = codecs.decode(
                    chat_template, "unicode_escape")

            logger.info("Using supplied chat template:\n%s",
                        tokenizer.chat_template)
        elif tokenizer.chat_template is not None:
            logger.info("Using default chat template:\n%s",
                        tokenizer.chat_template)
        else:
            logger.warning(
                "No chat template provided. Chat API will not work.")

    def _parse_chat_message_content_parts(
        self,
        role: str,
        parts: Iterable[ChatCompletionContentPartParam],
    ) -> ChatMessageParseResult:
        texts: List[str] = []
        
        is_multimodal = False

        for _, part in enumerate(parts):
            part_type = part["type"]
            if part_type == "text":
                text = cast(ChatCompletionContentPartTextParam, part)["text"]

                texts.append(text)
            elif part_type == 'image_url':
                is_multimodal = True
                
                image_url = cast(ChatCompletionContentPartImageParam, part)["image_url"]['url']
                image_data = re.sub('^data:image/.+;base64,', '', image_url)
                if len(image_data) > 0:
                    image_data = base64.b64decode(image_data)
                    # print(len(image_data))
                    image = Image.open(io.BytesIO(image_data))
                else:
                    image = Image.open(image_url)
                
                texts.append(image)
            else:
                raise NotImplementedError(f"Unknown part type: {part_type}")

        if is_multimodal:
            messages = [ConversationMessage(role=role, content=texts)]
        else:
            messages = [ConversationMessage(role=role, content="\n".join(texts))]

        return ChatMessageParseResult(
            messages=messages, 
            is_multimodal=is_multimodal
        )

    def _parse_chat_message_content(
        self,
        message: ChatCompletionMessageParam,
    ) -> ChatMessageParseResult:
        role = message["role"]
        content = message.get("content")

        if content is None:
            return ChatMessageParseResult(messages=[])
        if isinstance(content, str):
            messages = [ConversationMessage(role=role, content=content)]
            return ChatMessageParseResult(messages=messages)

        return self._parse_chat_message_content_parts(role, content)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None
    ) -> Union[ErrorResponse, AsyncGenerator[str, None],
               ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        ChatCompletion API.

        NOTE: Currently we do not support the following feature:
            - function_call (Users should implement this by themselves)
        """
        #request.model = self.served_model
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        is_multimodal = False
        image_index = 1
        images = []
        try:
            conversation: List[ConversationMessage] = []

            for msg in request.messages:
                parsed_msg = self._parse_chat_message_content(msg)
                is_multimodal = is_multimodal or parsed_msg.is_multimodal
                conversation.extend(parsed_msg.messages)
            
            if is_multimodal:
                # convert conversation texts: List[str] to texts: str
                for message in conversation:
                    if isinstance(message['content'], (list, tuple)):
                        for idx_message in range(len(message['content'])):
                            if isinstance(message['content'][idx_message], Image.Image):
                                image = message['content'][idx_message] # type: Image.Image
                                images.append(image)
                                message['content'][idx_message] = \
                                    f'[System]: User just provided their image.\n'\
                                    f'[System]: I will provide the image medatadata here. Image size (width, height in pixels): ({image.size[0]}, {image.size[1]})'\
                                    f'[System]: Here is user provided image:\n[Image starts]\n<|image_{image_index}|>\n[Image ends]'
                                image_index += 1
                        
                        for line in message['content']:
                            assert isinstance(line, str), "every line should be string"

                        message['content'] = '\n'.join(message['content'])
                    elif isinstance(message['content'], str): pass
                    else:
                        raise Exception(f'unknown message content type {type(message["content"])}')
                    
                    assert isinstance(message['content'], str), 'every content before chat template should be string'
            
            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
            )
        except Exception as e:
            traceback.print_exc()
            logger.error("Error in applying chat template from request: %s", e)
            return self.create_error_response(str(e))

        request_id = f"cmpl-{random_uuid()}"
        prompt_text = prompt_ids = None
        multi_modal_data = None
        try:
            # Tokenize/detokenize depending on prompt format (string/token list)
            prompt_ids, prompt_text = self._validate_prompt_and_tokenize(
                request, 
                prompt=prompt, 
                add_special_tokens=False,
            )
            if is_multimodal:
                processed_inputs = self.multimodal_processor(prompt, images, return_tensors="pt")
                multi_modal_data = MultiModalData(
                    type=MultiModalData.Type.IMAGE,
                    data=processed_inputs['pixel_values'], 
                )
                assert len(processed_inputs['input_ids']) == 1, 'multimodal does not support multi batch request'
                prompt_ids = processed_inputs['input_ids'][0].tolist()
            
            sampling_params = request.to_sampling_params()
            lora_request = self._maybe_get_lora(request)
            decoding_config = await self.engine.get_decoding_config()
            guided_decoding_backend = \
                request.guided_decoding_backend or\
                decoding_config.guided_decoding_backend
            guided_decode_logits_processor = (
                await get_guided_decoding_logits_processor(
                    guided_decoding_backend, request, await
                    self.engine.get_tokenizer())
                )
            if guided_decode_logits_processor:
                if sampling_params.logits_processors is None:
                    sampling_params.logits_processors = []
                sampling_params.logits_processors.append(
                    guided_decode_logits_processor
                )
        except ValueError as e:
            return self.create_error_response(str(e))

        result_generator = self.engine.generate(
            {
                "prompt": prompt_text,
                "prompt_token_ids": prompt_ids,
                "multi_modal_data": multi_modal_data,
            },
            sampling_params,
            request_id,
            lora_request,
        )
        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id, conversation
            )
        else:
            try:
                return await self.chat_completion_full_generator(
                    request, raw_request, result_generator, request_id,
                    conversation
                )
            except ValueError as e:
                # TODO: Use a vllm-specific Validation Error
                return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1]["role"]

    async def chat_completion_stream_generator(
            self, request: ChatCompletionRequest,
            result_generator: AsyncIterator[RequestOutput], request_id: str,
            conversation: List[ConversationMessage]
    ) -> AsyncGenerator[str, None]:
        model_name = self.served_model_names[0]
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        assert request.n is not None
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        finish_reason_sent = [False] * request.n
        performance_tracker = PerformanceTracker()
        try:
            async for res in result_generator:
                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)
                    for i in range(request.n):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(role=role),
                            logprobs=None,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponseWithStatistics(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                        )
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content = ""
                        if conversation and conversation[-1].get(
                                "content") and conversation[-1].get(
                                    "role") == role:
                            last_msg_content = conversation[-1]["content"]

                        if last_msg_content:
                            for i in range(request.n):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponseWithStatistics(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    logprobs=None,
                                    model=model_name,
                                )
                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False
                
                performance_statistics = performance_tracker.step(output)
                
                print(performance_statistics)
                
                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    delta_token_ids = output.token_ids[previous_num_tokens[i]:]
                    top_logprobs = output.logprobs[
                        previous_num_tokens[i]:] if output.logprobs else None

                    if request.logprobs:
                        logprobs = self._create_chat_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=top_logprobs,
                            num_output_top_logprobs=request.top_logprobs,
                        )
                    else:
                        logprobs = None

                    delta_text = output.text[len(previous_texts[i]):]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)
                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(content=delta_text),
                            logprobs=logprobs,
                            finish_reason=None
                        )
                        chunk = ChatCompletionStreamResponseWithStatistics(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            performance=performance_statistics
                        )
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                    else:
                        # Send the finish response for each request.n only once
                        prompt_tokens = len(res.prompt_token_ids)
                        final_usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=previous_num_tokens[i],
                            total_tokens=prompt_tokens +
                            previous_num_tokens[i],
                        )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(content=delta_text),
                            logprobs=logprobs,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason)
                        chunk = ChatCompletionStreamResponseWithStatistics(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            performance=performance_statistics
                        )
                        if final_usage is not None:
                            chunk.usage = final_usage
                        data = chunk.model_dump_json(exclude_unset=True,
                                                     exclude_none=True)
                        yield f"data: {data}\n\n"
                        finish_reason_sent[i] = True
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self, request: ChatCompletionRequest, raw_request: Optional[Request],
        result_generator: AsyncIterator[RequestOutput], request_id: str,
        conversation: List[ConversationMessage]
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        model_name = self.served_model_names[0]
        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        async for res in result_generator:
            if raw_request is not None and await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None

        choices = []

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            top_logprobs = output.logprobs

            if request.logprobs:
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=top_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                )
            else:
                logprobs = None

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role=role, content=output.text),
                logprobs=logprobs,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason)
            choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if conversation and conversation[-1].get(
                    "content") and conversation[-1].get("role") == role:
                last_msg_content = conversation[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response

    def _get_top_logprobs(
            self, logprobs: Dict[int, Logprob],
            top_logprobs: Optional[int]) -> List[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=self._get_decoded_token(p[1], p[0]),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(
                    self._get_decoded_token(p[1],
                                            p[0]).encode("utf-8",
                                                         errors="replace")))
            for i, p in enumerate(logprobs.items())
            if top_logprobs and i < top_logprobs
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[Dict[int, Logprob]]],
        num_output_top_logprobs: Optional[int] = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""

        logprobs_content = []

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self.tokenizer.decode(token_id),
                        bytes=list(
                            self.tokenizer.decode(token_id).encode(
                                "utf-8", errors="replace"))))
            else:
                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=step_top_logprobs[token_id].decoded_token,
                        logprob=max(step_top_logprobs[token_id].logprob,
                                    -9999.0),
                        bytes=list(
                            step_top_logprobs[token_id].decoded_token.encode(
                                "utf-8", errors="replace")),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs, num_output_top_logprobs)))

        return ChatCompletionLogProbs(content=logprobs_content)

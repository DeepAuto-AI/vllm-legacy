import re
from io import BytesIO
from PIL import Image

from transformers import PreTrainedTokenizer
from base64 import b64decode

from .protocol import ChatCompletionRequest
from .pdf_reader import read_doc


DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def make_prompt(request: ChatCompletionRequest,
                tokenizer: PreTrainedTokenizer):

    for msg in request.messages:
        if msg['role'] == 'system' and msg['content'] == "":
            msg['content'] = DEFAULT_SYSTEM_PROMPT

        def replace_file(match):
            display_text = match.group(1)
            mime_type = match.group(2)
            base64_string = match.group(3)

            # Convert base64 to bytesIO
            file = BytesIO(b64decode(base64_string, validate=True))
            if mime_type.startswith('image/'):
                image = Image.open(file)
                # TODO: encode image to tokens
                return display_text
            elif mime_type == "application/pdf":
                texts = read_doc(file, mime_type)
                result = (
                    f"[Begin document]" +
                    "\n".join([f"(Page {t.page}) {t.text}" for t in texts]) +
                    f"[End document]"
                )
                return result
            else:
                raise ValueError(f"Unsupported mime type: {mime_type}")

        msg['content'] = re.sub(
            r'!\[([^\]]*)]\(data:([^;]*);base64,([-A-Za-z0-9+/]*={0,3})\)',
            replace_file, msg['content']
        )

    prompt = tokenizer.apply_chat_template(
        conversation=request.messages,
        tokenize=False,
        add_generation_prompt=request.add_generation_prompt)

    return prompt

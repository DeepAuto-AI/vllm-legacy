import asyncio
from importlib.metadata import requires
import os
import subprocess

from PIL import Image
from transformers import AutoProcessor

from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from vllm.sequence import MultiModalData

async def wait_response(responses):
    text = None
    for results_generator in responses:
        async for request_output in results_generator:
            # print(request_output.outputs[0].text)
            text = request_output.outputs[0].text
    return text

async def run_phi3v():
    model_path = "microsoft/Phi-3-vision-128k-instruct"
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=model_path,
        trust_remote_code=True,
        image_input_type="pixel_values",
        image_token_id=-1,
        image_input_shape="1008, 1344",
        fake_image_input_shape="1, 16, 3, 336, 336",
        image_feature_size=1921,
        max_model_len=16000,
        max_num_seqs=4,
        kv_cache_dtype="fp8_e5m2",
        dtype='half',
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
    ))
    llm.start_background_loop()

    image_1 = Image.open("images/stop_sign.jpg")
    image_2 = Image.open("images/cherry_blossom.jpg")
    user_prompt = '<|user|>\n'
    assistant_prompt = '<|assistant|>\n'
    suffix = "<|end|>\n"

    # single-image prompt
    prompt = "What is shown in this image? Describe in very detail. Try your best as much as possible!! I want to hear really in details. **But before answering, First describe how many image you can see.**"
    prompt = 'Describe each image in detail.'
    prompt_1 = f"{user_prompt}Here is image 1:\n<|image_1|>\n{prompt}{suffix}{assistant_prompt}"
    prompt_2 = f"{user_prompt}Here is image 1:\n<|image_1|>\n{prompt}{suffix}{assistant_prompt}"
    prompt_3 = f"{user_prompt}Here is image 1:\n<|image_1|>\n\nHere is image 2:\n\n<|image_2|>\n{prompt}{suffix}{assistant_prompt}"
    inputs_1 = processor(prompt_1, image_1, return_tensors="pt")
    inputs_2 = processor(prompt_2, image_2, return_tensors="pt")
    inputs_3 = processor(prompt_3, [image_1, image_2], return_tensors="pt")
    multi_modal_data_1 = MultiModalData(
        type=MultiModalData.Type.IMAGE,
        data=inputs_1["pixel_values"]
    )
    multi_modal_data_2 = MultiModalData(
        type=MultiModalData.Type.IMAGE,
        data=inputs_2["pixel_values"]
    )
    multi_modal_data_3 = MultiModalData(
        type=MultiModalData.Type.IMAGE,
        data=inputs_3["pixel_values"]
    )
    sampling_params = SamplingParams(
        temperature=0.5, 
        top_p=0.9,
        top_k=32,
        max_tokens=512
    )
    
    outputs_1 = await llm.add_request(
        inputs={
            # 'prompt': prompt_1,
            'prompt_token_ids': inputs_1["input_ids"].tolist()[0], 
            'multi_modal_data': multi_modal_data_1
        },
        params=sampling_params,
        request_id='1',
    )
    
    outputs_2 = await llm.add_request(
        inputs={
            # 'prompt': prompt_1,
            'prompt_token_ids': inputs_2["input_ids"].tolist()[0], 
            'multi_modal_data': multi_modal_data_2
        },
        params=sampling_params,
        request_id='2',
    )
    
    outputs_3 = await llm.add_request(
        inputs={
            # 'prompt': prompt_1,
            'prompt_token_ids': inputs_3["input_ids"].tolist()[0], 
            'multi_modal_data': multi_modal_data_3
        },
        params=sampling_params,
        request_id='3',
    )
    
    c1 = wait_response([outputs_1])
    c2 = wait_response([outputs_2])
    c3 = wait_response([outputs_3])
    
    t1 = await c1
    t2 = await c2
    t3 = await c3
    print(t1)
    print('---')
    print(t2)
    print('---')
    print(t3)
    print('---')
    
    llm.background_loop.cancel()

if __name__ == "__main__":
    s3_bucket_path = "s3://air-example-data-2/vllm_opensource_llava/"
    local_directory = "images"

    # Make sure the local directory exists or create it
    os.makedirs(local_directory, exist_ok=True)

    # Use AWS CLI to sync the directory, assume anonymous access
    subprocess.check_call([
        "aws",
        "s3",
        "sync",
        s3_bucket_path,
        local_directory,
        "--no-sign-request",
    ])
    
    asyncio.run(run_phi3v())
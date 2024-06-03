import os
import subprocess

from PIL import Image
from transformers import AutoProcessor

from vllm import LLM, SamplingParams
from vllm.sequence import MultiModalData


def run_phi3v():
    model_path = "microsoft/Phi-3-vision-128k-instruct"
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        image_input_type="pixel_values",
        image_token_id=-1,
        image_input_shape="1008, 1344",
        fake_image_input_shape="1, 16, 3, 336, 336",
        image_feature_size=1024,
        max_model_len=128000,
        max_num_seqs=4,
        kv_cache_dtype="fp8_e5m2",
        dtype='half',
        tensor_parallel_size=1,
    )

    image = Image.open("images/stop_sign.jpg")
    user_prompt = '<|user|>\n'
    assistant_prompt = '<|assistant|>\n'
    suffix = "<|end|>\n"

    # single-image prompt
    prompt = "What is shown in this image?"
    prompt = f"{user_prompt}<|image_1|>\n{prompt}{suffix}{assistant_prompt}"
    inputs = processor(prompt, image, return_tensors="pt")
    multi_modal_data = MultiModalData(type=MultiModalData.Type.IMAGE,
                                      data=inputs["pixel_values"])
    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.9,
        top_k=32,
        max_tokens=512
    )
    outputs = llm.generate(
        prompt_token_ids=inputs["input_ids"].tolist(),
        sampling_params=sampling_params,
        multi_modal_data=multi_modal_data
    )
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


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
    run_phi3v()

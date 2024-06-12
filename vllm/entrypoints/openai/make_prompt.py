import re
from io import BytesIO
from typing import List

from PIL import Image

from transformers import PreTrainedTokenizer
from base64 import b64decode

from .protocol import ChatCompletionRequest

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


INTERNLM_XCOMPOSER2_TEMPLATE = (
    "{% for message in messages %}"
    "{{'[UNUSED_TOKEN_146]' + message['role'] + '\n' + message['content'] + '[UNUSED_TOKEN_145]' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '[UNUSED_TOKEN_146]assistant\n' }}"
    "{% endif %}"
)

RE_TGI_IMAGE_MARKDOWN = re.compile(r'!\[([^\]]*)]\(data:([^;]*);base64,([-A-Za-z0-9+/]*={0,3})\)')
RE_OPENAI_IMAGE_URL = re.compile(r'data:([^;]*);base64,([-A-Za-z0-9+/]*={0,3})')


def convert_prompt(conversation: List["ConversationMessage"]):

    # Remove duplicate system message
    if len(conversation) >= 2 and conversation[1]['role'] == 'system':
        conversation = [conversation[0], *conversation[2:]]

    for idx, msg in enumerate(conversation):
        if msg['role'] == 'system' and msg['content'] == "":
            msg['content'] = DEFAULT_SYSTEM_PROMPT

        images = []

        def replace_tgi_image_markdown(match):
            display_text = match.group(1)
            mime_type = match.group(2)
            base64_string = match.group(3)
            return replace_image(mime_type, base64_string)

        def replace_openai_image_url(match):
            mime_type = match.group(1)
            base64_string = match.group(2)
            return replace_image(mime_type, base64_string)

        def replace_image(mime_type, base64_string):
            # Convert base64 to bytesIO
            file = BytesIO(b64decode(base64_string, validate=True))
            # FIXME: Filter out unsupported image types
            if mime_type.startswith('image/'):
                image = Image.open(file)
                if image.size != (16, 16):
                    images.append(image)
                    return ""
                return ""
            else:
                raise ValueError(f"Unsupported mime type: {mime_type}")

        if isinstance(msg['content'], str):
            # Replace TGI image markdown `![](base64)`` with placeholder
            msg['content'] = re.sub(
                RE_TGI_IMAGE_MARKDOWN,
                replace_tgi_image_markdown,
                msg['content'],
            )
            if len(images) > 0:
                msg['content'] = [*images, msg['content']]

    return conversation

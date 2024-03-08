import os
from typing import Optional

from transformers import AutoConfig, PretrainedConfig
from vllm.logger import init_logger

from vllm.transformers_utils.configs import *

_CONFIG_REGISTRY = {
    "aquila": AquilaConfig,
    "baichuan": BaiChuanConfig,
    "chatglm": ChatGLMConfig,
    "mpt": MPTConfig,
    "qwen": QWenConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
    "yi": YiConfig,
}

logger = init_logger(__name__)

# NOTE: For benchmarking
FORCE_SIGNLE_LAYER = 0

def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None
) -> PretrainedConfig:
    global FORCE_SIGNLE_LAYER
    
    try:
        config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision)
    except ValueError as e:
        if (not trust_remote_code and
                "requires you to execute the configuration file" in str(e)):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model, revision=revision)
    
    # NOTE: DEBUG
    if FORCE_SIGNLE_LAYER > 0:
        assert isinstance(FORCE_SIGNLE_LAYER, int)
        config.num_hidden_layers = FORCE_SIGNLE_LAYER
    
    if 'timber' in [os.getenv('PAGED_ATTENTION_BACKEND', 'timber'), os.getenv('PROMPT_ATTENTION_BACKEND', 'timber')] and hasattr(config, 'sliding_window'):
        logger.info(f'sliding window ({config.sliding_window}) disabled -> {config.max_position_embeddings}')
        config.sliding_window = config.max_position_embeddings
    
    return config

"""Attention layer."""
from functools import lru_cache
import os
from typing import List, Literal, Optional

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.input_metadata import InputMetadata
from vllm.utils import is_hip

logger = init_logger(__name__)


class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        layer_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        self.attention_backend = os.getenv('ATTENTION_BACKEND', 'hip')
        
        if self.attention_backend == 'hip':
            from vllm.model_executor.layers.attention.backends.hip import HipAttentionBackend
            self.backend = HipAttentionBackend(
                num_heads, head_size, scale,
                num_kv_heads, alibi_slopes,
                sliding_window, layer_index,
            )
        elif _use_flash_attn():
            from vllm.model_executor.layers.attention.backends.flash_attn import FlashAttentionBackend  # noqa: E501
            self.backend = FlashAttentionBackend(
                num_heads, head_size, scale,
                num_kv_heads, alibi_slopes,
                sliding_window
            )
        else:
            from vllm.model_executor.layers.attention.backends.xformers import XFormersBackend  # noqa: E501
            self.backend = XFormersBackend(
                num_heads, head_size, scale,
                num_kv_heads, alibi_slopes,
                sliding_window
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        
        rope_method: Literal['none', 'self_extend'] = 'none',
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.attention_backend == 'hip':
            return self.backend.forward(
                query,
                key,
                value,
                key_cache,
                value_cache,
                input_metadata,
                
                rope_method,
                rope_cos,
                rope_sin, 
                position_ids,
            )
        else:
            return self.backend.forward(
                query,
                key,
                value,
                key_cache,
                value_cache,
                input_metadata
            )


@lru_cache(maxsize=1)
def _use_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.info("flash_attn is not found. Using xformers backend.")
        return False

    if is_hip():
        # AMD GPUs.
        return False
    if torch.cuda.get_device_capability()[0] < 8:
        # Volta and Turing NVIDIA GPUs.
        logger.info("flash_attn is not supported on Turing or older GPUs. "
                    "Using xformers backend.")
        return False
    if torch.get_default_dtype() not in (torch.float16, torch.bfloat16):
        logger.info(
            "flash_attn only supports torch.float16 or torch.bfloat16. "
            "Using xformers backend.")
        return False

    logger.info("Using flash_attn backend.")
    return True

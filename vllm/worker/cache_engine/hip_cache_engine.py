from typing import List, Tuple
from torch import Tensor
import torch

from .cache_engine import CacheEngine, init_logger, ModelConfig

logger = init_logger(__name__)

KVCache = Tensor

class HipCache:
    pass

# GPU (compressed) / CPU
class HipKeyCache(HipCache):
    def __init__(
        self, 
        model_config: ModelConfig, 
        layer_index: int,
        num_blocks: int,
        block_shape: Tuple[int],
    ):
        self.model_config = model_config
        self.layer_index = layer_index

# CPU
class HipValueCache(HipCache):
    def __init__(
        self, 
        model_config: ModelConfig, 
        layer_index: int,
        num_blocks: int,
        block_shape: Tuple[int],
    ):
        self.model_config = model_config
        self.layer_index = layer_index

HipKVCache = Tuple[HipKeyCache, HipValueCache]

class HipCacheEngine(CacheEngine):
    def allocate_cpu_cache(self) -> List[KVCache]:
        raise Exception('CPU cache is not supported')
    
    def allocate_gpu_cache(self) -> List[HipKVCache]:
        gpu_cache: List[HipKVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for layer_index in range(self.num_layers):
            key_blocks = HipKeyCache(
                self.model_config,
                layer_index,
                self.num_gpu_blocks,
                key_block_shape
            )
            value_blocks = HipValueCache(
                self.model_config,
                layer_index,
                self.num_gpu_blocks,
                value_block_shape
            )
            logger.info(f'layer {layer_index} key: {key_blocks.shape}[{key_blocks.numel():,}] value: {value_blocks.shape}[{value_blocks.numel():,}]; {(self.num_gpu_blocks * 16) // self.model_config.max_model_len}')
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache
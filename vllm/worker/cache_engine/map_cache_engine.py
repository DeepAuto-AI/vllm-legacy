import torch

import numpy as np
import cupy as cp
from typing import List, Tuple
from .cache_engine import CacheConfig, CacheEngine, KVCache, init_logger, ModelConfig, ParallelConfig, STR_DTYPE_TO_TORCH_DTYPE, _get_dtype_size

logger = init_logger(__name__)

class ManagedTensor:
    def __init__(self, shape, dtype, byte_size):
        self.shape = shape
        self.byte_size = byte_size
        self.dtype = dtype
        
        self.managed_ptr = cp.cuda.malloc_managed(self.byte_size) #type: cp.cuda.MemoryPointer
        self.data_ptr = lambda: self.managed_ptr.ptr
        logger.info(f'managed allocated {self.data_ptr:02X}')

    def numel(self):
        size = 1
        for i in self.shape:
            size *= i
        return size

class MapCacheEngine(CacheEngine):
    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            value_blocks = ManagedTensor(
                shape=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                byte_size=_get_dtype_size(self.dtype) * key_blocks.numel(),
            )
            logger.info(f'layer {_} key: {key_blocks.shape}[{key_blocks.numel():,}] value: {value_blocks.shape}[{value_blocks.numel():,}]; {self.num_gpu_blocks // self.model_config.max_model_len}')
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache
    
    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = 0 # stored in CPU always
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total
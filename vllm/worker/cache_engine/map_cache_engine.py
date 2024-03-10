import os
import torch

import numpy as np
import cupy as cp
from typing import List, Tuple
from .cache_engine import CacheConfig, CacheEngine, KVCache, init_logger, ModelConfig, ParallelConfig, STR_DTYPE_TO_TORCH_DTYPE, _get_dtype_size

logger = init_logger(__name__)

def numel(shape):
    size = 1
    for i in shape:
        size *= i
    return size

cudaMemAdviseSetReadMostly = 1
cudaMemAdviseUnsetReadMostly = 2
cudaMemAdviseSetPreferredLocation = 3
cudaMemAdviseUnsetPreferredLocation = 4
cudaMemAdviseSetAccessedBy = 5
cudaMemAdviseUnsetAccessedBy = 6

cudaCpuDeviceId = -1

class ManagedTensor:
    def __init__(self, shape, dtype, byte_size, device):
        self.shape = shape
        self.byte_size = byte_size
        self.dtype = dtype
        self.device = torch.device(device)
        
        self._stride = [1]
        for i in range(len(self.shape) - 1):
            j = len(self.shape) - i - 1
            self._stride.append(self._stride[-1] * self.shape[j])
        self._stride.reverse()
        self._stride = tuple(self._stride)
        self.stride = lambda: self._stride
        
        self.managed_ptr = cp.cuda.malloc_managed(self.byte_size) #type: cp.cuda.MemoryPointer
        self.data_ptr = lambda: self.managed_ptr.ptr
        tensor = cp.ndarray((self.byte_size,), dtype=cp.uint8, memptr=self.managed_ptr)
        tensor[:] = 0
        
        logger.info(f'managed allocated {self.data_ptr():02X} {self.byte_size:,}')

    def readonly_start(self):
        pass
        cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId)
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseSetAccessedBy, 0)
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseSetReadMostly, 0)
    
    def readonly_end(self):
        pass
        cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId)
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseUnsetAccessedBy, 0)
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseUnsetReadMostly, 0)

    def numel(self):
        return numel(self.shape)

class MapCacheEngine(CacheEngine):
    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for layer_index in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            if layer_index < int(os.getenv('HIP_DENSE_LAYERS', '3')):
                value_blocks = torch.empty(
                    size=(self.num_gpu_blocks, *value_block_shape),
                    dtype=self.dtype,
                    device="cuda",
                )
            else:
                value_blocks = ManagedTensor(
                    shape=(self.num_gpu_blocks, *value_block_shape),
                    dtype=self.dtype,
                    byte_size=_get_dtype_size(self.dtype) * numel((self.num_gpu_blocks, *value_block_shape)),
                    device='cuda',
                )
            logger.info(f'layer {layer_index} key: {key_blocks.shape}[{key_blocks.numel():,}] value: {value_blocks.shape}[{value_blocks.numel():,}]; {(self.num_gpu_blocks * 16) // self.model_config.max_model_len}')
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
import math
import os
import time
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
        self.size = lambda x: self.shape[x]
        self.byte_size = byte_size
        self.elem_size = _get_dtype_size(dtype)
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
        cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId)
        tensor = cp.ndarray((self.byte_size,), dtype=cp.uint8, memptr=self.managed_ptr)
        tensor[:] = 0 # touch every page and set on cpu
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseUnsetPreferredLocation, cudaCpuDeviceId)
        self.tensor = tensor
        
        logger.info(f'managed allocated {self.data_ptr():02X} {self.byte_size:,} bytes')

    def decode_start(self):
        cp.cuda.runtime.memAdvise(
            self.data_ptr(), 
            self.byte_size, 
            cudaMemAdviseSetPreferredLocation, 
            cudaCpuDeviceId
        )
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId)
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseSetReadMostly, cudaCpuDeviceId)
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseSetAccessedBy, torch.cuda.current_device())
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseSetReadMostly, torch.cuda.current_device())
    
    def decode_end(self):
        cp.cuda.runtime.memAdvise(
            self.data_ptr(), 
            self.byte_size, 
            cudaMemAdviseUnsetPreferredLocation, 
            cudaCpuDeviceId
        )
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseUnsetAccessedBy, cudaCpuDeviceId)
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseUnsetReadMostly, cudaCpuDeviceId)
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseUnsetAccessedBy, torch.cuda.current_device())
        # cp.cuda.runtime.memAdvise(self.data_ptr(), self.byte_size, cudaMemAdviseUnsetReadMostly, torch.cuda.current_device())
        
        # cp.cuda.runtime.memPrefetchAsync(
        #     self.data_ptr(), 
        #     self.byte_size, 
        #     cudaCpuDeviceId, 
        #     torch.cuda.current_stream().stream_id
        # )
        
        return

    def prompt_start(self, block_indices):
        # print(block_indices)
        PAGE_SIZE = 2**(10+6) # 64KB
        # dump all to GPU uwu. PCIe is cool :>
        # to(0, non_blocking=True)
        
        self.last_prompt_block_indices = block_indices
        block_stride = self.stride()[0]
        base_ptr = self.data_ptr()
        target_device = torch.cuda.current_device()
        stream_id = torch.cuda.current_stream(target_device).stream_id
        num_blocks = self.shape[0]
        
        t = time.time()
        torch.cuda.synchronize(target_device)
        # print('pre-synced', time.time() - t)
        
        # cp.cuda.runtime.memAdvise(
        #     self.data_ptr(), 
        #     self.byte_size, 
        #     cudaMemAdviseSetPreferredLocation, 
        #     target_device,
        #     # cudaCpuDeviceId,
        # )
        # cp.cuda.runtime.memPrefetchAsync(
        #     self.data_ptr(),
        #     self.byte_size,
        #     target_device, 
        #     stream_id
        # )
        
        last_block_ptr = block_ptr = None
        block_ptr_size = 0
        commited = 0
        for i in range(len(block_indices)):
            block_index = block_indices[i].item()
            if block_index >= num_blocks: continue
            if block_index < 0: continue
            
            new_block_ptr = ((base_ptr + block_index * block_stride * self.elem_size) // PAGE_SIZE) * PAGE_SIZE
            new_block_ptr_size = math.ceil((block_stride * self.elem_size) / PAGE_SIZE) * PAGE_SIZE
            # print(block_index, self.shape, block_ptr, block_ptr_size, self.elem_size, flush=True)
            
            if block_ptr is None:
                last_block_ptr = block_ptr = new_block_ptr
                block_ptr_size = new_block_ptr_size
            
            need_flush = False
            if (new_block_ptr - last_block_ptr) == new_block_ptr_size:
                block_ptr_size += new_block_ptr_size
                # print('extend', last_block_ptr, new_block_ptr)
                last_block_ptr = new_block_ptr
            elif new_block_ptr == last_block_ptr:
                # print('aligned')
                pass
            else:
                need_flush = True
            
            if i == (len(block_indices) - 1):
                need_flush = True
            
            if need_flush:
                block_ptr = max(block_ptr, base_ptr)
                left_size = max(0, (base_ptr + self.byte_size) - block_ptr)
                # print('commit', block_ptr, left_size, block_ptr_size, base_ptr, self.byte_size)
                # print('fetch', block_ptr, min(left_size, block_ptr_size))
                cp.cuda.runtime.memAdvise(
                    block_ptr,
                    min(left_size, block_ptr_size),
                    # cudaMemAdviseSetPreferredLocation, 
                    cudaMemAdviseSetAccessedBy,
                    target_device,
                )
                cp.cuda.runtime.memPrefetchAsync(
                    block_ptr,
                    min(left_size, block_ptr_size),
                    target_device, 
                    stream_id
                )
                commited += min(left_size, block_ptr_size)
                if new_block_ptr == last_block_ptr:
                    last_block_ptr = block_ptr = None
                    block_ptr_size = 0
                else:
                    block_ptr = new_block_ptr
                    block_ptr_size = new_block_ptr_size
        
        if commited == PAGE_SIZE:
            print('single page') # this is happend when measure peak memory
            cp.cuda.runtime.memAdvise(
                self.data_ptr(), 
                self.byte_size, 
                # cudaMemAdviseSetPreferredLocation, 
                cudaMemAdviseSetAccessedBy,
                target_device,
            )
            cp.cuda.runtime.memPrefetchAsync(
                self.data_ptr(),
                self.byte_size,
                target_device, 
                stream_id
            )
        
        t = time.time()
        # torch.cuda.synchronize(target_device)
        # print('synced', time.time() - t)
        
        # cp.cuda.runtime.memPrefetchAsync(
        #     self.data_ptr(),
        #     self.byte_size,
        #     target_device, 
        #     stream_id
        # )
        
        self.t_start = time.time()
    
    def prompt_end(self):
        # prepare decoding
        # to(cpu, non_blocking=True)
        # cp.cuda.runtime.memAdvise(
        #     self.data_ptr(), 
        #     self.byte_size, 
        #     cudaMemAdviseSetPreferredLocation, 
        #     cudaCpuDeviceId
        # )
        # print('prompt zone done', time.time() - self.t_start)

        # print(block_indices)
        PAGE_SIZE = 2**(10+6) # 64KB
        # dump all to GPU uwu. PCIe is cool :>
        # to(0, non_blocking=True)
        
        block_indices = self.last_prompt_block_indices
        block_stride = self.stride()[0]
        base_ptr = self.data_ptr()
        target_device = torch.cuda.current_device()
        stream_id = torch.cuda.current_stream(target_device).stream_id
        num_blocks = self.shape[0]
        
        t = time.time()
        torch.cuda.synchronize(target_device)
        # print('pre-synced', time.time() - t)
        
        # cp.cuda.runtime.memAdvise(
        #     self.data_ptr(), 
        #     self.byte_size, 
        #     cudaMemAdviseSetPreferredLocation, 
        #     target_device,
        #     # cudaCpuDeviceId,
        # )
        # cp.cuda.runtime.memPrefetchAsync(
        #     self.data_ptr(),
        #     self.byte_size,
        #     target_device, 
        #     stream_id
        # )
        
        last_block_ptr = block_ptr = None
        block_ptr_size = 0
        commited = 0
        for i in range(len(block_indices)):
            block_index = block_indices[i].item()
            if block_index >= num_blocks: continue
            if block_index < 0: continue
            
            new_block_ptr = ((base_ptr + block_index * block_stride * self.elem_size) // PAGE_SIZE) * PAGE_SIZE
            new_block_ptr_size = math.ceil((block_stride * self.elem_size) / PAGE_SIZE) * PAGE_SIZE
            # print(block_index, self.shape, block_ptr, block_ptr_size, self.elem_size, flush=True)
            
            if block_ptr is None:
                last_block_ptr = block_ptr = new_block_ptr
                block_ptr_size = new_block_ptr_size
            
            need_flush = False
            if (new_block_ptr - last_block_ptr) == new_block_ptr_size:
                block_ptr_size += new_block_ptr_size
                # print('extend', last_block_ptr, new_block_ptr)
                last_block_ptr = new_block_ptr
            elif new_block_ptr == last_block_ptr:
                # print('aligned')
                pass
            else:
                need_flush = True
            
            if i == (len(block_indices) - 1):
                need_flush = True
            
            if need_flush:
                block_ptr = max(block_ptr, base_ptr)
                left_size = max(0, (base_ptr + self.byte_size) - block_ptr)
                # print('commit', block_ptr, left_size, block_ptr_size, base_ptr, self.byte_size)
                # print('fetch', block_ptr, min(left_size, block_ptr_size))
                cp.cuda.runtime.memAdvise(
                    block_ptr,
                    min(left_size, block_ptr_size),
                    # cudaMemAdviseSetPreferredLocation, 
                    cudaMemAdviseUnsetAccessedBy,
                    cudaCpuDeviceId,
                )
                # cp.cuda.runtime.memPrefetchAsync(
                #     block_ptr,
                #     min(left_size, block_ptr_size),
                #     target_device, 
                #     stream_id
                # )
                commited += min(left_size, block_ptr_size)
                if new_block_ptr == last_block_ptr:
                    last_block_ptr = block_ptr = None
                    block_ptr_size = 0
                else:
                    block_ptr = new_block_ptr
                    block_ptr_size = new_block_ptr_size
        
        if commited == PAGE_SIZE:
            print('single page') # this is happend when measure peak memory
            cp.cuda.runtime.memAdvise(
                self.data_ptr(), 
                self.byte_size, 
                # cudaMemAdviseSetPreferredLocation, 
                cudaMemAdviseUnsetAccessedBy,
                cudaCpuDeviceId,
            )
            cp.cuda.runtime.memPrefetchAsync(
                self.data_ptr(),
                self.byte_size,
                target_device, 
                stream_id
            )


    def numel(self):
        return numel(self.shape)

class MapCacheEngine(CacheEngine):
    def __init__(
        self, 
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        k_offload: bool = False,
        v_offload: bool = True,
    ) -> None:
        self.k_offload = k_offload
        self.v_offload = v_offload
        
        super().__init__(cache_config, model_config, parallel_config)
    
    def allocate_gpu_cache(self) -> List[KVCache]:
        num_dense_layer = int(os.getenv('HIP_DENSE_LAYERS', '3'))
        
        gpu_cache: List[KVCache] = []
        
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        
        for layer_index in range(self.num_layers):
            if (layer_index < num_dense_layer) or (not self.k_offload):
                key_blocks = torch.empty(
                    size=(self.num_gpu_blocks, *key_block_shape),
                    dtype=self.dtype,
                    device="cuda",
                )
            else:
                key_blocks = ManagedTensor(
                    shape=(self.num_gpu_blocks, *key_block_shape),
                    dtype=self.dtype,
                    byte_size=_get_dtype_size(self.dtype) * numel((self.num_gpu_blocks, *key_block_shape)),
                    device='cuda',
                )
            
            if (layer_index < num_dense_layer) or (not self.v_offload):
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
            
            logger.info(f'layer {layer_index} key: {key_blocks.shape}[{key_blocks.numel():,}] value: {value_blocks.shape}[{value_blocks.numel():,}]. Expected Max Batch: {(self.num_gpu_blocks * 16) // self.model_config.max_model_len}')
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache
    
    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        k_offload: bool,
        v_offload: bool,
    ) -> int:
        num_dense_layer = int(os.getenv('HIP_DENSE_LAYERS', '3'))
        
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)
        num_cached_layers = num_layers - num_dense_layer

        dense_key_cache_block = key_cache_block = block_size * num_heads * head_size
        dense_value_cache_block = value_cache_block = key_cache_block
        
        if k_offload:
            key_cache_block = 0
        if v_offload:
            value_cache_block = 0
        
        total = num_cached_layers * (key_cache_block + value_cache_block) + num_dense_layer * (dense_key_cache_block + dense_value_cache_block)
        if cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return max(1, dtype_size * total)
        # return 0

class VMapCacheEngine(MapCacheEngine):
    def __init__(
        self, 
        cache_config: CacheConfig, 
        model_config: ModelConfig, 
        parallel_config: ParallelConfig
    ) -> None:
        super().__init__(cache_config, model_config, parallel_config, False, True)
    
    @staticmethod
    def get_cache_block_size(block_size: int, cache_dtype: str, model_config: ModelConfig, parallel_config: ParallelConfig) -> int:
        return MapCacheEngine.get_cache_block_size(block_size, cache_dtype, model_config, parallel_config, False, True)
    
class KVMapCacheEngine(MapCacheEngine):
    def __init__(
        self, 
        cache_config: CacheConfig, 
        model_config: ModelConfig, 
        parallel_config: ParallelConfig
    ) -> None:
        super().__init__(cache_config, model_config, parallel_config, True, True)
    
    @staticmethod
    def get_cache_block_size(block_size: int, cache_dtype: str, model_config: ModelConfig, parallel_config: ParallelConfig) -> int:
        return MapCacheEngine.get_cache_block_size(block_size, cache_dtype, model_config, parallel_config, True, True)
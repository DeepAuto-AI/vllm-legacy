"""Attention layer with Flash and PagedAttention."""
import math
import os
from typing import List, Literal, Optional
import warnings

from flash_attn import flash_attn_func
import torch

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention.ops.paged_attn import (
    PagedAttentionImpl
)
from timber import paged_timber_attention, timber_attention
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalMask, 
    LowerTriangularMaskWithTensorBias
)

class HipAttentionBackend:
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
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        suppored_head_sizes = PagedAttentionImpl.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}."
            )

        self.sliding_window = (
            (self.sliding_window, self.sliding_window) if self.sliding_window is not None else (-1, -1)
        )
        self.layer_index = layer_index
        assert layer_index is not None, 'layer index should be not none'
        
        self.hip_dense_layers = list(range(int(os.environ.get('HIP_DENSE_LAYERS', '3'))))
        self.hip_high_k_layers = {}
        
        self.self_extend_scale = int(os.getenv('SE_SCALE', '8'))
        self.self_extend_window = int(os.getenv('SE_WINDOW', '1024'))
        
        self.checkout_last = False
        self.last_ks = None
        self.last_indices = None
        
        self.using_precomputed_mask = False
        self.precomputed_ks = None
        self.precomputed_indices = None

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
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x, block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size, block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        
        need_prompt_prefetch = hasattr(key_cache, 'prompt_start') or hasattr(value_cache, 'prompt_start')
        need_prompt_prefetch = need_prompt_prefetch and (input_metadata.is_prompt and key_cache is not None and value_cache is not None)
        if need_prompt_prefetch:
            assert not torch.cuda.is_current_stream_capturing()
            # it is okay to use linear cost here
            block_size = value_cache.shape[-1]
            # print(block_size, input_metadata.slot_mapping, input_metadata.slot_mapping.shape)
            block_idx = (input_metadata.slot_mapping.view(-1)[::block_size] // block_size).sort().values.to(
                dtype=torch.int32, 
                device='cpu', 
                non_blocking=True
            )
            # print(block_idx)
            if hasattr(key_cache, 'prompt_start'):
                key_cache.prompt_start(block_idx)
            if hasattr(value_cache, 'prompt_start'):
                value_cache.prompt_start(block_idx)

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            PagedAttentionImpl.reshape_and_cache(
                key, 
                value, 
                key_cache,
                value_cache, 
                input_metadata
            )
            
        hip_k = int(os.environ.get('HIP_K', '1024'))
        
        benchmark_prompt_attention = os.environ.get('BENCHMARK_PAGED_ATTENTION', '0') == '1'
        prompt_backend = os.environ.get('PROMPT_ATTENTION_BACKEND', 'hip')
        
        benchmark_paged_attention = os.environ.get('BENCHMARK_PAGED_ATTENTION', '0') == '1'
        paged_backend = os.environ.get('PAGED_ATTENTION_BACKEND', 'hip')
        
        assert not benchmark_paged_attention, 'droped'
        assert not benchmark_prompt_attention, 'droped'
        assert prompt_backend in ['vllm', 'hip']
        assert paged_backend in ['vllm', 'hip']
        
        if self.layer_index in self.hip_dense_layers:
            prompt_backend = 'vllm'
            paged_backend = 'vllm'
            
            if rope_method == 'self_extend':
                paged_backend = 'hip'
            if rope_method == 'self_extend':
                prompt_backend = 'hip'
        
        if rope_method == 'self_extend':
            assert paged_backend == 'hip'
            assert prompt_backend == 'hip'
        
        has_high_k = (self.layer_index in self.hip_high_k_layers)
        has_self_extend_dense = (rope_method == 'self_extend' and self.layer_index in self.hip_dense_layers)
        if (paged_backend == 'hip' or prompt_backend == 'hip') and (has_high_k or has_self_extend_dense):
            if self.layer_index in self.hip_high_k_layers:
                hip_k *= self.hip_high_k_layers[self.layer_index]
            else:
                hip_k *= 2

        if input_metadata.is_prompt:
            is_normal_prompt = (key_cache is None or value_cache is None or input_metadata.block_tables.numel() == 0)
            # Prompt run.
            if is_normal_prompt:
                # normal attention [N, T, H, HID]
                if prompt_backend == 'vllm':
                    assert rope_method in ['none']
                    query = query.unflatten(0, (batch_size, seq_len))
                    key = key.unflatten(0, (batch_size, seq_len))
                    value = value.unflatten(0, (batch_size, seq_len))
                    
                    output = flash_attn_func(
                        query,
                        key,
                        value,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=self.sliding_window,
                        alibi_slopes=self.alibi_slopes,
                    )
                elif prompt_backend == 'hip':
                    assert rope_method in ['none', 'self_extend']
                    
                    warnings.warn('prompt attention backend is hip')
                    
                    N = batch_size
                    N_TDST, H, HID = query.shape
                    N_TSRC, H_KV, _HID = key.shape
                    assert key.shape[:-1] == value.shape[:-1]
                    assert HID == _HID
                    
                    TDST = N_TDST // N
                    TSRC = N_TSRC // N
                    
                    query = query.view(batch_size, TDST, H, HID)
                    key = key.view(batch_size, TSRC, H_KV, HID)
                    value = value.view(batch_size, TSRC, H_KV, HID)
                    
                    query = query.permute(0, 2, 1, 3).reshape(-1, TDST, HID)
                    key = key.permute(0, 2, 1, 3).reshape(-1, TSRC, HID)
                    value = value.permute(0, 2, 1, 3).reshape(-1, TSRC, HID)
                    
                    assert (input_metadata.attn_bias is None) or isinstance(input_metadata.attn_bias, BlockDiagonalCausalMask), f'{input_metadata.attn_bias}'
                    assert self.alibi_slopes is None
                    
                    output, _ = timber_attention(
                        q=query * self.scale,
                        k=key,
                        v=value,
                        attention_mask=None,
                        
                        mask_k=hip_k,
                        block_size_q=32,
                        block_size_k=2,
                        
                        dense_queries_exp=None if rope_method == 'none' else 0,
                        
                        rope_method=rope_method,
                        rope_cos=rope_cos,
                        rope_sin=rope_sin,
                        position_ids=position_ids.repeat_interleave(self.num_heads, 0) if rope_method != 'none' else None,
                        
                        self_extend_scale=self.self_extend_scale,
                        self_extend_window=self.self_extend_window,
                    )
                    
                    output = output.view(
                        batch_size,
                        H,
                        TDST,
                        HID,
                    )
                    output = output.permute(0, 2, 1, 3).reshape(
                        batch_size,
                        seq_len,
                        hidden_size,
                    )
                else:
                    raise Exception()
            else:
                # prefix-enabled attention
                output = PagedAttentionImpl.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    input_metadata,
                    self.alibi_slopes,
                )
        else:
            # Decoding run.
            if paged_backend == 'vllm':
                output = PagedAttentionImpl.forward_decode(
                    query,
                    key_cache,
                    value_cache,
                    input_metadata,
                    self.num_kv_heads,
                    self.scale,
                    self.alibi_slopes,
                )
            elif paged_backend == 'hip':
                assert rope_method in ['none', 'self_extend']
                
                warnings.warn('paged attention backend is hip')
                
                output, (indices, ks, _) = paged_timber_attention(
                    q=query,
                    q_scale=self.scale,
                    k=key_cache,
                    v=value_cache,
                    attention_mask=None,
                    
                    block_tables=input_metadata.block_tables,
                    context_lens=input_metadata.context_lens,
                    max_context_len=input_metadata.max_context_len,
                    
                    mask_k=hip_k,
                    block_size_q=32,
                    block_size_k=2,
                    
                    rope_method=rope_method,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    position_ids=position_ids.repeat_interleave(self.num_heads, 0) if rope_method != 'none' else None,
                    
                    self_extend_scale=self.self_extend_scale,
                    self_extend_window=self.self_extend_window,
                    
                    using_precomputed_mask=self.using_precomputed_mask,
                    precomputed_ks=self.precomputed_ks,
                    precomputed_indices=self.precomputed_indices,
                )
                
                if self.checkout_last:
                    self.last_ks = ks
                    self.last_indices = indices
                
                N_H, _, HID = output.shape
                N = query.shape[0]
                H = N_H // N
                output = output.view(N, H, HID)

        if input_metadata.is_prompt and key_cache is not None and value_cache is not None:
            assert not torch.cuda.is_current_stream_capturing()
            
            if hasattr(key_cache, 'prompt_end'):
                key_cache.prompt_end()
            if hasattr(value_cache, 'prompt_end'):
                value_cache.prompt_end()
        
        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)

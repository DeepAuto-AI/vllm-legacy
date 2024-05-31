"""Attention layer with Flash and PagedAttention."""
import math
from dataclasses import dataclass
import os
from typing import List, Literal, Optional, Dict, Any, Tuple, Type
import warnings

import torch
from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from vllm._C import cache_ops

from vllm.attention.backends.abstract import (
    AttentionBackend, 
    AttentionImpl,
    AttentionMetadata
)
from vllm.attention.ops.paged_attn import (
    PagedAttention
)
from timber import paged_timber_attention, timber_attention

@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int]
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor]
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor]
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    _cached_prefill_metadata: Optional["FlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["FlashAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.query_start_loc is not None
        assert self.context_lens_tensor is not None
        assert self.block_tables is not None
        assert self.seq_start_loc is not None

        self._cached_prefill_metadata = FlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1],
            seq_start_loc=self.seq_start_loc[:self.num_prefills + 1],
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills],
            block_tables=self.block_tables[:self.num_prefills],
            use_cuda_graph=False,
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            max_query_len=None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills:],
            use_cuda_graph=self.use_cuda_graph,
        )
        return self._cached_decode_metadata

class HiPAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[Dict[str, Any]] = None,
        layer_index: Optional[int] = None,
    ) -> None:
        assert blocksparse_params is None, ValueError(
            "HiP does not support block-sparse attention.")
        
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
        suppored_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}."
            )

        self.sliding_window = (
            (self.sliding_window, self.sliding_window) 
            if self.sliding_window is not None else 
            (-1, -1)
        )
        self.kv_cache_dtype = kv_cache_dtype
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
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert kv_scale == 1.0, "kv_scale is not supported in FlashAttention."

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            cache_ops.reshape_and_cache_flash(
                key,
                value,
                
                str(key_cache.data_ptr()),
                key_cache.size(1),
                key_cache.stride(0),
                str(value_cache.data_ptr()),
                value_cache.stride(0),
                
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
            )

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_tokens]
        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (kv_cache is None or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                out = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                )
                assert output[:num_prefill_tokens].shape == out.shape
                output[:num_prefill_tokens] = out
            else:
                # prefix-enabled attention
                assert prefill_meta.seq_lens is not None
                max_seq_len = max(prefill_meta.seq_lens)
                output[:num_prefill_tokens] = flash_attn_varlen_func(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=prefill_meta.query_start_loc,
                    max_seqlen_q=prefill_meta.max_query_len,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_k=max_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=self.alibi_slopes,
                    block_table=prefill_meta.block_tables,
                )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output[num_prefill_tokens:] = flash_attn_with_kvcache(
                decode_query.unsqueeze(1),
                key_cache,
                value_cache,
                block_table=decode_meta.block_tables,
                cache_seqlens=decode_meta.seq_lens_tensor,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
            ).squeeze(1)

        # Reshape the output tensor.
        return output.view(num_tokens, hidden_size)

    def forward_old(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        kv_scale: float = 1.0,
        
        rope_method: Literal['none', 'self_extend'] = 'none',
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        print(query.shape, attn_metadata)
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        
        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, 
                self.num_kv_heads, 
                self.head_size
            )
        
            need_prompt_prefetch = hasattr(key_cache, 'prompt_start') or hasattr(value_cache, 'prompt_start')
            need_prompt_prefetch = need_prompt_prefetch and (attn_metadata.is_prompt and key_cache is not None and value_cache is not None)
            if need_prompt_prefetch:
                assert not torch.cuda.is_current_stream_capturing()
                # it is okay to use linear cost here
                block_size = value_cache.shape[-1]
                # print(block_size, input_metadata.slot_mapping, input_metadata.slot_mapping.shape)
                block_idx = (attn_metadata.slot_mapping.view(-1)[::block_size] // block_size).sort().values.to(
                    dtype=torch.int32, 
                    device='cpu', 
                    non_blocking=True
                )
                # print(block_idx)
                if hasattr(key_cache, 'prompt_start'):
                    key_cache.prompt_start(block_idx)
                if hasattr(value_cache, 'prompt_start'):
                    value_cache.prompt_start(block_idx)
            
            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory profiling run.
            PagedAttention.write_to_paged_cache(
                key, 
                value, 
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype, 
                kv_scale
            )
            
        hip_k = int(os.environ.get('HIP_K', '1024'))
        
        prompt_backend = os.environ.get('PROMPT_ATTENTION_BACKEND', 'hip')
        paged_backend = os.environ.get('PAGED_ATTENTION_BACKEND', 'hip')
        
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

        if attn_metadata.is_prompt:
            is_normal_prompt = (key_cache is None or value_cache is None or attn_metadata.block_tables.numel() == 0)
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
                    
                    assert (attn_metadata.attn_bias is None) or isinstance(attn_metadata.attn_bias, BlockDiagonalCausalMask), f'{attn_metadata.attn_bias}'
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
                output = PagedAttention.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata,
                    self.alibi_slopes,
                )
        else:
            # Decoding run.
            if paged_backend == 'vllm':
                output = PagedAttention.forward_decode(
                    query,
                    key_cache,
                    value_cache,
                    attn_metadata,
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
                    
                    block_tables=attn_metadata.block_tables,
                    context_lens=attn_metadata.context_lens,
                    max_context_len=attn_metadata.max_context_len,
                    
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

        if attn_metadata.is_prompt and key_cache is not None and value_cache is not None:
            assert not torch.cuda.is_current_stream_capturing()
            
            if hasattr(key_cache, 'prompt_end'):
                key_cache.prompt_end()
            if hasattr(value_cache, 'prompt_end'):
                value_cache.prompt_end()
        
        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)

class HiPAttentionBackend(AttentionBackend):
    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "flash-attn"

    @staticmethod
    def get_impl_cls() -> Type["HiPAttentionImpl"]:
        return HiPAttentionImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "FlashAttentionMetadata":
        return FlashAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dists)
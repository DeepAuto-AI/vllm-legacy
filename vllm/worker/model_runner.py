import os
import time
import warnings
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.attention.backends.hip import HiPAttentionBackend, HiPAttentionImpl
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.distributed import broadcast_tensor_dict
from vllm.distributed.communication_op import graph_capture
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingParams
from vllm.sequence import (MultiModalData, SamplerOutput, SequenceData,
                           SequenceGroupMetadata)
from vllm.utils import (CudaMemoryProfiler, get_kv_cache_torch_dtype, is_hip,
                        is_pin_memory_available, make_tensor_with_pad)
from vllm.attention import Attention
from vllm.attention.backends.hip import HiPAttentionBackend

logger = init_logger(__name__)

_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
_BATCH_SIZE_ALIGNMENT = 8
# Capture graphs for token size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 33)
]


class ModelInput(NamedTuple):
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: Optional[AttentionMetadata]
    seq_lens: List[int]
    query_lens: List[int]
    lora_mapping: Optional[LoRAMapping]
    lora_requests: Set[LoRARequest]
    multi_modal_input: Optional[torch.Tensor]
    slot_mapping: torch.Tensor
    num_prefill_tokens: int
    num_decode_tokens: int
    num_prefills: int

    @classmethod
    def empty(cls, device):
        return ModelInput(
            input_tokens=torch.empty(0, device=device),
            input_positions=torch.empty(0, device=device),
            attn_metadata=None,
            seq_lens=[],
            query_lens=[],
            lora_mapping=None,
            lora_requests=set(),
            multi_modal_input=None,
            slot_mapping=torch.empty(0, device=device),
            num_prefill_tokens=0,
            num_decode_tokens=0,
            num_prefills=0,
        )

class MetricTracer:
    def __init__(self, n_warmup = 0):
        self.x_sum = 0
        self.x_count = 0
        self.n_warmup = n_warmup
    
    def update(self, x):
        if self.n_warmup > 0:
            self.n_warmup -= 1
            return 0
        else:
            self.x_sum += x
            self.x_count += 1
            return self.avg()
    
    def avg(self):
        if self.n_warmup > 0:
            return 0
        return self.x_sum / self.x_count

class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        vision_language_config: Optional[VisionLanguageConfig] = None,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        self.vision_language_config = vision_language_config

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_seq_len_to_capture = self.model_config.max_seq_len_to_capture
        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool: Optional[Tuple[
            int, int]] = None  # Set during graph capture.
        # When using CUDA graph, the input block tables must be padded to
        # max_seq_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), self.get_max_block_per_batch()),
            dtype=np.int32)
        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
        )

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        # Set if the backend is flashinfer.
        self.flashinfer_workspace_buffer: torch.Tensor
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None
        
        self.hip_refresh_interval = int(os.getenv('HIP_REFRESH_INTERVAL', '8'))
        self.is_hip_attention = self.attn_backend == HiPAttentionBackend
        
        if not self.is_hip_attention:
            self.hip_refresh_interval = 1
        
        self.metrics = {
            True: { # is prompt
                'model': MetricTracer(),
            },
            False: { # is decoding
                'model': MetricTracer(n_warmup=16),
            }
        }

    def load_model(self) -> None:
        with CudaMemoryProfiler() as m:
            self.model = get_model(
                model_config=self.model_config,
                device_config=self.device_config,
                load_config=self.load_config,
                lora_config=self.lora_config,
                vision_language_config=self.vision_language_config,
                parallel_config=self.parallel_config,
                scheduler_config=self.scheduler_config,
                cache_config=self.cache_config,
            )

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

        if self.lora_config:
            assert hasattr(self.model, "supported_lora_modules"
                           ) and self.model.supported_lora_modules, (
                               "Model does not support LoRA")
            assert hasattr(
                self.model,
                "embedding_modules"), "Model does not have embedding_modules"
            assert hasattr(self.model, "embedding_padding_modules"
                           ), "Model does not have embedding_padding_modules"
            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
                self.vocab_size,
                self.lora_config,
                self.device,
                self.model.embedding_modules,
                self.model.embedding_padding_modules,
                max_position_embeddings=self.model.config.
                max_position_embeddings,
            )
            self.model = self.lora_manager.create_lora_manager(self.model)

        if self.kv_cache_dtype == "fp8" and is_hip():
            # Currently only ROCm accepts kv-cache scaling factors
            # via quantization_param_path and this will be deprecated
            # in the future.
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    warnings.warn(
                        "Loading kv cache scaling factor from JSON is "
                        "deprecated and will be removed. Please include "
                        "kv cache scaling factors in the model checkpoint.",
                        FutureWarning,
                        stacklevel=2)
                    self.model.load_kv_cache_scales(
                        self.model_config.quantization_param_path)
                    logger.info("Loaded KV cache scaling factors from %s",
                                self.model_config.quantization_param_path)
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__)
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!")

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader.loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_seq_len_to_capture + block_size - 1) // block_size

    def _prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> ModelInput:
        """Prepare the model input based on a given sequence group.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        If cuda graph is required, this API automatically pads inputs.
        """
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        seq_lens: List[int] = []
        prefill_seq_lens: List[int] = []
        decode_seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        block_tables: List[List[int]] = []
        multi_modal_input_list: List[torch.Tensor] = []
        decode_only = True
        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = 0

        # The following fields are only for flashinfer
        # Please follow https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout
        # for the precise definition of the following fields.
        # An example:
        # request 1, page indices [0, 5, 8]
        # request 2, page indices [1, 6, 7]
        # request 3, page indices [3, 4]
        # paged_kv_indices is a concatenation of page indices of all requests:
        # [0, 5, 8, 1, 6, 7, 3, 4]
        # paged_kv_indptr is used to index into paged_kv_indices:
        # [0, 3, 6, 8]
        paged_kv_indices: List[int] = []
        # 0 at the beginning of paged_kv_indptr indicates the start of the
        # first request’s page indices in the paged_kv_indices list.
        paged_kv_indptr: List[int] = [0]
        # paged_kv_last_page_len is the length of the last page of each request
        paged_kv_last_page_len: List[int] = []

        if len(seq_group_metadata_list) == 0:
            return ModelInput.empty(self.device)

        if self.sliding_window is not None:
            sliding_window_blocks = (self.sliding_window + self.block_size -
                                     1) // self.block_size
            block_aligned_sliding_window = \
                sliding_window_blocks * self.block_size

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            is_prompt = seq_group_metadata.is_prompt

            for seq_id in seq_ids:
                computed_block_nums = seq_group_metadata.computed_block_nums
                if (self.scheduler_config is not None
                        and self.scheduler_config.chunked_prefill_enabled
                        and not (computed_block_nums is None
                                 or computed_block_nums == [])):
                    raise RuntimeError(
                        "chunked prefill cannot be used with prefix caching "
                        "now.")

                seq_data = seq_group_metadata.seq_data[seq_id]
                if is_prompt:
                    context_len = seq_data.get_num_computed_tokens()
                else:
                    # get_num_computed_tokens is incorrect for spec decoding.
                    # So, we should have a special logic here.
                    # TODO(sang): Fix it.
                    context_len = seq_data.get_len() - 1

                seq_len = min(
                    seq_data.get_len(),
                    context_len + seq_group_metadata.token_chunk_size)
                if is_prompt:
                    tokens = seq_data.get_token_ids()[context_len:seq_len]
                else:
                    # Optimization. get_token_ids requires the entire copy of
                    # tokens.
                    tokens = [seq_data.get_last_token_id()]

                # Prefix cache was hit.
                # Prefix is not supported with sliding_window
                prefix_cache_hit = (computed_block_nums is not None
                                    and len(computed_block_nums) > 0
                                    and self.sliding_window is None
                                    and is_prompt)

                # These are seq_len/context_len capped to the sliding window.
                # They are passed to decode kernel.
                # We still need original seq_len/context_len to compute slot
                # mapping (and input position) below.
                curr_sliding_window_blocks = None
                sliding_seq_len = seq_len
                sliding_context_len = context_len

                # TODO(sang): This is a hack to make sliding window work with
                # paged attn. We can remove it if we make paged attn kernel
                # to properly handle slinding window attn.
                if (self.sliding_window is not None and not is_prompt):
                    curr_sliding_window_blocks = sliding_window_blocks
                    if self.scheduler_config.use_v2_block_manager:
                        # number of elements in last block
                        suff_len = seq_len % self.block_size
                        sliding_seq_len = min(
                            seq_len, block_aligned_sliding_window + suff_len)
                        if suff_len > 0:
                            curr_sliding_window_blocks += 1
                    else:
                        sliding_seq_len = min(seq_len, self.sliding_window)
                    sliding_context_len = sliding_seq_len - 1

                # TODO(sang): Combine chunked prefill and prefix caching by
                # only allowing multiple of block_size chunk size.
                # NOTE: This only works for oooooooxxx style attention.
                if prefix_cache_hit:
                    assert computed_block_nums is not None
                    context_len = len(computed_block_nums) * self.block_size
                    tokens = tokens[context_len:]

                    # need to think what to set it to when we have both sliding
                    # window and prefix caching...
                    assert self.sliding_window is None, \
                        "Prefix caching is not supported with sliding window"
                    sliding_context_len = context_len

                    if self.attn_backend.get_name() == "flash-attn":
                        # NOTE(woosuk): For flash-attn, the block table should
                        # include the entries for the incoming prefill tokens.
                        # TODO(woosuk): This is a temporary fix. We should
                        # provide a unified interface for different backends.
                        block_table = seq_group_metadata.block_tables[seq_id]
                    else:
                        block_table = computed_block_nums
                elif (self.scheduler_config.chunked_prefill_enabled
                      or not is_prompt):
                    if seq_group_metadata.block_tables is not None:
                        # chunked prefill or decode
                        block_table = seq_group_metadata.block_tables[seq_id]
                        if curr_sliding_window_blocks is not None:
                            block_table = block_table[
                                -curr_sliding_window_blocks:]
                        if self.attn_backend.get_name() == "flashinfer":
                            paged_kv_indices.extend(block_table)
                            paged_kv_indptr.append(paged_kv_indptr[-1] +
                                                   len(block_table))
                            last_page_len = seq_data.get_len(
                            ) % self.block_size
                            if last_page_len == 0:
                                last_page_len = self.block_size
                            paged_kv_last_page_len.append(last_page_len)
                    else:
                        # Only happens when memory profiling runs.
                        block_table = []
                else:
                    # Prefill without chunked prefill or memory profiling.
                    block_table = []
                block_tables.append(block_table)

                seq_lens.append(sliding_seq_len)
                context_lens.append(sliding_context_len)
                query_len = sliding_seq_len - sliding_context_len
                query_lens.append(query_len)
                input_tokens.extend(tokens)
                input_positions.extend(list(range(context_len, seq_len)))
                lora_id = seq_group_metadata.lora_int_id

                if is_prompt:
                    assert len(seq_ids) == 1
                    num_prefills += 1
                    num_prefill_tokens += len(tokens)
                    decode_only = False
                    prefill_seq_lens.append(seq_len)
                else:
                    assert query_len == 1, (
                        "seq_len: {}, context_len: {}, query_len: {}".format(
                            seq_len, context_len, query_len))
                    num_decode_tokens += query_len
                    decode_seq_lens.append(sliding_seq_len)

                if lora_id > 0:
                    lora_requests.add(seq_group_metadata.lora_request)

                lora_index_mapping += [lora_id] * query_len
                lora_prompt_mapping.extend(
                    [lora_id] *
                    (query_len if seq_group_metadata.sampling_params
                     and seq_group_metadata.sampling_params.prompt_logprobs
                     else 1))

                if seq_group_metadata.multi_modal_data:
                    multi_modal_input_list.append(
                        seq_group_metadata.multi_modal_data.data)

                if _is_block_tables_empty(seq_group_metadata.block_tables):
                    # During memory profiling, the block tables are not
                    # initialized yet. In this case, we just use a dummy
                    # slot mapping.
                    # In embeddings, the block tables are {seq_id: None}.
                    slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
                    continue

                # Compute the slot mapping.
                block_table = seq_group_metadata.block_tables[seq_id]

                # Mask the [0, start_idx) tokens of the prompt with
                # _PAD_SLOT_ID, where start_idx is max(0, seq_len -
                # sliding_window). For example, if the prompt len is 10,
                # sliding window is 8, and block size is 4, the first two
                # tokens are masked and the slot mapping will be
                # [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
                start_idx = 0
                if self.sliding_window is not None:
                    if is_prompt:
                        assert self.scheduler_config.use_v2_block_manager \
                            or context_len == 0, (
                            "Prefix caching is currently not supported with "
                            "sliding window attention in V1 block manager")
                    # It is an optimization. When it is decoding, it is always
                    # 0. When prefill, we use it to not write slots to kv cache
                    # to save memory.
                    start_idx = max(0, query_len - self.sliding_window)

                for i in range(context_len, seq_len):
                    if i < start_idx:
                        slot_mapping.append(_PAD_SLOT_ID)
                        continue

                    block_number = block_table[i // self.block_size]
                    block_offset = i % self.block_size
                    slot = block_number * self.block_size + block_offset
                    slot_mapping.append(slot)

        batch_size = len(input_tokens)
        max_query_len = max(query_lens)
        max_prefill_seq_len = max(prefill_seq_lens, default=0)
        max_decode_seq_len = max(decode_seq_lens, default=0)

        # If cuda graph can be used, pad tensors accordingly.
        # See `capture_model` API for more details.
        # vLLM uses cuda graph only for decoding requests.
        use_captured_graph = (
            decode_only and not self.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_decode_seq_len <= self.max_seq_len_to_capture)
        if use_captured_graph:
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append(0)
                input_positions.append(0)
                slot_mapping.append(_PAD_SLOT_ID)
                seq_lens.append(1)
                block_tables.append([])
                lora_index_mapping.append(0)
            batch_size = graph_batch_size
            num_decode_tokens = batch_size

        if use_captured_graph:
            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device=self.device)
        else:
            max_block_table_len = max(
                len(block_table) for block_table in block_tables)
            block_tables = make_tensor_with_pad(
                block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device=self.device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device=self.device)

        if multi_modal_input_list:
            assert self.vision_language_config, (
                "Multi-modal inputs are only supported by "
                "vision language models."
            )
            multi_modal_input = torch.cat(
                multi_modal_input_list, dim=0
            ).to(self.device)
        else:
            multi_modal_input = None

        query_lens_tensor = torch.tensor(query_lens,
                                         dtype=torch.long,
                                         device=self.device)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=self.device)

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)

        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device=self.device)
        input_positions_tensor = torch.tensor(input_positions,
                                              dtype=torch.long,
                                              device=self.device)
        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.long,
                                           device=self.device)

        if self.attn_backend.get_name() == "flashinfer":
            if not hasattr(self, "flashinfer_workspace_buffer"):
                # Allocate 16MB workspace buffer
                # Follow the example of flashinfer: https://docs.flashinfer.ai/api/python/decode.html
                self.flashinfer_workspace_buffer = torch.empty(
                    16 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            paged_kv_indptr_tensor = torch.tensor(paged_kv_indptr,
                                                  dtype=torch.int,
                                                  device=self.device)
            paged_kv_indices_tensor = torch.tensor(paged_kv_indices,
                                                   dtype=torch.int,
                                                   device=self.device)
            paged_kv_last_page_len_tensor = torch.tensor(
                paged_kv_last_page_len, dtype=torch.int, device=self.device)
            kv_cache_dtype = get_kv_cache_torch_dtype(self.kv_cache_dtype,
                                                      self.model_config.dtype)
            attn_metadata = self.attn_backend.make_metadata(
                num_prefills=num_prefills,
                slot_mapping=slot_mapping_tensor,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                use_cuda_graph=False,
                max_prefill_seq_len=max_prefill_seq_len,
                block_tables=block_tables,
                workspace_buffer=self.flashinfer_workspace_buffer,
                paged_kv_indptr=paged_kv_indptr_tensor,
                paged_kv_indices=paged_kv_indices_tensor,
                paged_kv_last_page_len=paged_kv_last_page_len_tensor,
                num_qo_heads=self.model_config.get_num_attention_heads(
                    self.parallel_config),
                num_kv_heads=self.model_config.get_num_kv_heads(
                    self.parallel_config),
                head_dim=self.model_config.get_head_size(),
                page_size=16,
                seq_start_loc=seq_start_loc,
                data_type=kv_cache_dtype)
        else:
            attn_metadata = self.attn_backend.make_metadata(
                num_prefills=num_prefills,
                slot_mapping=slot_mapping_tensor,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                seq_lens=seq_lens,
                seq_lens_tensor=seq_lens_tensor,
                max_query_len=max_query_len,
                max_prefill_seq_len=max_prefill_seq_len,
                max_decode_seq_len=max_decode_seq_len,
                query_start_loc=query_start_loc,
                seq_start_loc=seq_start_loc,
                context_lens_tensor=context_lens_tensor,
                block_tables=block_tables,
                use_cuda_graph=use_captured_graph,
            )

        if self.lora_config:
            lora_mapping = LoRAMapping(
                lora_index_mapping,
                lora_prompt_mapping,
            )
        else:
            lora_mapping = None

        return ModelInput(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
            multi_modal_input=multi_modal_input,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
        )

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata,
               Set[LoRARequest], LoRAMapping, torch.Tensor]:
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            # Prepare input tensors.
            (
                input_tokens,
                input_positions,
                attn_metadata,
                seq_lens,
                query_lens,
                lora_mapping,
                lora_requests,
                multi_modal_input,
                slot_mapping,
                num_prefill_tokens,
                num_decode_tokens,
                num_prefills,
            ) = self._prepare_model_input(seq_group_metadata_list)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, seq_lens, query_lens, self.device,
                self.pin_memory)

            metadata_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "selected_token_indices":
                sampling_metadata.selected_token_indices,
                "lora_requests": lora_requests,
                "lora_mapping": lora_mapping,
                "multi_modal_input": multi_modal_input,
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "slot_mapping": slot_mapping,
                "num_prefills": num_prefills,
            }
            if attn_metadata:
                metadata_dict.update(attn_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            selected_token_indices = metadata_dict.pop(
                "selected_token_indices")
            lora_mapping = metadata_dict.pop("lora_mapping")
            lora_requests = metadata_dict.pop("lora_requests")
            multi_modal_input = metadata_dict.pop("multi_modal_input")
            if metadata_dict:
                attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            else:
                attn_metadata = None
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                num_prompts=0,
            )

        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, lora_requests, lora_mapping,
                multi_modal_input)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        BENCHMARK_RUNNER = os.environ.get('BENCHMARK_RUNNER', '0') == '1'
        
        if BENCHMARK_RUNNER:
            t_start = time.time()
            
            start_prepare = torch.cuda.Event(enable_timing=True)
            end_prepare = torch.cuda.Event(enable_timing=True)
            
            start_model = torch.cuda.Event(enable_timing=True)
            end_model = torch.cuda.Event(enable_timing=True)
            
            start_sample = torch.cuda.Event(enable_timing=True)
            end_sample = torch.cuda.Event(enable_timing=True)
        
        # if BENCHMARK_RUNNER:
        #     torch.cuda.synchronize()
        
        if BENCHMARK_RUNNER: start_prepare.record()
        (
            input_tokens,
            input_positions, 
            attn_metadata, 
            sampling_metadata, 
            lora_requests,
            lora_mapping,
            multi_modal_input,
        ) = self.prepare_input_tensors(seq_group_metadata_list)
        
        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)
        
        if BENCHMARK_RUNNER: end_prepare.record()
        
        # if BENCHMARK_RUNNER:
        #     torch.cuda.synchronize()

        if BENCHMARK_RUNNER: start_model.record()
        # Execute the model.
        is_prompt = input_tokens.shape[-1] > 1
        
        # notify decoding
        # prompt notification is in HiPAttentionBackend
        for kv_cache in kv_caches:
            if kv_cache is not None and not is_prompt:
                k_cache, v_cache = kv_cache
                
                if hasattr(k_cache, 'decode_start'):
                    k_cache.decode_start()
                if hasattr(v_cache, 'decode_start'):
                    v_cache.decode_start()
        
        prefill_meta = attn_metadata.prefill_metadata
        decode_meta = attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
            if isinstance(model_executable, CUDAGraphRunner):
                pass
            elif isinstance(model_executable, HipGraphRunnerCounter):
                model_executable = model_executable.get_next_graph()
            else:
                raise Exception()
        else:
            # we counter the prompt
            model_executable = self.model
            # reset cuda graphs caches
            for runner in self.graph_runners.values():
                if isinstance(runner, CUDAGraphRunner):
                    runner.need_offload_prefetch = True
                elif isinstance(runner, HipGraphRunnerCounter):
                    runner.counter = 0
                    runner.graph_checkout.need_offload_prefetch = True
                    runner.graph_precomputed_mask.need_offload_prefetch = True
                else:
                    raise Exception()
        
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }
        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})
        print('input', input_tokens.shape)
        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)
        
        for kv_cache in kv_caches:
            if kv_cache is not None and not is_prompt:
                k_cache, v_cache = kv_cache
                
                if hasattr(k_cache, 'decode_end'):
                    k_cache.decode_end()
                if hasattr(v_cache, 'decode_end'):
                    v_cache.decode_end()
        
        if BENCHMARK_RUNNER: end_model.record()

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return None

        # Sample the next token.
        if BENCHMARK_RUNNER: start_sample.record()
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        if BENCHMARK_RUNNER: end_sample.record()
        
        if BENCHMARK_RUNNER:
            torch.cuda.synchronize()
            elapsed_prepare = start_prepare.elapsed_time(end_prepare)
            elapsed_model = start_model.elapsed_time(end_model)
            elapsed_sample = start_sample.elapsed_time(end_sample)
            elapsed_total = (time.time() - t_start) * 1000
            
            if seq_group_metadata_list is not None:
                # in main process
                print(f'[{time.time() * 1000:.3f}][{len(seq_group_metadata_list) if seq_group_metadata_list is not None else None}](is_prompt: {is_prompt}) prepare: {elapsed_prepare:.3f}, model: {elapsed_model:.3f}({self.metrics[is_prompt]["model"].update(elapsed_model):.3f}), sample: {elapsed_sample:.3f}, total: {elapsed_total:.3f}')
        
        return output

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests = []
        dummy_lora_requests_per_seq = []
        if self.lora_config:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_local_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                     rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for vision encoding, which needs
        # to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        if self.vision_language_config:
            max_num_seqs = min(
                max_num_seqs,
                int(max_num_batched_tokens /
                    self.vision_language_config.image_feature_size))
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data, fake_multi_modal_input = _prepare_fake_inputs(
                seq_len, self.vision_language_config)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=fake_multi_modal_input,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return

    def remove_all_loras(self):
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.remove_all_loras()

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_loras(lora_requests, lora_mapping)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_loras()

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[torch.Tensor]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")
        start_time = time.perf_counter()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        seq_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()

        graph_batch_size = _get_graph_batch_size(
            self.scheduler_config.max_num_seqs)
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]

        logger.info(f'graph will capture batches {batch_size_capture_list}')
        with graph_capture() as graph_capture_context:
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            for batch_size in reversed(batch_size_capture_list):
                # Create dummy attn_metadata.
                attn_metadata = self.attn_backend.make_metadata(
                    num_prefills=0,
                    num_prefill_tokens=0,
                    num_decode_tokens=batch_size,
                    slot_mapping=slot_mapping[:batch_size],
                    seq_lens=None,
                    seq_lens_tensor=seq_lens[:batch_size],
                    max_query_len=None,
                    max_prefill_seq_len=0,
                    max_decode_seq_len=self.max_seq_len_to_capture,
                    query_start_loc=None,
                    seq_start_loc=None,
                    context_lens_tensor=None,
                    block_tables=block_tables[:batch_size],
                    use_cuda_graph=True,
                )

                if self.lora_config:
                    lora_mapping = LoRAMapping(
                        [0] * batch_size,
                        [0] * batch_size,
                    )
                    self.set_active_loras(set(), lora_mapping)

                if self.hip_refresh_interval > 1:
                    graph_runner_checkout = CUDAGraphRunner(
                        self.model,
                        checkout_computed_hip_mask=True,
                    )
                    graph_runner_checkout.capture(
                        input_tokens[:batch_size],
                        input_positions[:batch_size],
                        kv_caches,
                        attn_metadata,
                        memory_pool=self.graph_memory_pool,
                        stream=graph_capture_context.stream,
                    )
                    self.graph_memory_pool = graph_runner_checkout.graph.pool()
                    
                    graph_runner_precomputed_mask = CUDAGraphRunner(
                        self.model,
                        using_precomputed_hip_mask=True,
                        precomputed_hip_caches=graph_runner_checkout.output_buffers['hip_caches'],
                    )
                    graph_runner_precomputed_mask.capture(
                        input_tokens[:batch_size],
                        input_positions[:batch_size],
                        kv_caches,
                        attn_metadata,
                        memory_pool=self.graph_memory_pool,
                        stream=graph_capture_context.stream,
                    )
                    self.graph_memory_pool = graph_runner_precomputed_mask.graph.pool()
                    
                    self.graph_runners[batch_size] = HipGraphRunnerCounter(
                        graph_runner_checkout, 
                        graph_runner_precomputed_mask,
                        self.hip_refresh_interval,
                    )
                else:
                    graph_runner = CUDAGraphRunner(self.model)
                    graph_runner.capture(
                        input_tokens[:batch_size],
                        input_positions[:batch_size],
                        kv_caches,
                        attn_metadata,
                        memory_pool=self.graph_memory_pool,
                        stream=graph_capture_context.stream,
                    )
                    self.graph_memory_pool = graph_runner.graph.pool()
                    self.graph_runners[batch_size] = graph_runner

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info("Graph capturing finished in %.0f secs.", elapsed_time)

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

class HipGraphRunnerCounter:
    def __init__(
        self, 
        graph_checkout: "CUDAGraphRunner", 
        graph_precomputed_mask: "CUDAGraphRunner",
        refresh_interval: int,
    ):
        self.counter = 0
        
        self.refresh_interval = refresh_interval
        
        self.graph_checkout = graph_checkout
        self.graph_precomputed_mask = graph_precomputed_mask
        
    def get_next_graph(self):
        if (self.counter % self.refresh_interval) == 0:
            graph = self.graph_checkout
        else:
            graph = self.graph_precomputed_mask
        self.counter += 1
        return graph

HipCache = Tuple[
    str,            #name
    torch.Tensor,   #indices
    torch.Tensor,   #ks
    torch.Tensor,   #blocks
]

class CUDAGraphRunner:

    def __init__(
        self, 
        model: nn.Module,
        
        checkout_computed_hip_mask: bool = False,
        using_precomputed_hip_mask: bool = False,
        precomputed_hip_caches: List[HipCache] = None,
    ):
        self.model = model
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}
        
        self.offload_prefetch = \
            (os.getenv('OFFLOAD_PREFETCH', '0') == '1') and\
            (os.getenv('CACHE_ENGINE', 'vllm') in ['offload_v', 'offload_kv'])
        self.need_offload_prefetch = False
        
        self.checkout_computed_hip_mask = checkout_computed_hip_mask
        self.using_precomputed_hip_mask = using_precomputed_hip_mask
        self.precomputed_hip_caches = precomputed_hip_caches

        self._graph: Optional[torch.cuda.CUDAGraph] = None

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        memory_pool: Optional[Tuple[int, int]],
        stream: torch.cuda.Stream,
        **kwargs,
    ) -> None:
        need_checkout_hip_caches = self.offload_prefetch or self.checkout_computed_hip_mask
        
        assert self._graph is None

        # Capture the graph.
        # NOTE(woosuk): Python 3.8 does not support multi-line with statements.
        # https://stackoverflow.com/questions/31039022/python-multi-line-with-statement
        if need_checkout_hip_caches:
            for m in self.model.modules():
                if isinstance(m, Attention):
                    backend = m.impl # type: HiPAttentionImpl
                    assert isinstance(backend, HiPAttentionImpl)
                    backend.checkout_last = True
                    backend.last_indices = None
                    backend.last_ks = None
        
        if self.using_precomputed_hip_mask:
            assert not self.checkout_computed_hip_mask
            idx_module = 0
            for m in self.model.modules():
                if isinstance(m, Attention):
                    backend = m.impl # type: HiPAttentionImpl
                    assert isinstance(backend, HiPAttentionImpl)
                    backend.using_precomputed_mask = True
                    backend.checkout_last = False
                    backend.precomputed_indices = self.precomputed_hip_caches[idx_module][1]
                    backend.precomputed_ks = self.precomputed_hip_caches[idx_module][2]
                    idx_module += 1
        else:
            for m in self.model.modules():
                if isinstance(m, Attention):
                    backend = m.impl # type: HiPAttentionImpl
                    if isinstance(backend, HiPAttentionImpl):
                        backend.using_precomputed_mask = False
        
        # Run the model once without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            **kwargs,
        )
        torch.cuda.synchronize()
        
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool, stream=stream):  # noqa: SIM117
            # with _maybe_cupy_nccl():
            hidden_states = self.model(
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                **kwargs,
            )
        torch.cuda.synchronize()
        
        if need_checkout_hip_caches:
            last_hip_caches = []
            for name, m in self.model.named_modules():
                if isinstance(m, Attention):
                    backend = m.impl # type: HiPAttentionImpl
                    backend.checkout_last = False
                    
                    last_hip_caches.append([
                        name, 
                        backend.last_indices, 
                        backend.last_ks, 
                        None,
                    ])
                    
                    backend.last_indices = None
                    backend.last_ks = None
        
        # after capture clear it
        for m in self.model.modules():
            if isinstance(m, Attention):
                backend = m.impl
                if isinstance(backend, HiPAttentionImpl):
                    backend.checkout_last = False
                    backend.using_precomputed_mask = False
                    backend.last_indices = None
                    backend.last_ks = None
                    backend.precomputed_indices = None
                    backend.precomputed_ks = None
        
        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": attn_metadata.slot_mapping,
            "seq_lens_tensor": attn_metadata.decode_metadata.seq_lens_tensor,
            "block_tables": attn_metadata.decode_metadata.block_tables,
        }
        self.output_buffers = {"hidden_states": hidden_states}
        if need_checkout_hip_caches:
            self.output_buffers['hip_caches'] = last_hip_caches
        return

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches
        
        benchmark_graph_runner = os.getenv('BENCHMARK_GRAPH_RUNNER', '0') == '1'
        
        if benchmark_graph_runner:
            start_prepare = torch.cuda.Event(enable_timing=True)
            end_prepare = torch.cuda.Event(enable_timing=True)
            start_replay = torch.cuda.Event(enable_timing=True)
            end_replay = torch.cuda.Event(enable_timing=True)
            start_post = torch.cuda.Event(enable_timing=True)
            end_post = torch.cuda.Event(enable_timing=True)
            
            start_prepare.record()
        
        # prefetch previously accessed tokens
        if self.offload_prefetch and self.need_offload_prefetch:
            from vllm.model_executor.layers.attention import Attention
            from third_party.vllm.vllm.attention.backends.hip import HiPAttentionImpl
            from vllm.worker.cache_engine.map_cache_engine import MapCacheEngine, ManagedTensor
            kv_caches = self.input_buffers['kv_caches']
            hip_caches = self.output_buffers['hip_caches']
            
            # prefetch KV
            torch.cuda.synchronize(torch.cuda.current_device())
            t = time.time()
            
            for kv_cache, hip_cache in zip(kv_caches, hip_caches):
                _, v_cache = kv_cache
                if isinstance(v_cache, ManagedTensor):
                    name, indices, ks, cpu_indices = hip_cache
                    if (cpu_indices is not None):
                        v_cache.prefetch_blocks(cpu_indices, v_cache.target_device)
                        hip_cache[-1] = None
                        self.need_offload_prefetch = False
            
            torch.cuda.synchronize(torch.cuda.current_device())
            
            print(f'prefetch v: {(time.time() - t)*1000:.3f} ms')
            # prefetch K

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(attn_metadata.slot_mapping,
                                                 non_blocking=True)
        self.input_buffers["seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.seq_lens_tensor, non_blocking=True)
        self.input_buffers["block_tables"].copy_(
            attn_metadata.decode_metadata.block_tables, non_blocking=True)
        # Run the graph.
        self.graph.replay()

        if benchmark_graph_runner:
            end_replay.record()
            start_post.record()
        
        if self.offload_prefetch and self.need_offload_prefetch:
            kv_caches = self.input_buffers['kv_caches']
            hip_caches = self.output_buffers['hip_caches']
            for kv_cache, hip_cache in zip(kv_caches, hip_caches):
                k_cache, v_cache = kv_cache
                name, indices, ks, _ = hip_cache
                if \
                    (isinstance(k_cache, ManagedTensor) or isinstance(v_cache, ManagedTensor)) and\
                    (indices is not None and ks is not None):
                    batch_size, seq_len = input_ids.shape
                    block_size = 16
                    
                    N_H, TDST, K = indices.shape
                    N = batch_size
                    H = N_H // batch_size
                    
                    indices = indices.view(N, H, TDST, K)
                    
                    block_indices_batch = []
                    
                    table = attn_metadata.block_tables
                    
                    for ibatch in range(N):
                        block_indices = (indices[ibatch] // block_size).view(-1)
                        block_indices = torch.unique(block_indices)
                        # print('marker1112:', block_indices.shape[0])
                        block_indices = torch.clamp(block_indices, 0, table.shape[-1] - 1).to(torch.int64)
                        block_indices = table[ibatch].gather(0, index=block_indices)
                        assert block_indices.ndim == 1
                        block_indices_batch.append(block_indices)
                    
                    block_indices = torch.concat(block_indices_batch).sort().values.to('cpu', non_blocking=True)
                    
                    hip_cache[-1] = block_indices

        if benchmark_graph_runner:
            end_post.record()
            torch.cuda.synchronize()
            
            elapsed_prepare = start_prepare.elapsed_time(end_prepare)
            elapsed_replay = start_replay.elapsed_time(end_replay)
            elapsed_post = start_post.elapsed_time(end_post)
            print(f'[{time.time() * 1000:.3f}] CUDAGraphRunner.forward: prepare: {elapsed_prepare:.3f}, replay: {elapsed_replay:.3f}, post: {elapsed_post:.3f}')

        # Return the output tensor.
        return self.output_buffers["hidden_states"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)


def _prepare_fake_inputs(
    seq_len: int, 
    vision_language_config: Optional[VisionLanguageConfig]
):
    """Prepare fake inputs for profile run."""
    if vision_language_config:
        prompt_tokens = [
            vision_language_config.image_token_id
        ] * vision_language_config.image_feature_size + [0] * (
            seq_len - vision_language_config.image_feature_size)
        fake_image_input = MultiModalData(
            type=MultiModalData.Type.IMAGE,
            data=torch.zeros(
                vision_language_config.image_input_shape 
                if vision_language_config.fake_image_input_shape is None else 
                vision_language_config.fake_image_input_shape,
                dtype=torch.float16
            )
        )
    else:
        prompt_tokens = [0] * seq_len
        fake_image_input = None
    return SequenceData(prompt_tokens), fake_image_input


def _is_block_tables_empty(block_tables: Union[None, Dict]):
    """
    Check if block_tables is None or a dictionary with all None values.
    """
    if block_tables is None:
        return True
    if isinstance(block_tables, dict) and all(
            value is None for value in block_tables.values()):
        return True
    return False

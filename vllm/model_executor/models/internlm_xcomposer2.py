# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               ColumnParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class PLoRA(nn.Module):
    def __init__(self, in_features, out_features, linear_layer, lora_r, lora_alpha, lora_len):
        super().__init__()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_len = lora_len
        self.lora_scaling = self.lora_alpha / self.lora_r
        self.linear = linear_layer
        if isinstance(linear_layer, ColumnParallelLinear):
            self.Plora_A = torch.nn.Linear(in_features, self.lora_r, bias=False)
            self.Plora_B = ColumnParallelLinear(self.lora_r, out_features, bias=False)
        else:
            self.Plora_A = RowParallelLinear(in_features, self.lora_r, bias=False)
            self.Plora_B = torch.nn.Linear(self.lora_r, out_features, bias=False)

    def forward(self, x, im_mask=None):
        res, bias = self.linear(x)
        if im_mask is not None:
            x = self.Plora_A(x)
            x = x[0] if isinstance(x, tuple) else x
            x = self.Plora_B(x)
            x = x[0] if isinstance(x, tuple) else x
            res = torch.where(im_mask[..., None], res, res + x * self.lora_scaling)
        return res, bias


class InternLM2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.w1 = PLoRA(
            hidden_size, intermediate_size,
            ColumnParallelLinear(hidden_size,
                                 intermediate_size,
                                 bias=False,
                                 linear_method=linear_method),
            256, 256, 1225
        )
        self.w3 = PLoRA(
            hidden_size, intermediate_size,
            ColumnParallelLinear(hidden_size,
                                 intermediate_size,
                                 bias=False,
                                 linear_method=linear_method),
            256, 256, 1225
        )
        self.w2 = PLoRA(
            intermediate_size, hidden_size,
            RowParallelLinear(intermediate_size,
                              hidden_size,
                              bias=False,
                              linear_method=linear_method),
            256, 256, 1225
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x, im_mask=None):
        gate_up_1, _ = self.w1(x, im_mask)
        gate_up_3, _ = self.w3(x, im_mask)
        x = self.act_fn(torch.cat([gate_up_1, gate_up_3], dim=-1))
        x, _ = self.w2(x, im_mask)
        return x


class InternLM2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        linear_method: Optional[LinearMethodBase] = None,
        layer_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.wqkv = PLoRA(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim,
            QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                linear_method=linear_method,
            ),
            8, 16, 0
        )
        self.wo = PLoRA(
            self.total_num_heads * self.head_dim,
            hidden_size,
            RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=False,
                linear_method=linear_method,
            ),
            256, 256, 1225
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              layer_index=layer_index)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        im_mask: Optional[torch.Tensor],
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.wqkv(hidden_states, im_mask)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.wo(attn_output, im_mask)
        return output


class InternLMDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
        layer_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.attention = InternLM2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            linear_method=linear_method,
            layer_index=layer_index,
        )
        self.feed_forward = InternLM2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.attention_norm = RMSNorm(config.hidden_size,
                                      eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        im_mask: Optional[torch.Tensor],
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)
        else:
            hidden_states, residual = self.attention_norm(
                hidden_states, residual)
        hidden_states = self.attention(
            positions=positions,
            hidden_states=hidden_states,
            im_mask=im_mask,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states, im_mask)
        return hidden_states, residual


class InternLM2Model(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            InternLMDecoderLayer(config, linear_method, layer_index=layer_index)
            for layer_index in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor],
        input_im_masks: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.tok_embeddings(input_ids)
        if input_embeds is not None:
            hidden_states = input_embeds.to(hidden_states.dtype)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                input_im_masks,
                kv_caches[i],
                input_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class InternLM2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = InternLM2Model(config, linear_method)
        self.output = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        input_embeds: Optional[torch.Tensor] = None,
        input_im_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_im_masks = input_im_masks.bool() if input_im_masks is not None else None
        hidden_states = self.model(input_ids, input_embeds, input_im_masks, positions, kv_caches,
                                   input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.output.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if ("rotary_emb.inv_freq" in name
                    or name.startswith('vit.')
                    or name.startswith('vision_proj.')
                    or name in ["plora_glb_GN", "plora_sub_GN"]):
                continue
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            if name.endswith(".weight") and name not in params_dict:
                name = name.replace(".weight", ".linear.weight")
            param = params_dict[name]
            if "wqkv.linear.weight" in name:
                config = self.config
                kv_groups = config.num_attention_heads // config.num_key_value_heads
                head_dim = config.hidden_size // config.num_attention_heads
                loaded_weight = loaded_weight.view(-1, 2 + kv_groups, head_dim, loaded_weight.shape[-1])
                wq, wk, wv = torch.split(loaded_weight, [kv_groups, 1, 1], dim=1)
                wq = wq.reshape(-1, wq.shape[-1])
                wk = wk.reshape(-1, wk.shape[-1])
                wv = wv.reshape(-1, wv.shape[-1])
                weight_loader = param.weight_loader
                weight_loader(param, wq, 'q')
                weight_loader(param, wk, 'k')
                weight_loader(param, wv, 'v')
            else:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

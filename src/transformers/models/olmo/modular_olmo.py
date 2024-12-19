from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_olmo import OlmoConfig


logger = logging.get_logger(__name__)


class OlmoLayerNorm(nn.Module):
    """LayerNorm but with no learnable weight or bias."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.normalized_shape = (hidden_size,)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        return F.layer_norm(hidden_states.to(dtype=torch.float32), self.normalized_shape, None, None, eps=1e-5).to(
            orig_dtype
        )


class OlmoMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class OlmoAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OlmoDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: OlmoConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.input_layernorm = OlmoLayerNorm(config.hidden_size)
        self.post_attention_layernorm = OlmoLayerNorm(config.hidden_size)
        self.self_attn = OlmoAttention(config=config, layer_idx=layer_idx)


class OlmoModel(LlamaModel):
    def __init__(self, config: OlmoConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [OlmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OlmoLayerNorm(config.hidden_size)


class OlmoForCausalLM(LlamaForCausalLM):
    pass

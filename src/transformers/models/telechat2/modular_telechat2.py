from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaModel,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_telechat2 import TeleChat2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "TeleAI/TeleChat2-3B"
_CONFIG_FOR_DOC = "TeleChat2Config"


class TeleChat2MLP(nn.Module):
    """
    TeleChat2 MLP block with a gated activation function.
    """

    def __init__(self, config: TeleChat2Config):
        super().__init__()
        hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, hidden_size, bias=True)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states):
        intermediate_output = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        output = F.dropout(intermediate_output, p=self.hidden_dropout, training=self.training)
        return output


class TeleChat2Attention(LlamaAttention):
    def __init__(self, config: TeleChat2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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


class TeleChat2PreTrainedModel(LlamaPreTrainedModel):
    pass


class TeleChat2Model(LlamaModel):
    pass


class TeleChat2ForCausalLM(LlamaForCausalLM):
    pass


class TeleChat2ForSequenceClassification(LlamaForSequenceClassification):
    pass


class TeleChat2ForQuestionAnswering(LlamaForQuestionAnswering):
    pass


class TeleChat2ForTokenClassification(LlamaForTokenClassification):
    pass


__all__ = [
    "TeleChat2ForCausalLM",
    "TeleChat2ForQuestionAnswering",
    "TeleChat2ForSequenceClassification",
    "TeleChat2ForTokenClassification",
    "TeleChat2Model",
    "TeleChat2PreTrainedModel",
]

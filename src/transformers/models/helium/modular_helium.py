# coding=utf-8
# Copyright 2024 The Kyutai and HuggingFace Inc. teams. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...utils import logging
from ..gemma.modeling_gemma import (
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
)
from ..granite.modeling_granite import (
    GraniteAttention,
)
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
)
from .configuration_helium import HeliumConfig


logger = logging.get_logger(__name__)


class HeliumRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class HeliumRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class HeliumMLP(LlamaMLP):
    pass


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Interleave them instead of usual shape
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class HeliumAttention(GraniteAttention):
    def __init__(self, config: HeliumConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.scaling = 1 / math.sqrt(self.head_dim)


class HeliumDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HeliumConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.mlp = HeliumMLP(config)
        self.input_layernorm = HeliumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HeliumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class HeliumPreTrainedModel(LlamaPreTrainedModel):
    pass


class HeliumModel(HeliumPreTrainedModel, LlamaModel):
    def __init__(self, config: HeliumConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [HeliumDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HeliumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HeliumRotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class HeliumForCausalLM(GemmaForCausalLM):
    def __init__(self, config: HeliumConfig):
        super().__init__(config)
        self.model = HeliumModel(config)
        self.post_init()


class HeliumForSequenceClassification(GemmaForSequenceClassification):
    def __init__(self, config: HeliumConfig):
        super().__init__(config)
        self.model = HeliumModel(config)
        self.post_init()


class HeliumForTokenClassification(GemmaForTokenClassification):
    def __init__(self, config: HeliumConfig):
        super().__init__(config)
        self.model = HeliumModel(config)
        self.post_init()


__all__ = [
    "HeliumPreTrainedModel",
    "HeliumModel",
    "HeliumForCausalLM",
    "HeliumForSequenceClassification",
    "HeliumForTokenClassification",
]

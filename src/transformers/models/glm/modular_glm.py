# coding=utf-8
# Copyright 2024 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
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
    GraniteFlashAttention2,
    GraniteSdpaAttention,
)
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaModel,
    LlamaPreTrainedModel,
)
from ..phi3.modeling_phi3 import (
    Phi3MLP,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
)
from .configuration_glm import GlmConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "THUDM/glm-4-9b"


class GlmRMSNorm(Phi3RMSNorm):
    pass


class GlmRotaryEmbedding(Phi3RotaryEmbedding):
    pass


class GlmMLP(Phi3MLP):
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

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class GlmAttention(GraniteAttention):
    def __init__(self, config: GlmConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.scaling = 1 / math.sqrt(self.head_dim)


class GlmFlashAttention2(GlmAttention, GraniteFlashAttention2):
    pass


class GlmSdpaAttention(GraniteSdpaAttention):
    pass


GLM_ATTENTION_CLASSES = {
    "eager": GlmAttention,
    "flash_attention_2": GlmFlashAttention2,
    "sdpa": GlmSdpaAttention,
}


class GlmDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: GlmConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.mlp = GlmMLP(config)
        self.input_layernorm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class GlmPreTrainedModel(LlamaPreTrainedModel):
    pass


class GlmModel(GlmPreTrainedModel, LlamaModel):
    def __init__(self, config: GlmConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GlmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GlmRotaryEmbedding(
            dim=int(config.head_dim * config.partial_rotary_factor),
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class GlmForCausalLM(GemmaForCausalLM):
    def __init__(self, config: GlmConfig):
        super().__init__(config)
        self.model = GlmModel(config)
        self.post_init()


class GlmForSequenceClassification(GemmaForSequenceClassification):
    def __init__(self, config: GlmConfig):
        super().__init__(config)
        self.model = GlmModel(config)
        self.post_init()


class GlmForTokenClassification(GemmaForTokenClassification):
    def __init__(self, config: GlmConfig):
        super().__init__(config)
        self.model = GlmModel(config)
        self.post_init()


__all__ = [
    "GlmPreTrainedModel",
    "GlmModel",
    "GlmForCausalLM",
    "GlmForSequenceClassification",
    "GlmForTokenClassification",
]

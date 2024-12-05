# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""PyTorch LightGlue model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from transformers import PreTrainedModel, add_start_docstrings

from ...activations import ACT2FN
from ...utils import (
    ModelOutput,
    add_start_docstrings_to_model_forward,
    get_torch_version,
    is_flash_attn_2_available,
    logging,
)
from ..auto import AutoModelForKeypointDetection
from .configuration_lightglue import LightGlueConfig


if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC_ = "LightGlueConfig"
_CHECKPOINT_FOR_DOC_ = "stevenbucaille/lightglue_superpoint"


def normalize_keypoints(keypoints: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Normalize keypoints locations based on image image_shape

    Args:
        keypoints (`torch.Tensor` of shape `(batch_size, num_keypoints, 2)`):
            Keypoints locations in (x, y) format.
        height (`int`):
            Image height.
        width (`int`):
            Image width.

    Returns:
        Normalized keypoints locations of shape (`torch.Tensor` of shape `(batch_size, num_keypoints, 2)`).
    """
    size = torch.tensor([width, height], device=keypoints.device, dtype=keypoints.dtype)[None]
    shift = size / 2
    scale = size.max(-1).values / 2
    keypoints = (keypoints - shift[..., None, :]) / scale[..., None, None]
    return keypoints


@dataclass
class LightGlueKeypointMatchingOutput(ModelOutput):
    """
    Base class for outputs of keypoint matching models. Due to the nature of keypoint detection and matching, the number
    of keypoints is not fixed and can vary from image to image, which makes batching non-trivial. In the batch of
    images, the maximum number of matches is set as the dimension of the matches and matching scores. The mask tensor is
    used to indicate which values in the keypoints, matches and matching_scores tensors are keypoint matching
    information.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss computed during training.
        matches (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Index of keypoint matched in the other image.
        matching_scores (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Scores of predicted matches.
        keypoints (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`):
            Absolute (x, y) coordinates of predicted keypoints in a given image.
        prune (`torch.IntTensor` of shape `(batch_size, num_keypoints)`):
            Pruning mask indicating which keypoints are removed and at which layer.
        mask (`torch.BoolTensor` of shape `(batch_size, num_keypoints)`):
            Mask indicating which values in matches and matching_scores are keypoint matching information.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, 2, num_channels,
            num_keypoints)` returned when `output_hidden_states=True` is passed or when
            `config.output_hidden_states=True`
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, 2, num_heads, num_keypoints,
            num_keypoints)` returned when `output_attentions=True` is passed or when
            `config.output_attentions=True`
    """

    loss: Optional[torch.FloatTensor] = None
    matches: torch.FloatTensor = None
    matching_scores: torch.FloatTensor = None
    keypoints: torch.FloatTensor = None
    prune: torch.IntTensor = None
    mask: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LightGluePositionalEncoder(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()
        self.projector = nn.Linear(2, config.descriptor_dim // config.num_heads // 2, bias=False)

    def forward(
        self, keypoints: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        projected_keypoints = self.projector(keypoints)
        cosines = torch.cos(projected_keypoints)
        sines = torch.sin(projected_keypoints)
        embeddings = torch.cat([sines, cosines], -1)
        output = (embeddings, projected_keypoints) if output_hidden_states else (embeddings,)
        return output


# Copied from transformers.models.roformer.modeling_roformer.RoFormerSelfAttention with RoFormer->LightGlue
class LightGlueSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.rotary_value = config.rotary_value

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            if sinusoidal_pos is not None:
                if self.rotary_value:
                    query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer, value_layer
                    )
                else:
                    query_layer, key_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer
                    )
            if past_key_value is not None:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in LightGlueModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer


# Adapted from BertSdpaSelfAttention and RoFormerSelfAttention
class LightGlueSdpaSelfAttention(LightGlueSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    # Adapted from BertSelfAttention
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sinusoidal_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        is_cross_attention = encoder_hidden_states is not None

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        key_layer = self.transpose_for_scores(self.key(current_states))
        value_layer = self.transpose_for_scores(self.value(current_states))
        if sinusoidal_pos is not None:
            if self.rotary_value:
                query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_layer, key_layer, value_layer
                )
            else:
                query_layer, key_layer = self.apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

        outputs = (attn_output,)
        return outputs


class LightGlueFlashAttentionSelfAttention(LightGlueSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    # Adapted from BertSelfAttention
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sinusoidal_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        batch_size, q_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states)).contiguous()

        is_cross_attention = encoder_hidden_states is not None

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = encoder_attention_mask if is_cross_attention else attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze()
            attention_mask = torch.where(attention_mask == -0.0, 1, 0)

        key_layer = self.transpose_for_scores(self.key(current_states))
        value_layer = self.transpose_for_scores(self.value(current_states))
        if sinusoidal_pos is not None:
            if self.rotary_value:
                query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_layer, key_layer, value_layer
                )
            else:
                query_layer, key_layer = self.apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer)

        query_layer = query_layer.reshape(batch_size, q_len, self.num_attention_heads, self.attention_head_size)
        key_layer = key_layer.reshape(batch_size, q_len, self.num_attention_heads, self.attention_head_size)
        value_layer = value_layer.reshape(batch_size, q_len, self.num_attention_heads, self.attention_head_size)

        input_dtype = query_layer.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.query.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_layer = query_layer.to(target_dtype)
            key_layer = key_layer.to(target_dtype)
            value_layer = value_layer.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            q_len,
            is_causal=False,
        )

        attn_output = attn_output.reshape(batch_size, q_len, self.all_head_size).contiguous()
        outputs = (attn_output,)
        return outputs


class LightGlueMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        embeddings_dim = config.descriptor_dim
        self.dense = nn.Linear(2 * embeddings_dim, 2 * embeddings_dim)
        self.layer_norm = nn.LayerNorm(2 * embeddings_dim, elementwise_affine=True)
        self.activation = ACT2FN["gelu"]
        self.output = nn.Linear(2 * embeddings_dim, embeddings_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.output(hidden_states)
        return hidden_states


class LightGlueSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        embeddings_dim = config.descriptor_dim
        self.dense = nn.Linear(embeddings_dim, embeddings_dim)
        self.mlp = LightGlueMLP(config)

    def forward(
        self, context: torch.Tensor, descriptors: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        message = self.dense(context)
        hidden_states = torch.cat([descriptors, message], dim=-1)
        output_state = self.mlp(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, output_state)
        descriptors = descriptors + output_state
        return descriptors, all_hidden_states


LIGHTGLUE_SELF_ATTENTION_CLASSES = {
    "eager": LightGlueSelfAttention,
    "sdpa": LightGlueSdpaSelfAttention,
    "flash_attention_2": LightGlueFlashAttentionSelfAttention,
}


class LightGlueAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LIGHTGLUE_SELF_ATTENTION_CLASSES[config._attn_implementation](config)
        self.output = LightGlueSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_hidden_states=False,
        output_attentions=False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        self_output = self_outputs[0]
        attention_probs = self_outputs[1:] if output_attentions else None
        outputs = self.output(self_output, hidden_states, output_hidden_states=output_hidden_states)
        output = outputs[0]
        hidden_states = outputs[1] if output_hidden_states else None
        outputs = (output, hidden_states, attention_probs)
        return outputs


class LightGlueTransformerLayer(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()
        self.self_attention_block = LightGlueAttention(config)
        self.cross_attention_block = LightGlueAttention(config)

    def forward(
        self,
        descriptors: torch.Tensor,
        keypoints: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (descriptors,)

        batch_size, num_keypoints, descriptor_dim = descriptors.shape
        # (batch_size, num_keypoints, descriptor_dim) -> (batch_size, 1, num_keypoints, descriptor_dim)
        keypoints = keypoints.unsqueeze(-3)
        self_attention_output = self.self_attention_block(
            descriptors,
            sinusoidal_pos=keypoints,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        descriptors = self_attention_output[0]

        encoder_hidden_states = (
            descriptors.reshape(-1, 2, num_keypoints, descriptor_dim)
            .flip(1)
            .reshape(batch_size, num_keypoints, descriptor_dim)
        )
        encoder_attention_mask = (
            attention_mask.reshape(-1, 2, 1, 1, num_keypoints).flip(1).reshape(batch_size, 1, 1, num_keypoints)
            if attention_mask is not None
            else None
        )

        cross_attention_output = self.cross_attention_block(
            descriptors,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        descriptors = cross_attention_output[0]
        if output_hidden_states:
            all_hidden_states = (
                all_hidden_states
                + (self_attention_output[0].reshape(batch_size, num_keypoints, descriptor_dim),)
                + self_attention_output[1]
                + (cross_attention_output[0].reshape(batch_size, num_keypoints, descriptor_dim),)
                + cross_attention_output[1]
            )

        if output_attentions:
            all_attentions = all_attentions + self_attention_output[2] + cross_attention_output[2]

        return descriptors, all_hidden_states, all_attentions


def sigmoid_log_double_softmax(
    similarity: torch.Tensor, matchability0: torch.Tensor, matchability1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    batch_size, num_keypoints_0, num_keypoints_1 = similarity.shape
    certainties = nn.functional.logsigmoid(matchability0) + nn.functional.logsigmoid(matchability1).transpose(1, 2)
    scores0 = nn.functional.log_softmax(similarity, 2)
    scores1 = nn.functional.log_softmax(similarity.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = similarity.new_full((batch_size, num_keypoints_0 + 1, num_keypoints_1 + 1), 0)
    scores[:, :num_keypoints_0, :num_keypoints_1] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = nn.functional.logsigmoid(-matchability0.squeeze(-1))
    scores[:, -1, :-1] = nn.functional.logsigmoid(-matchability1.squeeze(-1))
    return scores


class LightGlueMatchAssignmentLayer(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.descriptor_dim = config.descriptor_dim
        self.final_projection = nn.Linear(self.descriptor_dim, self.descriptor_dim, bias=True)
        self.matchability = nn.Linear(self.descriptor_dim, 1, bias=True)

    def forward(self, descriptors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_keypoints, descriptor_dim = descriptors.shape
        # Final projection and similarity computation
        m_descriptors = self.final_projection(descriptors)
        m_descriptors = m_descriptors / self.descriptor_dim**0.25
        m_descriptors = m_descriptors.reshape(batch_size // 2, 2, num_keypoints, descriptor_dim)
        m_descriptors0 = m_descriptors[:, 0]
        m_descriptors1 = m_descriptors[:, 1]
        similarity = torch.einsum("bmd,bnd->bmn", m_descriptors0, m_descriptors1)
        if mask is not None:
            mask = mask.reshape(batch_size // 2, 2, num_keypoints)
            mask0 = mask[:, 0].unsqueeze(-1)
            mask1 = mask[:, 1].unsqueeze(-1).transpose(-1, -2)
            mask = mask0 * mask1
            similarity = similarity.masked_fill(mask == 0, torch.finfo(similarity.dtype).min)

        # Compute matchability of descriptors
        matchability = self.matchability(descriptors)
        matchability = matchability.reshape(batch_size // 2, 2, num_keypoints, 1)
        matchability_0 = matchability[:, 0]
        matchability_1 = matchability[:, 1]

        # Compute scores from similarity and matchability
        scores = sigmoid_log_double_softmax(similarity, matchability_0, matchability_1)
        return scores

    def get_matchability(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Get matchability of descriptors as a probability"""
        matchability = self.matchability(descriptors)
        matchability = nn.functional.sigmoid(matchability).squeeze(-1)
        return matchability


class LightGlueTokenConfidenceLayer(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.token = nn.Linear(config.descriptor_dim, 1)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        token = self.token(descriptors.detach())
        token = nn.functional.sigmoid(token).squeeze(-1)
        return token


def filter_matches(scores: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    batch_size, _, _ = scores.shape
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    matches0, matches1 = max0.indices, max1.indices
    indices0 = torch.arange(matches0.shape[1], device=matches0.device)[None]
    indices1 = torch.arange(matches1.shape[1], device=matches1.device)[None]
    mutual0 = indices0 == matches1.gather(1, matches0)
    mutual1 = indices1 == matches0.gather(1, matches1)
    max0 = max0.values.exp()
    zero = max0.new_tensor(0)
    matching_scores0 = torch.where(mutual0, max0, zero)
    matching_scores1 = torch.where(mutual1, matching_scores0.gather(1, matches1), zero)
    valid0 = mutual0 & (matching_scores0 > threshold)
    valid1 = mutual1 & valid0.gather(1, matches1)
    matches0 = torch.where(valid0, matches0, -1)
    matches1 = torch.where(valid1, matches1, -1)
    matches = torch.stack([matches0, matches1]).transpose(0, 1).reshape(batch_size * 2, -1)
    matching_scores = torch.stack([matching_scores0, matching_scores1]).transpose(0, 1).reshape(batch_size * 2, -1)

    return matches, matching_scores


class LightGluePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LightGlueConfig
    base_model_prefix = "superglue"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    _supports_flash_attn_2 = True
    _supports_sdpa = False

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


LIGHTGLUE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LightGlueConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

LIGHTGLUE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`LightPointImageProcessor`]. See
            [`LightPointImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "LightGlue model taking images as inputs and outputting the matching of them.",
    LIGHTGLUE_START_DOCSTRING,
)
class LightGlueForKeypointMatching(LightGluePreTrainedModel):
    """
    LightGlue is a model matching keypoints in images by leveraging detections from a keypoint detector such as
    SuperPoint. It is based on the SuperGlue architecture and is designed to be lightweight and efficient.
    It consists of :
        1. Keypoint Encoder
        2. A Graph Neural Network with self and cross attention layers
        3. Matching Assignment layers

    The correspondence ids use -1 to indicate non-matching points.

    Philipp Lindenberger, Paul-Edouard Sarlin and Marc Pollefeys. LightGlue: Local Feature Matching at Light Speed.
    In ICCV 2023. https://arxiv.org/pdf/2306.13643.pdf
    """

    def __init__(self, config: LightGlueConfig):
        super().__init__(config)

        self.keypoint_detector = AutoModelForKeypointDetection.from_config(config.keypoint_detector_config)

        self.descriptor_dim = config.descriptor_dim
        self.num_layers = config.num_layers
        self.filter_threshold = config.filter_threshold
        self.depth_confidence = config.depth_confidence
        self.width_confidence = config.width_confidence

        if self.descriptor_dim != config.keypoint_detector_config.descriptor_decoder_dim:
            self.input_projection = nn.Linear(
                config.keypoint_detector_config.descriptor_decoder_dim, self.descriptor_dim, bias=True
            )
        else:
            self.input_projection = nn.Identity()

        self.positional_encoder = LightGluePositionalEncoder(config)

        self.transformer_layers = nn.ModuleList([LightGlueTransformerLayer(config) for _ in range(config.num_layers)])
        self.match_assignment_layers = nn.ModuleList(
            [LightGlueMatchAssignmentLayer(config) for _ in range(config.num_layers)]
        )
        self.token_confidence = nn.ModuleList(
            [LightGlueTokenConfidenceLayer(config) for _ in range(config.num_layers - 1)]
        )

        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor([self._get_confidence_threshold(i) for i in range(self.num_layers)]),
        )

        self.post_init()

    def _get_confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold for a given layer"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.num_layers)
        return np.clip(threshold, 0, 1)

    def _keypoint_processing(
        self, descriptors: torch.Tensor, keypoints: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        descriptors = descriptors.detach().contiguous()
        projected_descriptors = self.input_projection(descriptors)
        keypoint_encoding_output = self.positional_encoder(keypoints, output_hidden_states=output_hidden_states)
        return projected_descriptors, keypoint_encoding_output

    def _get_early_stopped_image_pairs(
        self, keypoint_confidences: torch.Tensor, layer_index: int, mask: torch.Tensor, num_points: torch.Tensor
    ) -> torch.Tensor:
        """evaluate whether we should stop inference based on the confidence of the keypoints"""
        batch_size, _ = mask.shape
        if layer_index < self.num_layers - 1:
            # If the current layer is not the last layer, we compute the confidence of the keypoints and check
            # if we should stop the forward pass through the transformer layers for each pair of images.
            keypoint_confidences = keypoint_confidences.masked_fill(mask == 0, 1)
            keypoint_confidences = keypoint_confidences.reshape(batch_size // 2, -1)
            threshold = self.confidence_thresholds[layer_index]
            ratio_confident = 1.0 - (keypoint_confidences < threshold).float().sum(dim=1) / num_points
            early_stopped_pairs = ratio_confident > self.depth_confidence
        else:
            # If the current layer is the last layer, we stop the forward pass through the transformer layers for
            # all pairs of images.
            early_stopped_pairs = torch.ones(batch_size, dtype=torch.bool)
        return early_stopped_pairs

    def _get_keypoint_matching(self, descriptors, mask, layer_index, early_stops=None):
        if early_stops is not None:
            descriptors = descriptors[early_stops]
            mask = mask[early_stops]
        scores = self.match_assignment_layers[layer_index](descriptors, mask)
        matches, matching_scores = filter_matches(scores, self.filter_threshold)
        return matches, matching_scores

    def _get_pruning_mask(self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def _do_layer_keypoint_pruning(
        self,
        descriptors: torch.Tensor,
        keypoints: torch.Tensor,
        mask: torch.Tensor,
        indices: torch.Tensor,
        prune_output: torch.Tensor,
        keypoint_confidences: torch.Tensor,
        layer_index: int,
    ):
        """
        For a given layer, prune keypoints based on the confidence of the keypoints and the matchability of the
        descriptors.
        """
        batch_size, _, _ = descriptors.shape
        descriptors_matchability = self.match_assignment_layers[layer_index].get_matchability(descriptors)
        pruned_keypoints_mask = self._get_pruning_mask(keypoint_confidences, descriptors_matchability, layer_index)
        pruned_keypoints_mask = pruned_keypoints_mask.masked_fill(mask == 0, torch.tensor(False))

        # For each image, we extract the pruned indices and the corresponding descriptors and keypoints.
        pruned_descriptors, pruned_keypoints, pruned_mask, pruned_indices = (
            [t[mask] for t, mask in zip(tensor, pruned_keypoints_mask)]
            for tensor in [descriptors, keypoints, pruned_keypoints_mask, indices]
        )
        for i in range(batch_size):
            prune_output[i, pruned_indices[i]] += 1

        # Pad the pruned descriptors, keypoints, indices and mask to have the same shape across the batch.
        pruned_descriptors, pruned_keypoints, pruned_mask = (
            pad_sequence(pruned_tensor, batch_first=True)
            for pruned_tensor in [pruned_descriptors, pruned_keypoints, pruned_mask]
        )
        pruned_indices = pad_sequence(pruned_indices, batch_first=True, padding_value=-1)

        return pruned_descriptors, pruned_keypoints, pruned_indices, pruned_mask, prune_output

    def _do_final_keypoint_pruning(
        self,
        indices: torch.Tensor,
        matches: torch.Tensor,
        matching_scores: torch.Tensor,
        num_keypoints: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, num_keypoints) -> (batch_size // 2, 2, num_keypoints) -> 2 * (batch_size // 2, num_keypoints) to
        # have tensors from
        batch_size, _ = indices.shape
        indices, matches, matching_scores = (
            tensor.reshape(batch_size // 2, 2, -1) for tensor in [indices, matches, matching_scores]
        )
        indices0 = indices[:, 0]
        indices1 = indices[:, 1]
        matches0 = matches[:, 0]
        matches1 = matches[:, 1]
        matching_scores0 = matching_scores[:, 0]
        matching_scores1 = matching_scores[:, 1]

        # Prepare final matches and matching scores
        _matches = torch.full((batch_size // 2, 2, num_keypoints), -1, device=indices.device, dtype=matches.dtype)
        _matching_scores = torch.zeros(
            (batch_size // 2, 2, num_keypoints), device=indices.device, dtype=matching_scores.dtype
        )
        # Fill the matches and matching scores for each image pair
        for i in range(batch_size // 2):
            _matches[i, 0, indices0[i]] = torch.where(
                matches0[i] == -1, -1, indices1[i].gather(0, matches0[i].clamp(min=0))
            )
            _matches[i, 1, indices1[i]] = torch.where(
                matches1[i] == -1, -1, indices0[i].gather(0, matches1[i].clamp(min=0))
            )
            _matching_scores[i, 0, indices0[i]] = matching_scores0[i]
            _matching_scores[i, 1, indices1[i]] = matching_scores1[i]
        return _matches, _matching_scores

    def _match_image_pair(
        self,
        keypoints: torch.Tensor,
        descriptors: torch.Tensor,
        height: int,
        width: int,
        mask: torch.Tensor = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple, Tuple]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if keypoints.shape[2] == 0:  # no keypoints
            shape = keypoints.shape[:-1]
            return (
                keypoints.new_full(shape, -1, dtype=torch.int),
                keypoints.new_zeros(shape),
                keypoints.new_zeros(shape),
                all_hidden_states,
                all_attentions,
            )

        device = keypoints.device
        batch_size, _, initial_num_keypoints, _ = keypoints.shape
        num_points_per_pair = torch.sum(mask.reshape(batch_size, -1), dim=1)
        # (batch_size, 2, num_keypoints, 2) -> (batch_size * 2, num_keypoints, 2)
        keypoints = keypoints.reshape(batch_size * 2, initial_num_keypoints, 2)
        mask = mask.reshape(batch_size * 2, initial_num_keypoints) if mask is not None else None
        descriptors = descriptors.reshape(batch_size * 2, initial_num_keypoints, self.descriptor_dim)
        image_indices = torch.arange(batch_size * 2, device=device)
        # Keypoint normalization
        keypoints = normalize_keypoints(keypoints, height, width)

        descriptors, keypoint_encoding_output = self._keypoint_processing(
            descriptors, keypoints, output_hidden_states=output_hidden_states
        )

        keypoints = keypoint_encoding_output[0]

        # Early stop consists of stopping the forward pass through the transformer layers when the confidence of the
        # keypoints is above a certain threshold.
        do_early_stop = self.depth_confidence > 0
        # Keypoint pruning consists of removing keypoints from the input of the transformer layers when the confidence of
        # the keypoints is below a certain threshold.
        do_keypoint_pruning = self.width_confidence > 0

        early_stops_indices = []
        matches = []
        matching_scores = []
        final_pruned_keypoints_indices = []
        final_pruned_keypoints_iterations = []

        pruned_keypoints_indices = torch.arange(0, initial_num_keypoints, device=device).expand(batch_size * 2, -1)
        pruned_keypoints_iterations = torch.ones_like(pruned_keypoints_indices)

        for layer_index in range(self.num_layers):
            input_shape = descriptors.size()
            if mask is not None:
                extended_attention_mask = self.get_extended_attention_mask(mask, input_shape)
            else:
                extended_attention_mask = torch.ones((batch_size, input_shape[-2]), device=keypoints.device)
            layer_output = self.transformer_layers[layer_index](
                descriptors,
                keypoints,
                attention_mask=extended_attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            descriptors, hidden_states, attention = layer_output
            if output_hidden_states:
                all_hidden_states = all_hidden_states + hidden_states
            if output_attentions:
                all_attentions = all_attentions + attention

            if do_early_stop:
                if layer_index < self.num_layers - 1:
                    # Get the confidence of the keypoints for the current layer
                    keypoint_confidences = self.token_confidence[layer_index](descriptors)

                    # Determine which pairs of images should be early stopped based on the confidence of the keypoints for
                    # the current layer.
                    early_stopped_pairs = self._get_early_stopped_image_pairs(
                        keypoint_confidences, layer_index, mask, num_points=num_points_per_pair
                    )
                else:
                    # Early stopping always occurs at the last layer
                    early_stopped_pairs = torch.ones(batch_size, dtype=torch.bool)

                if torch.any(early_stopped_pairs):
                    # If a pair of images is considered early stopped, we compute the matches for the remaining
                    # keypoints and stop the forward pass through the transformer layers for this pair of images.
                    early_stops = early_stopped_pairs.repeat_interleave(2)
                    _, layer_num_keypoints = mask.shape
                    early_stopped_image_indices = image_indices[early_stops]
                    early_stopped_matches, early_stopped_matching_scores = self._get_keypoint_matching(
                        descriptors, mask, layer_index, early_stops=early_stops
                    )
                    early_stops_indices.extend(list(early_stopped_image_indices))
                    matches.extend(list(early_stopped_matches))
                    matching_scores.extend(list(early_stopped_matching_scores))
                    if do_keypoint_pruning:
                        final_pruned_keypoints_indices.extend(list(pruned_keypoints_indices[early_stops]))
                        final_pruned_keypoints_iterations.extend(list(pruned_keypoints_iterations[early_stops]))

                    # Remove image pairs that have been early stopped from the forward pass
                    num_points_per_pair = num_points_per_pair[~early_stopped_pairs]
                    descriptors, keypoints, mask, image_indices = tuple(
                        (tensor[~early_stops] for tensor in [descriptors, keypoints, mask, image_indices])
                    )
                    if do_keypoint_pruning:
                        pruned_keypoints_indices, pruned_keypoints_iterations, keypoint_confidences = tuple(
                            (
                                tensor[~early_stops]
                                for tensor in [
                                    pruned_keypoints_indices,
                                    pruned_keypoints_iterations,
                                    keypoint_confidences,
                                ]
                            )
                        )
                # If all pairs of images are early stopped, we stop the forward pass through the transformer
                # layers for all pairs of images.
                if torch.all(early_stopped_pairs):
                    break

            if do_keypoint_pruning:
                # Prune keypoints from the input of the transformer layers for the next iterations if the confidence of
                # the keypoints is below a certain threshold.
                descriptors, keypoints, pruned_keypoints_indices, mask, pruned_keypoints_iterations = (
                    self._do_layer_keypoint_pruning(
                        descriptors,
                        keypoints,
                        mask,
                        pruned_keypoints_indices,
                        pruned_keypoints_iterations,
                        keypoint_confidences,
                        layer_index,
                    )
                )

        if do_early_stop and do_keypoint_pruning:
            # Concatenate early stopped outputs together and perform final keypoint pruning
            final_pruned_keypoints_indices, final_pruned_keypoints_iterations, matches, matching_scores = (
                self._concat_early_stopped_outputs(
                    early_stops_indices,
                    final_pruned_keypoints_indices,
                    final_pruned_keypoints_iterations,
                    matches,
                    matching_scores,
                )
            )
            matches, matching_scores = self._do_final_keypoint_pruning(
                final_pruned_keypoints_indices,
                matches,
                matching_scores,
                initial_num_keypoints,
            )
        else:
            matches, matching_scores = self._get_keypoint_matching(descriptors, mask, self.num_layers - 1)
            final_pruned_keypoints_iterations = torch.ones_like(matching_scores) * self.num_layers

        final_pruned_keypoints_iterations = final_pruned_keypoints_iterations.reshape(
            batch_size, 2, initial_num_keypoints
        )

        return (
            matches,
            matching_scores,
            final_pruned_keypoints_iterations,
            all_hidden_states,
            all_attentions,
        )

    def _concat_early_stopped_outputs(
        self,
        early_stops_indices,
        final_pruned_keypoints_indices,
        final_pruned_keypoints_iterations,
        matches,
        matching_scores,
    ):
        early_stops_indices = torch.stack(early_stops_indices)
        matches, final_pruned_keypoints_indices = (
            pad_sequence(tensor, batch_first=True, padding_value=-1)
            for tensor in [matches, final_pruned_keypoints_indices]
        )
        matching_scores, final_pruned_keypoints_iterations = (
            pad_sequence(tensor, batch_first=True, padding_value=0)
            for tensor in [matching_scores, final_pruned_keypoints_iterations]
        )
        matches, matching_scores, final_pruned_keypoints_indices, final_pruned_keypoints_iterations = (
            tensor[early_stops_indices]
            for tensor in [
                matches,
                matching_scores,
                final_pruned_keypoints_indices,
                final_pruned_keypoints_iterations,
            ]
        )
        return final_pruned_keypoints_indices, final_pruned_keypoints_iterations, matches, matching_scores

    @add_start_docstrings_to_model_forward(LIGHTGLUE_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LightGlueKeypointMatchingOutput]:
        loss = None
        if labels is not None:
            raise ValueError("LightGlue is not trainable, no labels should be provided.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if len(pixel_values.size()) != 5 or pixel_values.size(1) != 2:
            raise ValueError("Input must be a 5D tensor of shape (batch_size, 2, num_channels, height, width)")

        batch_size, _, channels, height, width = pixel_values.shape

        list_keypoint_detection = [self.keypoint_detector(image_pair) for image_pair in pixel_values]

        max_keypoints = max([keypoint_detection[0].shape[1] for keypoint_detection in list_keypoint_detection])
        keypoints = torch.zeros(
            (batch_size, 2, max_keypoints, 2), device=pixel_values.device, dtype=pixel_values.dtype
        )
        descriptors = torch.zeros(
            (batch_size, 2, max_keypoints, self.config.descriptor_dim),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        mask = torch.zeros((batch_size, 2, max_keypoints), device=pixel_values.device, dtype=torch.int)

        for i, keypoint_detection_output in enumerate(list_keypoint_detection):
            _keypoints, _scores, _descriptors, _mask = keypoint_detection_output[:4]
            keypoints[i, :, : _keypoints.shape[1], :] = _keypoints
            descriptors[i, :, : _descriptors.shape[1], :] = _descriptors
            mask[i, :, : _mask.shape[1]] = _mask

        absolute_keypoints = keypoints.clone()
        absolute_keypoints[:, :, :, 0] = absolute_keypoints[:, :, :, 0] * width
        absolute_keypoints[:, :, :, 1] = absolute_keypoints[:, :, :, 1] * height

        matches, matching_scores, prune, hidden_states, attentions = self._match_image_pair(
            absolute_keypoints,
            descriptors,
            height,
            width,
            mask=mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if not return_dict:
            return tuple(
                v
                for v in [matches, matching_scores, keypoints, prune, mask, hidden_states, attentions]
                if v is not None
            )

        return LightGlueKeypointMatchingOutput(
            loss=loss,
            matches=matches,
            matching_scores=matching_scores,
            keypoints=keypoints,
            prune=prune,
            mask=mask,
            hidden_states=hidden_states,
            attentions=attentions,
        )


__all__ = ["LightGluePreTrainedModel", "LightGlueForKeypointMatching"]

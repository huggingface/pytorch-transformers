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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from transformers import PreTrainedModel, add_start_docstrings

from ...utils import ModelOutput, add_start_docstrings_to_model_forward, is_flash_attn_2_available, logging
from ..auto import AutoModelForKeypointDetection
from .configuration_lightglue import LightGlueConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC_ = "LightGlueConfig"

_CHECKPOINT_FOR_DOC_ = "stevenbucaille/lightglue"


def concat_pairs(tensor_tuple0: Tuple[torch.Tensor], tensor_tuple1: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
    """
    Concatenate two tuples of tensors pairwise

    Args:
        tensor_tuple0 (`Tuple[torch.Tensor]`): Tuple of tensors.
        tensor_tuple1 (`Tuple[torch.Tensor]`): Tuple of tensors.

    Returns:
        (`Tuple[torch.Tensor]`): Tuple of concatenated tensors.
    """
    return tuple([torch.cat([tensor0, tensor1]) for tensor0, tensor1 in zip(tensor_tuple0, tensor_tuple1)])


def normalize_keypoints(keypoints: torch.Tensor, height: int, width: int) -> torch.Tensor:
    size = torch.tensor([width, height], device=keypoints.device, dtype=keypoints.dtype)[None]
    shift = size / 2
    scale = size.max(-1).values / 2
    keypoints = (keypoints - shift[..., None, :]) / scale[..., None, None]
    return keypoints


def log_sinkhorn_iterations(
    log_cost_matrix: torch.Tensor,
    log_source_distribution: torch.Tensor,
    log_target_distribution: torch.Tensor,
    num_iterations: int,
) -> torch.Tensor:
    """
    Perform Sinkhorn Normalization in Log-space for stability

    Args:
        log_cost_matrix (`torch.Tensor` of shape `(batch_size, num_rows, num_columns)`): Logarithm of the cost matrix.
        log_source_distribution (`torch.Tensor` of shape `(batch_size, num_rows)`): Logarithm of the source
        distribution.
        log_target_distribution (`torch.Tensor` of shape `(batch_size, num_columns)`): Logarithm of the target
        distribution.

    Returns:
        log_cost_matrix (`torch.Tensor` of shape `(batch_size, num_rows, num_columns)`): Logarithm of the optimal
        transport matrix.
    """
    log_u_scaling = torch.zeros_like(log_source_distribution)
    log_v_scaling = torch.zeros_like(log_target_distribution)
    for _ in range(num_iterations):
        log_u_scaling = log_source_distribution - torch.logsumexp(log_cost_matrix + log_v_scaling.unsqueeze(1), dim=2)
        log_v_scaling = log_target_distribution - torch.logsumexp(log_cost_matrix + log_u_scaling.unsqueeze(2), dim=1)
    return log_cost_matrix + log_u_scaling.unsqueeze(2) + log_v_scaling.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, reg_param: torch.Tensor, iterations: int) -> torch.Tensor:
    """
    Perform Differentiable Optimal Transport in Log-space for stability

    Args:
        scores: (`torch.Tensor` of shape `(batch_size, num_rows, num_columns)`): Cost matrix.
        reg_param: (`torch.Tensor` of shape `(batch_size, 1, 1)`): Regularization parameter.
        iterations: (`int`): Number of Sinkhorn iterations.

    Returns:
        log_optimal_transport_matrix: (`torch.Tensor` of shape `(batch_size, num_rows, num_columns)`): Logarithm of the
        optimal transport matrix.
    """
    batch_size, num_rows, num_columns = scores.shape
    one_tensor = scores.new_tensor(1)
    num_rows_tensor, num_columns_tensor = (num_rows * one_tensor).to(scores), (num_columns * one_tensor).to(scores)

    source_reg_param = reg_param.expand(batch_size, num_rows, 1)
    target_reg_param = reg_param.expand(batch_size, 1, num_columns)
    reg_param = reg_param.expand(batch_size, 1, 1)

    couplings = torch.cat([torch.cat([scores, source_reg_param], -1), torch.cat([target_reg_param, reg_param], -1)], 1)

    log_normalization = -(num_rows_tensor + num_columns_tensor).log()
    log_source_distribution = torch.cat(
        [log_normalization.expand(num_rows), num_columns_tensor.log()[None] + log_normalization]
    )
    log_target_distribution = torch.cat(
        [log_normalization.expand(num_columns), num_rows_tensor.log()[None] + log_normalization]
    )
    log_source_distribution, log_target_distribution = (
        log_source_distribution[None].expand(batch_size, -1),
        log_target_distribution[None].expand(batch_size, -1),
    )

    log_optimal_transport_matrix = log_sinkhorn_iterations(
        couplings, log_source_distribution, log_target_distribution, num_iterations=iterations
    )
    log_optimal_transport_matrix = log_optimal_transport_matrix - log_normalization  # multiply probabilities by M+N
    return log_optimal_transport_matrix


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


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
        mask (`torch.BoolTensor` of shape `(batch_size, num_keypoints)`):
            Mask indicating which values in matches and matching_scores are keypoint matching information.
        matches (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Index of keypoint matched in the other image.
        matching_scores (`torch.FloatTensor` of shape `(batch_size, 2, num_matches)`):
            Scores of predicted matches.
        keypoints (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`):
            Absolute (x, y) coordinates of predicted keypoints in a given image.
        prune (`torch.IntTensor` of shape `(batch_size, num_keypoints)`):
            Pruning mask indicating which keypoints are removed and at which layer.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of each stage) of shape `(batch_size, 2, num_channels,
            num_keypoints)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, 2, num_heads, num_keypoints,
            num_keypoints)`.
    """

    loss: Optional[torch.FloatTensor] = None
    mask: torch.FloatTensor = None
    matches: torch.FloatTensor = None
    matching_scores: torch.FloatTensor = None
    keypoints: torch.FloatTensor = None
    prune: torch.IntTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def rotary_position_embedding(encoded_keypoints: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    first = state * encoded_keypoints[0]
    rotated = rotate_half(state)
    second = rotated * encoded_keypoints[1]
    return first + second


def eager_attention(q, k, v, mask=None):
    s = q.shape[-1] ** 0.5
    scores = torch.einsum("...id,...jd->...ij", q, k) / s
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
    attention = nn.functional.softmax(scores, -1, dtype=scores.dtype)
    output = torch.einsum("...ij,...jd->...id", attention, v)
    return output, attention


class LightGluePositionalEncoder(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.projector = nn.Linear(2, config.descriptor_dim // config.num_heads // 2, bias=False)
        self.gamma = 1.0
        nn.init.normal_(self.projector.weight.data, mean=0, std=self.gamma**-2)

    def forward(
        self, keypoints: torch.Tensor, output_hidden_states: Optional[bool] = False
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        projected_keypoints = self.projector(keypoints)
        cosines, sines = torch.cos(projected_keypoints), torch.sin(projected_keypoints)
        embeddings = torch.stack([cosines, sines], 0).unsqueeze(-3)
        embeddings = embeddings.repeat_interleave(2, dim=-1)
        output = (embeddings, projected_keypoints) if output_hidden_states else (embeddings,)
        return output


class LightGlueAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, attention = eager_attention(q, k, v, mask=mask)
        return output, attention


class LightGlueFlashAttention(LightGlueAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        attn_output = flash_attn_func(q, k, v)
        return attn_output, None


class LightGlueSdpaAttention(LightGlueAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        mask = mask.to(torch.bool)
        mask = mask.unsqueeze(1).unsqueeze(1)
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return attn_output, None


LIGHTGLUE_ATTENTION_CLASSES = {
    "eager": LightGlueAttention,
    "flash_attention_2": LightGlueFlashAttention,
    "sdpa": LightGlueSdpaAttention,
}


class LightGlueSelfAttentionBlock(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.num_heads = config.num_heads
        embeddings_dim = config.descriptor_dim

        self.Wqkv = nn.Linear(embeddings_dim, embeddings_dim * 3, bias=True)
        self.attention = LIGHTGLUE_ATTENTION_CLASSES[config._attn_implementation](config=config)
        self.output_projection = nn.Linear(embeddings_dim, embeddings_dim, bias=True)

        self.ffn = nn.ModuleList(
            [
                nn.Linear(2 * embeddings_dim, 2 * embeddings_dim),
                nn.LayerNorm(2 * embeddings_dim, elementwise_affine=True),
                nn.GELU(),
                nn.Linear(2 * embeddings_dim, embeddings_dim),
            ]
        )

    def forward_ffn(self, input, output_hidden_states: Optional[bool] = False):
        all_hidden_states = () if output_hidden_states else None
        for layer in self.ffn:
            input = layer(input)
            if output_hidden_states and isinstance(layer, nn.Linear):
                all_hidden_states = all_hidden_states + (input,)
        output = input
        return output, all_hidden_states

    def forward(
        self,
        descriptors: torch.Tensor,
        keypoints: torch.Tensor,
        mask: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        qkv = self.Wqkv(descriptors)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = rotary_position_embedding(keypoints, q)
        k = rotary_position_embedding(keypoints, k)

        if output_attentions and isinstance(self.attention, (LightGlueSdpaAttention, LightGlueFlashAttention)):
            context, attention = eager_attention(q, k, v, mask=mask)
        else:
            context, attention = self.attention(q, k, v, mask=mask)
        context = context.transpose(1, 2).flatten(start_dim=-2)
        message = self.output_projection(context)

        input = torch.cat([descriptors, message], -1)
        ffn_output = self.forward_ffn(input, output_hidden_states=output_hidden_states)

        last_hidden_state = ffn_output[0]
        hidden_states = ffn_output[1]
        descriptors = descriptors + last_hidden_state

        output = (descriptors, hidden_states, (attention,)) if output_attentions else (descriptors, hidden_states)

        return output


class LightGlueCrossAttentionBlock(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.num_heads = config.num_heads
        self.embeddings_dim = config.descriptor_dim
        head_dim = config.descriptor_dim // self.num_heads
        self.scale = head_dim**-0.5
        inner_dim = head_dim * self.num_heads
        self.to_qk = nn.Linear(self.embeddings_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(self.embeddings_dim, inner_dim, bias=True)
        self.attention = LIGHTGLUE_ATTENTION_CLASSES[config._attn_implementation](config=config)
        self.to_out = nn.Linear(inner_dim, self.embeddings_dim, bias=True)
        self.ffn = nn.ModuleList(
            [
                nn.Linear(2 * self.embeddings_dim, 2 * self.embeddings_dim),
                nn.LayerNorm(2 * self.embeddings_dim, elementwise_affine=True),
                nn.GELU(),
                nn.Linear(2 * self.embeddings_dim, self.embeddings_dim),
            ]
        )

    def input_projection(self, descriptors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qk = self.to_qk(descriptors)
        v = self.to_v(descriptors)
        qk = qk.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        v = v.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        return qk, v

    def forward_ffn(
        self, input, output_hidden_states: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        all_hidden_states = () if output_hidden_states else None
        for layer in self.ffn:
            input = layer(input)
            if output_hidden_states and isinstance(layer, nn.Linear):
                all_hidden_states = all_hidden_states + (input,)
        output = input
        return output, all_hidden_states

    def forward_message(
        self, descriptors, context, output_hidden_states: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        context = context.transpose(1, 2).flatten(start_dim=-2)
        message = self.to_out(context)
        input = torch.cat([descriptors, message], -1)
        ffn_output = self.forward_ffn(input, output_hidden_states=output_hidden_states)

        last_hidden_state = ffn_output[0]
        hidden_states = ffn_output[1]

        descriptors = descriptors + last_hidden_state

        output = (descriptors, hidden_states) if output_hidden_states else (descriptors,)
        return output

    def forward(
        self,
        descriptors: torch.Tensor,
        mask: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (descriptors,)

        q, v = self.input_projection(descriptors)
        batch_size, num_head, seq_len, head_dim = q.shape
        k = q.reshape(-1, 2, num_head, seq_len, head_dim).flip(1).reshape(batch_size, num_head, seq_len, head_dim)
        v = v.reshape(-1, 2, num_head, seq_len, head_dim).flip(1).reshape(batch_size, num_head, seq_len, head_dim)
        attention_mask = mask.reshape(-1, 2, seq_len).flip(1).reshape(batch_size, seq_len)
        if output_attentions and isinstance(self.attention, (LightGlueSdpaAttention, LightGlueFlashAttention)):
            context, attention = eager_attention(q, k, v, mask=attention_mask)
        else:
            context, attention = self.attention(q, k, v, mask=attention_mask)

        message_output = self.forward_message(descriptors, context, output_hidden_states=output_hidden_states)

        descriptors = message_output[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + message_output[1]
        if output_attentions:
            all_attentions = all_attentions + (attention,)

        return descriptors, all_hidden_states, all_attentions


class LightGlueTransformerLayer(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()
        self.descriptor_dim = config.descriptor_dim
        self.self_attention_block = LightGlueSelfAttentionBlock(config)
        self.cross_attention_block = LightGlueCrossAttentionBlock(config)

    def forward(
        self,
        descriptors: torch.Tensor,
        keypoints: torch.Tensor,
        mask: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if output_hidden_states:
            batch_size, num_keypoints, _ = descriptors.shape
            all_hidden_states = all_hidden_states + (
                descriptors.reshape(batch_size, num_keypoints, self.descriptor_dim),
            )

        self_attention_output = self.self_attention_block(
            descriptors,
            keypoints,
            mask=mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        descriptors = self_attention_output[0]

        output = self.cross_attention_block(
            descriptors,
            mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        descriptors = output[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + self_attention_output[1]
            all_hidden_states = all_hidden_states + output[1]
        if output_attentions:
            all_attentions = all_attentions + self_attention_output[2]
            all_attentions = all_attentions + output[2]

        return descriptors, all_hidden_states, all_attentions


def sigmoid_log_double_softmax(
    similarity: torch.Tensor, matchability_0: torch.Tensor, matchability_1: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    batch_size, num_keypoints_0, num_keypoints_1 = similarity.shape
    certainties = nn.functional.logsigmoid(matchability_0) + nn.functional.logsigmoid(matchability_1).transpose(1, 2)
    scores0 = nn.functional.log_softmax(similarity, 2)
    scores1 = nn.functional.log_softmax(similarity.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = similarity.new_full((batch_size, num_keypoints_0 + 1, num_keypoints_1 + 1), 0)
    scores[:, :num_keypoints_0, :num_keypoints_1] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = nn.functional.logsigmoid(-matchability_0.squeeze(-1))
    scores[:, -1, :-1] = nn.functional.logsigmoid(-matchability_1.squeeze(-1))
    return scores


class LightGlueMatchAssignmentLayer(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.descriptor_dim = config.descriptor_dim
        self.final_projection = nn.Linear(self.descriptor_dim, self.descriptor_dim, bias=True)
        self.matchability = nn.Linear(self.descriptor_dim, 1, bias=True)

    def forward(self, descriptors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m_descriptors = self.final_projection(descriptors)
        m_descriptors = m_descriptors / self.descriptor_dim**0.25
        matchability = self.matchability(descriptors)
        batch_size, num_keypoints, descriptor_dim = descriptors.shape
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

        matchability = matchability.reshape(batch_size // 2, 2, num_keypoints, 1)
        matchability_0 = matchability[:, 0]
        matchability_1 = matchability[:, 1]
        scores = sigmoid_log_double_softmax(similarity, matchability_0, matchability_1, mask)
        return scores

    def get_matchability(self, descriptors: torch.Tensor) -> torch.Tensor:
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

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm, nn.Conv1d]) -> None:
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
            Pixel values. Pixel values can be obtained using [`SuperPointImageProcessor`]. See
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
            torch.Tensor([self.confidence_threshold(i) for i in range(self.num_layers)]),
        )

        self.post_init()

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.num_layers)
        return np.clip(threshold, 0, 1)

    def keypoint_processing(
        self, descriptors, keypoints, output_hidden_states: Optional[bool] = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        descriptors = descriptors.detach().contiguous()
        projected_descriptors = self.input_projection(descriptors)
        keypoint_encoding_output = self.positional_encoder(keypoints, output_hidden_states=output_hidden_states)
        return projected_descriptors, keypoint_encoding_output

    def check_if_stop(
        self, confidences: torch.Tensor, layer_index: int, mask: torch.Tensor, num_points: torch.Tensor
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        batch_size, _ = mask.shape
        confidences = confidences.masked_fill(mask == 0, 1)
        confidences = confidences.reshape(batch_size // 2, -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum(dim=1) / num_points
        return ratio_confident > self.depth_confidence

    def get_pruning_mask(self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def do_final_point_pruning(self, indices, matches, matching_scores, num_keypoints):
        batch_size, _ = indices.shape
        indices = indices.reshape(batch_size // 2, 2, -1)
        indices0, indices1 = indices[:, 0], indices[:, 1]
        matches = matches.reshape(batch_size // 2, 2, -1)
        matches0, matches1 = matches[:, 0], matches[:, 1]
        matching_scores = matching_scores.reshape(batch_size // 2, 2, -1)
        matching_scores0, matching_scores1 = matching_scores[:, 0], matching_scores[:, 1]
        _matches = torch.full((batch_size // 2, 2, num_keypoints), -1, device=indices.device, dtype=matches.dtype)
        _matching_scores = torch.zeros(
            (batch_size // 2, 2, num_keypoints), device=indices.device, dtype=matching_scores.dtype
        )
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

    def do_layer_point_pruning(self, descriptors, keypoints, mask, i, indices, prune_output, token):
        scores = self.match_assignment_layers[i].get_matchability(descriptors)
        prune_mask = self.get_pruning_mask(token, scores, i)
        prune_mask = prune_mask.masked_fill(mask == 0, 0)
        _, batch_size, _, _, keypoint_dim = keypoints.shape

        list_kept_indices = []
        for i in range(batch_size):
            keep = torch.where(prune_mask[i])[0]
            list_kept_indices.append(keep)

        max_number_kept_keypoints = max(len(x) for x in list_kept_indices)
        pruned_indices = torch.full((batch_size, max_number_kept_keypoints), -1, device=descriptors.device)
        pruned_mask = torch.zeros((batch_size, max_number_kept_keypoints), device=descriptors.device)
        pruned_descriptors = torch.zeros(
            (batch_size, max_number_kept_keypoints, self.descriptor_dim),
            device=descriptors.device,
            dtype=descriptors.dtype,
        )
        pruned_keypoints = torch.zeros(
            (2, batch_size, 1, max_number_kept_keypoints, keypoint_dim), device=keypoints.device, dtype=keypoints.dtype
        )
        for i, keep in enumerate(list_kept_indices):
            num_kept_keypoints = len(keep)
            pruned_indices[i, :num_kept_keypoints] = indices[i].index_select(-1, keep)
            pruned_descriptors[i, :num_kept_keypoints] = descriptors[i].index_select(-2, keep)
            pruned_mask[i, :num_kept_keypoints] = 1
            pruned_keypoints[:, i, :, :num_kept_keypoints] = keypoints[:, i].index_select(-2, keep)
            prune_output[i, pruned_indices[i, :num_kept_keypoints]] += 1
        return pruned_descriptors, pruned_keypoints, pruned_indices, pruned_mask, prune_output

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
                all_hidden_states,
                all_attentions,
            )

        device = keypoints.device
        batch_size, _, initial_num_keypoints, _ = keypoints.shape
        num_points_per_pair = torch.sum(mask.reshape(batch_size, -1), dim=1)
        # (batch_size, 2, num_keypoints, 2) -> (batch_size * 2, num_keypoints, 2)
        keypoints = keypoints.reshape(batch_size * 2, initial_num_keypoints, 2)
        mask = mask.reshape(batch_size * 2, initial_num_keypoints)
        descriptors = descriptors.reshape(batch_size * 2, initial_num_keypoints, self.descriptor_dim)
        pair_indices = torch.arange(batch_size * 2, device=device)
        # Keypoint normalization
        keypoints = normalize_keypoints(keypoints, height, width)

        descriptors, keypoint_encoding_output = self.keypoint_processing(
            descriptors, keypoints, output_hidden_states=output_hidden_states
        )

        encoded_keypoints = keypoint_encoding_output[0]

        do_early_stop = self.depth_confidence > 0
        do_point_pruning = self.width_confidence > 0

        if do_point_pruning:
            prune_indices = torch.arange(0, initial_num_keypoints, device=device).expand(batch_size * 2, -1)
            prune_output = torch.ones_like(prune_indices)
        if do_early_stop:
            matches = torch.full((batch_size * 2, initial_num_keypoints), -1, device=device)
            matching_scores = torch.zeros(
                (batch_size * 2, initial_num_keypoints), device=device, dtype=descriptors.dtype
            )
            if do_point_pruning:
                final_prune_indices = torch.full((batch_size * 2, initial_num_keypoints), -1, device=device)
                final_prune_output = torch.zeros(
                    (batch_size * 2, initial_num_keypoints), device=device, dtype=prune_output.dtype
                )

        for layer_index in range(self.num_layers):
            transformer_output = self.transformer_layers[layer_index](
                descriptors,
                encoded_keypoints,
                mask=mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            descriptors = transformer_output[0]
            hidden_states = transformer_output[1]
            attention = transformer_output[2]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + hidden_states
            if output_attentions:
                all_attentions = all_attentions + attention

            if do_early_stop:
                if layer_index < self.num_layers - 1:
                    token = self.token_confidence[layer_index](descriptors)
                    early_stops_pair = self.check_if_stop(token, layer_index, mask, num_points=num_points_per_pair)
                else:
                    early_stops_pair = torch.ones(batch_size, dtype=torch.bool)
                if torch.any(early_stops_pair):
                    _, layer_num_keypoints = mask.shape
                    early_stops = early_stops_pair.repeat_interleave(2)
                    early_stopped_pair_indices = pair_indices[early_stops]
                    early_stopped_descriptors = descriptors[early_stops]
                    early_stopped_mask = mask[early_stops]
                    scores = self.match_assignment_layers[layer_index](early_stopped_descriptors, early_stopped_mask)
                    early_stopped_matches, early_stopped_matching_scores = filter_matches(
                        scores, self.filter_threshold
                    )
                    matches[early_stopped_pair_indices, :layer_num_keypoints] = early_stopped_matches
                    matching_scores[early_stopped_pair_indices, :layer_num_keypoints] = early_stopped_matching_scores
                    if do_point_pruning:
                        final_prune_indices[early_stopped_pair_indices, :layer_num_keypoints] = prune_indices[
                            early_stops
                        ]
                        final_prune_output[early_stopped_pair_indices] = prune_output[early_stops]
                    if torch.all(early_stops):
                        break
                    num_points_per_pair = num_points_per_pair[~early_stops_pair]
                    descriptors = descriptors[~early_stops]
                    encoded_keypoints = encoded_keypoints[:, ~early_stops]
                    mask = mask[~early_stops]
                    if do_point_pruning:
                        prune_indices = prune_indices[~early_stops]
                        prune_output = prune_output[~early_stops]
                        token = token[~early_stops]
                    pair_indices = pair_indices[~early_stops]

            if do_point_pruning:
                descriptors, encoded_keypoints, prune_indices, mask, prune_output = self.do_layer_point_pruning(
                    descriptors, encoded_keypoints, mask, layer_index, prune_indices, prune_output, token
                )

        if do_early_stop and do_point_pruning:
            # Concatenate early stopped outputs with the final outputs
            prune_indices = final_prune_indices
            prune_output = final_prune_output
            matches, matching_scores = self.do_final_point_pruning(
                prune_indices, matches, matching_scores, initial_num_keypoints
            )
        else:
            scores = self.match_assignment_layers[layer_index](descriptors, mask)
            matches, matching_scores = filter_matches(scores, self.filter_threshold)
            prune_output = torch.ones_like(matching_scores) * self.num_layers

        prune_output = prune_output.reshape(batch_size, 2, initial_num_keypoints)

        return (
            matches,
            matching_scores,
            prune_output,
            all_hidden_states,
            all_attentions,
        )

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

        matches, matching_scores, prune, hidden_states, attentions = self._match_image_pair(
            keypoints,
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
                for v in [mask, matches, matching_scores, keypoints, prune, hidden_states, attentions]
                if v is not None
            )

        return LightGlueKeypointMatchingOutput(
            loss=loss,
            mask=mask,
            matches=matches,
            matching_scores=matching_scores,
            keypoints=keypoints,
            prune=prune,
            hidden_states=hidden_states,
            attentions=attentions,
        )

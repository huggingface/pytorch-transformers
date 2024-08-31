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
from typing import List, Optional, Tuple, Union

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

_CHECKPOINT_FOR_DOC_ = "stevenbucaille/superglue_indoor"


def concat_inconsistent_pairs(
    inconsistent_tensor0: Tuple[torch.Tensor], inconsistent_tensor1: Tuple[torch.Tensor]
) -> Tuple[torch.Tensor]:
    """
    Concatenate two tuples of tensors with inconsistent dimensions pairwise. We assume that each tuple has the same
    number of tensors and and the tensors are at least 3D. These tensors can be anything but this function is mainly
    used to concatenate hidden states and attention probabilities tensors of a keypoint matching model. The
    inconsistency comes from the dimension containing the number of keypoints.

    If the pair of tensors are hidden states, they are assumed to be of shape `(batch_size, num_keypoints0,
    num_channels)` and `(batch_size, num_keypoints1, num_channels)` and the output will then be of the form:
    `(2 * batch_size, max(num_keypoints0, num_keypoints1), num_channels)`
    If the pair of tensors are attention probabilities, they are assumed to be of shape `(batch_size, num_heads,
    num_keypoints0, num_keypoints0)` and `(batch_size, num_heads, num_keypoints1, num_keypoints1)` and the output will
    then be of the form:
    `(2 * batch_size, num_heads, max(num_keypoints0, num_keypoints1), max(num_keypoints0, num_keypoints1))`

    For `inconsistent_tensors0 = (tensor0_1, tensor0_1, ..., tensor0_N)` and `inconsistent_tensors1 = (tensor1_0,
    tensor1_1, ..., tensor1_N)`, the output will be `(concat(tensor0_0, tensor1_0), concat(tensor0_1, tensor1_1), ...,
    concat(tensor0_N, tensor1_N))`. The concatenation is done by padding the tensors from the tuple with the lower
    number of keypoints with zeros.

    Args:
        inconsistent_tensor0 (`Tuple[torch.Tensor]` of shape `(batch_size, num_keypoints0, num_channels)` or
        `torch.Tensor` of shape `(batch_size, num_heads, num_keypoints0, num_keypoints0)`): Tuple of tensors with
        inconsistent dimensions.
        inconsistent_tensor1 (`Tuple[torch.Tensor]` of shape `(batch_size, num_keypoints1, num_channels)` or
        `torch.Tensor` of shape `(batch_size, num_heads, num_keypoints1, num_keypoints1)`): Tuple of tensors with
        inconsistent dimensions.

    Returns:
        consistent_tensors (`Tuple[torch.Tensor]` of shape `(2 * batch_size, max(num_keypoints0, num_keypoints1),
        num_channels)` or `Tuple[torch.Tensor]` of shape `(2 * batch_size, num_heads,
        max(num_keypoints0, num_keypoints1) ,max(num_keypoints0, num_keypoints1))`):
        Tuple of zero padded tensors with consistent dimensions.
    """
    if len(inconsistent_tensor0) != len(inconsistent_tensor1):
        raise ValueError("The two tuples must contain the same number of tensors.")
    if len(inconsistent_tensor0[0].shape) < 3:
        raise ValueError("The tensors must be at least 3D.")
    consistent_tensors = ()
    for tensor0, tensor1 in zip(inconsistent_tensor0, inconsistent_tensor1):
        if tensor0.shape != tensor1.shape:
            squeeze_tensors = len(tensor0.shape) == 3 and len(tensor1.shape) == 3
            if squeeze_tensors:
                tensor0 = tensor0[..., None]
                tensor1 = tensor1[..., None]
            # max_dim1 = num_heads if tensor is attention probabilities else num_keypoints
            max_dim1 = max(tensor0.shape[1], tensor1.shape[1])
            # max_dim2 = max_dim3 = num_keypoints if tensor is attention probabilities else num_channels
            max_dim2 = max(tensor0.shape[2], tensor1.shape[2])
            # max_dim3 = num_keypoints if tensor is attention probabilities else 1
            max_dim3 = max(tensor0.shape[3], tensor1.shape[3])
            consistent_tensor = torch.zeros(2, max_dim1, max_dim2, max_dim3, device=tensor0.device)
            consistent_tensor[0, : tensor0.shape[1], : tensor0.shape[2], : tensor0.shape[3]] = tensor0
            consistent_tensor[1, : tensor1.shape[1], : tensor1.shape[2], : tensor1.shape[3]] = tensor1
            if squeeze_tensors:
                consistent_tensor = consistent_tensor.squeeze(-1)
            consistent_tensors = consistent_tensors + (consistent_tensor,)
        else:
            consistent_tensors = consistent_tensors + (torch.cat([tensor0, tensor1]),)
    return consistent_tensors


def stack_inconsistent_tensor_list(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack a list of tensors with inconsistent dimensions. We assume that each tensor is at least 3D. The tensors can be
    anything but this function is mainly used to stack hidden states tensors of a keypoint matching model. The
    inconsistency comes from the dimension containing the number of keypoints.

    If the tensors are hidden states, they are assumed to be of shape `(batch_size, num_keypoints0, num_channels),
    (batch_size, num_keypoints1, num_channels), ..., (batch_size, num_keypointsN, num_channels)` and the output will
    then be of the form: `(N, batch_size, max(num_keypoints0, num_keypoints1, ..., num_keypointsN), num_channels)`.
    If the tensors are attention probabilities, they are assumed to be of shape `(batch_size, num_heads, num_keypoints0,
    num_keypoints0), (batch_size, num_heads, num_keypoints1, num_keypoints1), ..., (batch_size, num_heads,
    num_keypointsN, num_keypointsN)` and the output will then be of the form: `(N, batch_size, num_heads,
    max(num_keypoints0, num_keypoints1, ..., num_keypointsN), max(num_keypoints0, num_keypoints1, ..., num_keypointsN))`.

    For `tensor_list = [tensor0, tensor1, ..., tensorN]` and `max_number_of_keypoints = max(num_keypoints0,
    num_keypoints1, ..., num_keypointsN)`, the output will be a tensor of shape `(N, batch_size,
    max_number_of_keypoints, num_channels)`  or `(N, batch_size, num_heads,max_number_of_keypoints,
    max_number_of_keypoints)`. The stacking is done by padding the tensors with the lower number of keypoints with
    zeros.


    Args:
        tensor_list (`List[torch.Tensor]` of shape `(batch_size, num_keypoints, num_channels)` or `List[torch.Tensor]`
        of shape `(batch_size, num_heads, num_keypoints, num_keypoints)`): List of tensors with inconsistent dimensions.

    Returns:
        (`torch.Tensor` of shape `(N, batch_size, max(num_keypoints0, num_keypoints1, ..., num_keypointsN),
        num_channels)` or `torch.Tensor` of shape `(N, batch_size, num_heads, max(num_keypoints0, num_keypoints1, ...,
        num_keypointsN), max(num_keypoints0, num_keypoints1, ..., num_keypointsN))`): Stacked tensors with consistent
        dimensions.
    """
    current_shape = tensor_list[0].shape
    all_same_shape = all(tensor.shape == current_shape for tensor in tensor_list)
    if all_same_shape:
        return torch.stack(tensor_list, dim=0)

    squeeze_tensors = len(tensor_list[0].shape) == 3
    if squeeze_tensors:
        tensor_list = [tensor[..., None] for tensor in tensor_list]

    max_dim1 = max(tensor.shape[1] for tensor in tensor_list)
    max_dim2 = max(tensor.shape[2] for tensor in tensor_list)
    max_dim3 = max(tensor.shape[3] for tensor in tensor_list)
    stacked_tensors = torch.zeros(len(tensor_list), 2, max_dim1, max_dim2, max_dim3, device=tensor_list[0].device)
    for i, tensor in enumerate(tensor_list):
        stacked_tensors[i, :, : tensor.shape[1], : tensor.shape[2], : tensor.shape[3]] = tensor
    if squeeze_tensors:
        stacked_tensors = stacked_tensors.squeeze(-1)
    return stacked_tensors


def batch_inconsistent_tensor_list(tensor_list: List[Tuple[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Batch a list of tuples of tensors with inconsistent dimensions.
    For a list of N tuples of T tensors : `[(tensor0_0, tensor0_1, ..., tensor0_T), (tensor1_0, tensor1_1, ...,
    tensor1_T), ..., (tensorN_0, tensorN_1, ..., tensorN_T)]`, the output will be a list of T stacked tensors:
    `[stack(tensor0_0, tensor1_0, ..., tensorN_0), stack(tensor0_1, tensor1_1, ..., tensorN_1), ..., stack(tensor0_T,
    tensor1_T, ..., tensorN_T)]`.

    Args:
        tensor_list (`List[Tuple[torch.Tensor]]`): List of tuples of tensors with inconsistent dimensions.

    Returns: (`List[torch.Tensor]`): List of tensors with consistent dimensions.
    """
    list_of_tuples = list(zip(*map(list, tensor_list)))
    return [stack_inconsistent_tensor_list(element) for element in list_of_tuples]


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
class KeypointMatchingOutput(ModelOutput):
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
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


def eager_attention(q, k, v):
    s = q.shape[-1] ** -0.5
    sim = torch.einsum("...id,...jd->...ij", q, k) * s
    attention = nn.functional.softmax(sim, -1)
    output = torch.einsum("...ij,...jd->...id", attention, v)
    return output, attention


class LightGluePositionalEncoder(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        M = 2 + 2 * config.add_scale_ori
        self.projector = nn.Linear(M, config.descriptor_dim // config.num_heads // 2, bias=False)
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

    def forward(self, q, k, v):
        output, attention = eager_attention(q, k, v)
        return output, attention


class LightGlueFlashAttention(LightGlueAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, q, k, v):
        attn_output = flash_attn_func(q, k, v)
        return attn_output, None


class LightGlueSdpaAttention(LightGlueAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, q, k, v):
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
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
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor], torch.Tensor]:
        qkv = self.Wqkv(descriptors)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(keypoints, q)
        k = apply_cached_rotary_emb(keypoints, k)

        if output_attentions and isinstance(self.attention, (LightGlueSdpaAttention, LightGlueFlashAttention)):
            context, attention = eager_attention(q, k, v)
        else:
            context, attention = self.attention(q, k, v)
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

    def input_projection(self, descriptors):
        qk = self.to_qk(descriptors)
        v = self.to_v(descriptors)
        qk = qk.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        v = v.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        return qk, v

    def forward_ffn(self, input, output_hidden_states: Optional[bool] = False):
        all_hidden_states = () if output_hidden_states else None
        for layer in self.ffn:
            input = layer(input)
            if output_hidden_states and isinstance(layer, nn.Linear):
                all_hidden_states = all_hidden_states + (input,)
        output = input
        return output, all_hidden_states

    def forward_message(self, descriptors, context, output_hidden_states: Optional[bool] = False):
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
        descriptors_0: torch.Tensor,
        descriptors_1: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            new_hidden_state = concat_inconsistent_pairs((descriptors_0,), (descriptors_1,))
            all_hidden_states = all_hidden_states + new_hidden_state

        qk0, v0 = self.input_projection(descriptors_0)
        qk1, v1 = self.input_projection(descriptors_1)

        if output_attentions and isinstance(self.attention, (LightGlueSdpaAttention, LightGlueFlashAttention)):
            context0, attention0 = eager_attention(qk0, qk1, v1)
            context1, attention1 = eager_attention(qk1, qk0, v0)
        else:
            context0, attention0 = self.attention(qk0, qk1, v1)
            context1, attention1 = self.attention(qk1, qk0, v0)

        message_output_0 = self.forward_message(descriptors_0, context0, output_hidden_states=output_hidden_states)
        message_output_1 = self.forward_message(descriptors_1, context1, output_hidden_states=output_hidden_states)

        descriptors_0 = message_output_0[0]
        descriptors_1 = message_output_1[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + concat_inconsistent_pairs(message_output_0[1], message_output_1[1])
        if output_attentions:
            all_attentions = all_attentions + concat_inconsistent_pairs((attention0,), (attention1,))

        return descriptors_0, descriptors_1, all_hidden_states, all_attentions


class LightGlueTransformerLayer(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.self_attention_block = LightGlueSelfAttentionBlock(config)
        self.cross_attention_block = LightGlueCrossAttentionBlock(config)

    def forward(
        self,
        descriptors_0: torch.Tensor,
        descriptors_1: torch.Tensor,
        keypoints0: torch.Tensor,
        keypoints1: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if output_hidden_states:
            new_hidden_state = concat_inconsistent_pairs((descriptors_0,), (descriptors_1,))
            all_hidden_states = all_hidden_states + new_hidden_state
        self_attention_output0 = self.self_attention_block(
            descriptors_0, keypoints0, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        self_attention_output1 = self.self_attention_block(
            descriptors_1, keypoints1, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )

        descriptors_0 = self_attention_output0[0]
        descriptors_1 = self_attention_output1[0]

        output = self.cross_attention_block(
            descriptors_0,
            descriptors_1,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        descriptors_0 = output[0]
        descriptors_1 = output[1]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + concat_inconsistent_pairs(
                self_attention_output0[1], self_attention_output1[1]
            )
            all_hidden_states = all_hidden_states + output[2]
        if output_attentions:
            all_attentions = all_attentions + concat_inconsistent_pairs(
                self_attention_output0[2], self_attention_output1[2]
            )
            all_attentions = all_attentions + output[3]

        return descriptors_0, descriptors_1, all_hidden_states, all_attentions


def sigmoid_log_double_softmax(sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = nn.functional.logsigmoid(z0) + nn.functional.logsigmoid(z1).transpose(1, 2)
    scores0 = nn.functional.log_softmax(sim, 2)
    scores1 = nn.functional.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = nn.functional.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = nn.functional.logsigmoid(-z1.squeeze(-1))
    return scores


class LightGlueMatchAssignmentLayer(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.descriptor_dim = config.descriptor_dim
        self.final_projection = nn.Linear(self.descriptor_dim, self.descriptor_dim, bias=True)
        self.matchability = nn.Linear(self.descriptor_dim, 1, bias=True)

    def forward(self, descriptors0: torch.Tensor, descriptors1: torch.Tensor) -> torch.Tensor:
        m_descriptors0 = self.final_projection(descriptors0)
        m_descriptors1 = self.final_projection(descriptors1)
        _, _, d = m_descriptors0.shape
        m_descriptors0 = m_descriptors0 / self.descriptor_dim**0.25
        m_descriptors1 = m_descriptors1 / self.descriptor_dim**0.25
        sim = torch.einsum("bmd,bnd->bmn", m_descriptors0, m_descriptors1)
        z0 = self.matchability(descriptors0)
        z1 = self.matchability(descriptors1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores

    def get_matchability(self, descriptors: torch.Tensor) -> torch.Tensor:
        return nn.functional.sigmoid(self.matchability(descriptors)).squeeze(-1)


class LightGlueTokenConfidenceLayer(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.token = nn.Linear(config.descriptor_dim, 1)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        token = self.token(descriptors.detach())
        token = nn.functional.sigmoid(token).squeeze(-1)
        return token


def filter_matches(scores: torch.Tensor, threshold: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max_0, max_1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    matches_0, matches_1 = max_0.indices, max_1.indices
    indices_0 = torch.arange(matches_0.shape[1], device=matches_0.device)[None]
    indices_1 = torch.arange(matches_1.shape[1], device=matches_1.device)[None]
    mutual0 = indices_0 == matches_1.gather(1, matches_0)
    mutual1 = indices_1 == matches_0.gather(1, matches_1)
    max_0 = max_0.values.exp()
    zero = max_0.new_tensor(0)
    matching_scores_0 = torch.where(mutual0, max_0, zero)
    matching_scores_1 = torch.where(mutual1, matching_scores_0.gather(1, matches_1), zero)
    valid_0 = mutual0 & (matching_scores_0 > threshold)
    valid_1 = mutual1 & valid_0.gather(1, matches_1)
    matches_0 = torch.where(valid_0, matches_0, -1)
    matches_1 = torch.where(valid_1, matches_1, -1)
    return matches_0, matches_1, matching_scores_0, matching_scores_1


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
    """TODO: Add docstring"""

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
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.depth_confidence

    def get_pruning_mask(self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def do_final_point_pruning(
        self, batch_size, device, indices_self, indices_other, matches, matching_scores, num_keypoints
    ):
        _matches = torch.full((batch_size, num_keypoints), -1, device=device, dtype=matches.dtype)
        _matches[:, indices_self] = torch.where(matches == -1, -1, indices_other.gather(1, matches.clamp(min=0)))
        _matching_scores = torch.zeros((batch_size, num_keypoints), device=device, dtype=matching_scores.dtype)
        _matching_scores[:, indices_self] = matching_scores
        return _matches, _matching_scores

    def do_layer_point_pruning(self, descriptors, keypoints, i, indices, prune, token):
        scores = self.match_assignment_layers[i].get_matchability(descriptors)
        prune_mask = self.get_pruning_mask(token, scores, i)
        keep = torch.where(prune_mask)[1]
        indices = indices.index_select(1, keep)
        descriptors = descriptors.index_select(-2, keep)
        keypoints = keypoints.index_select(-2, keep)
        prune[:, indices] += 1
        return descriptors, keypoints, indices

    def match_image_pair(
        self,
        keypoints_0: torch.Tensor,
        descriptors_0: torch.Tensor,
        keypoints_1: torch.Tensor,
        descriptors_1: torch.Tensor,
        height: int,
        width: int,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple, Tuple]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if keypoints_0.shape[1] == 0 or keypoints_1.shape[1] == 0:  # no keypoints
            shape0, shape1 = keypoints_0.shape[:-1], keypoints_1.shape[:-1]
            return (
                keypoints_0.new_full(shape0, -1, dtype=torch.int),
                keypoints_1.new_full(shape1, -1, dtype=torch.int),
                keypoints_0.new_zeros(shape0),
                keypoints_1.new_zeros(shape1),
                None,
                None,
                all_hidden_states,
                all_attentions,
            )

        device = keypoints_0.device
        batch_size, num_keypoints_0 = keypoints_0.shape[:2]
        num_keypoints_1 = keypoints_1.shape[1]

        # Keypoint normalization
        keypoints_0 = normalize_keypoints(keypoints_0, height, width)
        keypoints_1 = normalize_keypoints(keypoints_1, height, width)
        descriptors_0, keypoint_encoding_output_0 = self.keypoint_processing(
            descriptors_0, keypoints_0, output_hidden_states=output_hidden_states
        )
        descriptors_1, keypoint_encoding_output_1 = self.keypoint_processing(
            descriptors_1, keypoints_1, output_hidden_states=output_hidden_states
        )

        encoded_keypoints_0 = keypoint_encoding_output_0[0]
        encoded_keypoints_1 = keypoint_encoding_output_1[0]

        do_early_stop = self.depth_confidence > 0
        do_point_pruning = self.width_confidence > 0

        if do_point_pruning:
            indices_0 = torch.arange(0, num_keypoints_0, device=device)[None]
            indices_1 = torch.arange(0, num_keypoints_1, device=device)[None]
            prune_0 = torch.ones_like(indices_0)
            prune_1 = torch.ones_like(indices_1)

        for i in range(self.num_layers):
            transformer_output = self.transformer_layers[i](
                descriptors_0,
                descriptors_1,
                encoded_keypoints_0,
                encoded_keypoints_1,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            descriptors_0, descriptors_1 = transformer_output[:2]
            hidden_states = transformer_output[2]
            attention = transformer_output[3]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + hidden_states
            if output_attentions:
                all_attentions = all_attentions + attention

            if do_early_stop and i < self.num_layers - 1:
                assert batch_size == 1
                token_0 = self.token_confidence[i](descriptors_0)
                token_1 = self.token_confidence[i](descriptors_1)

                early_stop = self.check_if_stop(
                    token_0[..., :num_keypoints_0, :],
                    token_1[..., :num_keypoints_1, :],
                    layer_index=i,
                    num_points=num_keypoints_0 + num_keypoints_1,
                )
                if early_stop:
                    break
            if do_point_pruning:
                assert batch_size == 1
                descriptors_0, encoded_keypoints_0, indices_0 = self.do_layer_point_pruning(
                    descriptors_0, encoded_keypoints_0, i, indices_0, prune_0, token_0
                )
                descriptors_1, encoded_keypoints_1, indices_1 = self.do_layer_point_pruning(
                    descriptors_1, encoded_keypoints_1, i, indices_1, prune_1, token_1
                )

        descriptors_0 = descriptors_0[..., :num_keypoints_0, :]
        descriptors_1 = descriptors_1[..., :num_keypoints_1, :]

        scores = self.match_assignment_layers[i](descriptors_0, descriptors_1)
        matches_0, matches_1, matching_scores_0, matching_scores_1 = filter_matches(scores, self.filter_threshold)

        if do_point_pruning:
            matches_0, matching_scores_0 = self.do_final_point_pruning(
                batch_size, device, indices_0, indices_1, matches_0, matching_scores_0, num_keypoints_0
            )

            matches_1, matching_scores_1 = self.do_final_point_pruning(
                batch_size, device, indices_1, indices_0, matches_1, matching_scores_1, num_keypoints_1
            )
        else:
            prune_0 = torch.ones_like(matching_scores_0) * self.num_layers
            prune_1 = torch.ones_like(matching_scores_1) * self.num_layers

        return (
            matches_0,
            matches_1,
            matching_scores_0,
            matching_scores_1,
            prune_0,
            prune_1,
            all_hidden_states,
            all_attentions,
        )

    def batch_output(
        self,
        batch_size,
        list_attentions,
        list_hidden_states,
        list_keypoints_0,
        list_keypoints_1,
        list_matches_0,
        list_matches_1,
        list_matching_scores_0,
        list_matching_scores_1,
        list_prune_0,
        list_prune_1,
        pixel_values,
        output_attentions,
        output_hidden_states,
    ):
        maximum_matches = max(
            [
                max([matches_0.shape[1] for matches_0 in list_matches_0]),
                max([matches_1.shape[1] for matches_1 in list_matches_1]),
            ]
        )
        matches = torch.full(
            (batch_size, 2, maximum_matches),
            -1,
            device=pixel_values.device,
            dtype=torch.int,
        )
        matching_scores = torch.zeros((batch_size, 2, maximum_matches), device=pixel_values.device)
        prune = torch.zeros((batch_size, 2, maximum_matches), device=pixel_values.device)
        matches_mask = torch.zeros(
            (batch_size, 2, maximum_matches),
            device=pixel_values.device,
            dtype=torch.int,
        )
        keypoints = torch.zeros(
            (batch_size, 2, maximum_matches, 2),
            device=pixel_values.device,
        )
        for i, (
            _matches_0,
            _matches_1,
            _matching_scores_0,
            _matching_scores_1,
            _prune_0,
            _prune_1,
            _keypoints_0,
            _keypoints_1,
        ) in enumerate(
            zip(
                list_matches_0,
                list_matches_1,
                list_matching_scores_0,
                list_matching_scores_1,
                list_prune_0,
                list_prune_1,
                list_keypoints_0,
                list_keypoints_1,
            )
        ):
            matches[i, 0, : _matches_0.shape[1]] = _matches_0
            matches[i, 1, : _matches_1.shape[1]] = _matches_1
            matching_scores[i, 0, : _matching_scores_0.shape[1]] = _matching_scores_0
            matching_scores[i, 1, : _matching_scores_1.shape[1]] = _matching_scores_1
            prune[i, 0, : _prune_0.shape[1]] = _prune_0
            prune[i, 1, : _prune_1.shape[1]] = _prune_1
            matches_mask[i, 0, : _matches_0.shape[1]] = 1
            matches_mask[i, 1, : _matches_1.shape[1]] = 1
            keypoints[i, 0, : _keypoints_0.shape[1], :] = _keypoints_0
            keypoints[i, 1, : _keypoints_1.shape[1], :] = _keypoints_1

        if output_hidden_states:
            hidden_states = batch_inconsistent_tensor_list(list_hidden_states)
        else:
            hidden_states = None
        if output_attentions:
            attentions = batch_inconsistent_tensor_list(list_attentions)
        else:
            attentions = None
        return attentions, hidden_states, keypoints, matches, matches_mask, matching_scores, prune

    @add_start_docstrings_to_model_forward(LIGHTGLUE_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, KeypointMatchingOutput]:
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

        list_matches_0 = []
        list_matches_1 = []
        list_matching_scores_0 = []
        list_matching_scores_1 = []
        list_keypoints_0 = []
        list_keypoints_1 = []
        list_prune_0 = []
        list_prune_1 = []
        list_hidden_states = []
        list_attentions = []

        for i in range(pixel_values.size(0)):
            image_pair = pixel_values[i]
            keypoint_detection_output = self.keypoint_detector(
                image_pair,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            keypoints, scores, descriptors, mask = keypoint_detection_output[:4]

            image0_indices = torch.nonzero(mask[0]).squeeze()
            image0_keypoints = torch.unsqueeze(keypoints[0][image0_indices], dim=0)
            image0_descriptors = torch.unsqueeze(descriptors[0][image0_indices], dim=0)
            image1_indices = torch.nonzero(mask[1]).squeeze()
            image1_keypoints = torch.unsqueeze(keypoints[1][image1_indices], dim=0)
            image1_descriptors = torch.unsqueeze(descriptors[1][image1_indices], dim=0)

            match_image_output = self.match_image_pair(
                image0_keypoints,
                image0_descriptors,
                image1_keypoints,
                image1_descriptors,
                height,
                width,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            (
                matches_0,
                matches_1,
                matching_scores_0,
                matching_scores_1,
                prune_0,
                prune_1,
                hidden_states,
                attentions,
            ) = match_image_output
            list_matches_0.append(matches_0)
            list_matches_1.append(matches_1)
            list_matching_scores_0.append(matching_scores_0)
            list_matching_scores_1.append(matching_scores_1)
            list_keypoints_0.append(image0_keypoints)
            list_keypoints_1.append(image1_keypoints)
            list_prune_0.append(prune_0)
            list_prune_1.append(prune_1)
            list_hidden_states.append(hidden_states)
            list_attentions.append(attentions)

        attentions, hidden_states, keypoints, matches, matches_mask, matching_scores, prune = self.batch_output(
            batch_size,
            list_attentions,
            list_hidden_states,
            list_keypoints_0,
            list_keypoints_1,
            list_matches_0,
            list_matches_1,
            list_matching_scores_0,
            list_matching_scores_1,
            list_prune_0,
            list_prune_1,
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            return tuple(
                v
                for v in [matches_mask, matches, matching_scores, keypoints, hidden_states, attentions]
                if v is not None
            )

        return KeypointMatchingOutput(
            loss=loss,
            mask=matches_mask,
            matches=matches,
            matching_scores=matching_scores,
            keypoints=keypoints,
            hidden_states=hidden_states,
            attentions=attentions,
        )

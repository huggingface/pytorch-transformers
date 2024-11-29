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

import torch
from torch import nn

from transformers import PreTrainedModel, add_start_docstrings

from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging
from ..auto import AutoModelForKeypointDetection
from .configuration_lightglue import LightGlueConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC_ = "LightGlueConfig"

_CHECKPOINT_FOR_DOC_ = "stevenbucaille/superglue_indoor"


def concat_attentions_tuples_pair(
    attention_probs_0: Tuple[torch.Tensor], attention_probs_1: Tuple[torch.Tensor]
) -> Tuple[torch.Tensor]:
    """
    Concatenate two tuple of attention probabilities into one.
    We assume that the attention probabilities are of shape (batch_size, num_heads, num_keypoints_a, num_keypoints_b).
    The tuples are assumed to have the same length and of the following form:
    attention_probs_0 = (attention_probs_0_0, attention_probs_0_1, ...)
    attention_probs_1 = (attention_probs_1_0, attention_probs_1_1, ...)
    The output will then be of the form:
    new_attention_probs = (torch.cat(attention_probs_0_0, attention_probs_1_0), torch.cat(attention_probs_0_1, attention_probs_1_1), ...)
    If the attention probabilities have different shapes, the smaller one will be padded with zeros :
    (batch_size * 2, num_heads, max(num_keypoints_a_0, num_keypoints_a_1), max(num_keypoints_b_0, num_keypoints_b_1))
    """
    new_attention_probs = ()
    for attention_prob_0, attention_prob_1 in zip(attention_probs_0, attention_probs_1):
        if attention_prob_0.size() != attention_prob_1.size():
            max_dim2 = max(attention_prob_0.shape[2], attention_prob_1.shape[2])
            max_dim3 = max(attention_prob_0.shape[2], attention_prob_1.shape[2])
            new_attention_prob = torch.zeros(
                2, attention_prob_0.shape[1], max_dim2, max_dim3, device=attention_prob_0.device
            )
            new_attention_prob[0, :, : attention_prob_0.shape[2], : attention_prob_0.shape[3]] = attention_prob_0
            new_attention_prob[1, :, : attention_prob_1.shape[2], : attention_prob_1.shape[3]] = attention_prob_1
            new_attention_probs = new_attention_probs + (new_attention_prob,)
        else:
            new_attention_probs = new_attention_probs + (torch.cat([attention_prob_0, attention_prob_1]),)
    return new_attention_probs


def stack_attention_probs_list(attention_probs: List[torch.Tensor]) -> torch.Tensor:
    current_shape = attention_probs[0].shape
    all_same_shape = all(attention_prob.shape == current_shape for attention_prob in attention_probs)
    if all_same_shape:
        return torch.stack(attention_probs, dim=0)

    max_dim2 = max(attention_prob.shape[2] for attention_prob in attention_probs)
    max_dim3 = max(attention_prob.shape[3] for attention_prob in attention_probs)
    stacked_attention_probs = torch.zeros(
        len(attention_probs), 2, attention_probs[0].shape[1], max_dim2, max_dim3, device=attention_probs[0].device
    )
    for i, attention_prob in enumerate(attention_probs):
        stacked_attention_probs[i, :, :, : attention_prob.shape[2], : attention_prob.shape[3]] = attention_prob
    return stacked_attention_probs


def batch_attention_probs_list(
    attention_probs: Union[List[torch.Tensor], List[Tuple[torch.Tensor]]],
) -> Union[List[torch.Tensor], List[Tuple[torch.Tensor]]]:
    """
    Given a list of attention probabilities, batch them together.
    We assume that the attention probabilities are of shape (batch_size, num_heads, num_keypoints_a, num_keypoints_b).
    The list must be in the following form :
    - List of attention probabilities: [attention_probs_0, attention_probs_1, ...]
    We stack the attention probabilities along the batch dimension for each tuple:
    -> [torch.stack([attention_probs_0_0, attention_probs_1_0, ...], dim=0), torch.stack([attention_probs_0_1, attention_probs_1_1, ...], dim=0), ...]
    """

    list_of_tuples = [tuple([element[i] for element in attention_probs]) for i in range(len(attention_probs[0]))]
    return [stack_attention_probs_list(element) for element in list_of_tuples]


def concat_hidden_states_tuples_pair(
    hidden_states_0: Tuple[torch.Tensor], hidden_states_1: Tuple[torch.Tensor]
) -> Tuple[torch.Tensor]:
    """
    Concatenate two tuple of hidden states into one.
    We assume that the hidden states are of shape (batch_size, hidden_state_size, num_keypoints).
    The tuples are assumed to have the same length and of the following form:
    hidden_states_0 = (hidden_state_0_0, hidden_state_0_1, ...)
    hidden_states_1 = (hidden_state_1_0, hidden_state_1_1, ...)
    The output will then be of the form:
    new_hidden_states = (torch.cat(hidden_state_0_0, hidden_state_1_0), torch.cat(hidden_state_0_1, hidden_state_1_1), ...)
    If the number of keypoints are different among hidden_states, the smaller one will be padded with zeros :
    (batch_size * 2, hidden_state_size, max(num_keypoints_0, num_keypoints_1))
    """
    hidden_states = ()
    for hidden_state_0, hidden_state_1 in zip(hidden_states_0, hidden_states_1):
        if hidden_state_0.shape != hidden_state_1.shape:
            max_num_keypoints = max(hidden_state_0.shape[2], hidden_state_1.shape[2])
            new_hidden_state = torch.zeros(2, hidden_state_0.shape[1], max_num_keypoints, device=hidden_state_0.device)
            new_hidden_state[0, :, : hidden_state_0.shape[2]] = hidden_state_0
            new_hidden_state[1, :, : hidden_state_1.shape[2]] = hidden_state_1
            hidden_states = hidden_states + (new_hidden_state,)
        else:
            hidden_states = hidden_states + (torch.cat([hidden_state_0, hidden_state_1]),)
    return hidden_states


def stack_hidden_states_list(hidden_states: List[torch.Tensor]) -> torch.Tensor:
    """
    Given a list of hidden states tensors, stack them together using torch.stack.
    We assume that the hidden states are of shape (batch_size, hidden_state_size, num_keypoints).
    If all hidden states have the same shape, we stack them along the batch dimension:
    [hidden_state_0, hidden_state_1, ...] -> torch.stack([hidden_state_0, hidden_state_1, ...], dim=0)
    If the hidden states have different shapes, the smaller ones will be padded with zeros:
    (batch_size * 2, hidden_state_size, max(num_keypoints_0, num_keypoints_1))
    """
    current_shape = hidden_states[0].shape
    all_same_shape = all(hidden_state.shape == current_shape for hidden_state in hidden_states)
    if all_same_shape:
        return torch.stack(hidden_states, dim=0)

    max_num_keypoints = max(hidden_state.shape[2] for hidden_state in hidden_states)
    stacked_hidden_state = torch.zeros(
        len(hidden_states),
        2,
        hidden_states[0].shape[1],
        max_num_keypoints,
        device=hidden_states[0].device,
    )
    for i, hidden_state in enumerate(hidden_states):
        stacked_hidden_state[i, :, :, : hidden_state.shape[2]] = hidden_state
    return stacked_hidden_state


def batch_hidden_states(
    hidden_states: Union[List[torch.Tensor], List[Tuple[torch.Tensor]]],
) -> Union[List[torch.Tensor], List[Tuple[torch.Tensor]]]:
    """
    Given a list of hidden states, batch them together using torch.stack.
    We assume that the hidden states are of shape (batch_size, hidden_state_size, num_keypoints).
    The list must be in the following form :
    [(hidden_state_0_0, hidden_state_1_0, ...), (hidden_state_0_1, hidden_state_1_1, ...), ...]
    We stack the hidden states along the batch dimension for each tuple:
    -> [torch.stack([hidden_state_0_0, hidden_state_1_0, ...], dim=0), torch.stack([hidden_state_0_1, hidden_state_1_1, ...], dim=0), ...]
    """

    list_of_tuples = [tuple([element[i] for element in hidden_states]) for i in range(len(hidden_states[0]))]
    return [stack_hidden_states_list(element) for element in list_of_tuples]


def normalize_keypoints(keypoints: torch.Tensor, height: int, width: int):
    """Normalize keypoints locations based on image image_shape"""
    one = keypoints.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (keypoints - center[:, None, :]) / scaling[:, None, :]


def log_sinkhorn_iterations(
    log_cost_matrix: torch.Tensor,
    log_source_distribution: torch.Tensor,
    log_target_distribution: torch.Tensor,
    num_iterations: int,
) -> torch.Tensor:
    """Perform Sinkhorn Normalization in Log-space for stability"""
    log_u_scaling = torch.zeros_like(log_source_distribution)
    log_v_scaling = torch.zeros_like(log_target_distribution)
    for _ in range(num_iterations):
        log_u_scaling = log_source_distribution - torch.logsumexp(log_cost_matrix + log_v_scaling.unsqueeze(1), dim=2)
        log_v_scaling = log_target_distribution - torch.logsumexp(log_cost_matrix + log_u_scaling.unsqueeze(2), dim=1)
    return log_cost_matrix + log_u_scaling.unsqueeze(2) + log_v_scaling.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, reg_param: torch.Tensor, iterations: int) -> torch.Tensor:
    """Perform Differentiable Optimal Transport in Log-space for stability"""
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
    Base class for outputs of keypoint matching models. Due to the nature of keypoint detection and matching, the number of
    keypoints is not fixed and can vary from image to image, which makes batching non-trivial. In the batch of images,
    the maximum number of matches is set as the dimension of the matches and matching scores. The mask
    tensor is used to indicate which values in the keypoints, matches and matching_scores tensors are keypoint matching
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


class LightGluePositionalEncoding(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        M = 2 + 2 * config.add_scale_ori
        F_dim = M
        head_dim = config.descriptor_dim // config.num_heads
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        self.gamma = 1.0
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        projected_keypoints = self.Wr(keypoints)
        cosines, sines = torch.cos(projected_keypoints), torch.sin(projected_keypoints)
        embeddings = torch.stack([cosines, sines], 0).unsqueeze(-3)
        embeddings = embeddings.repeat_interleave(2, dim=-1)
        return embeddings


class LightGlueSelfAttentionBlock(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.num_heads = config.num_heads
        embeddings_dim = config.descriptor_dim

        head_dim = config.descriptor_dim // self.num_heads

        self.Wqkv = nn.Linear(embeddings_dim, embeddings_dim * 3, bias=True)
        self.output_projection = nn.Linear(embeddings_dim, embeddings_dim, bias=True)

        self.ffn = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 2 * embeddings_dim),
            nn.LayerNorm(2 * embeddings_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embeddings_dim, embeddings_dim),
        )

    def forward(
        self,
        descriptors: torch.Tensor,
        keypoints: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        qkv = self.Wqkv(descriptors)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(keypoints, q)
        k = apply_cached_rotary_emb(keypoints, k)
        context, attention = self.attention(q, k, v)
        context = context.transpose(1, 2).flatten(start_dim=-2)
        message = self.output_projection(context)
        ffn_output = self.ffn(torch.cat([descriptors, message], -1))
        output = descriptors + ffn_output
        return output

    def attention(self, q, k, v):
        # TODO add flash attention
        s = q.shape[-1] ** -0.5
        sim = torch.einsum("...id,...jd->...ij", q, k) * s
        attention = nn.functional.softmax(sim, -1)
        output = torch.einsum("...ij,...jd->...id", attention, v)
        return output, attention


class LightGlueCrossAttentionBlock(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.num_heads = config.num_heads
        embeddings_dim = config.descriptor_dim
        head_dim = config.descriptor_dim // self.num_heads
        self.scale = head_dim**-0.5
        inner_dim = head_dim * self.num_heads
        self.to_qk = nn.Linear(embeddings_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(embeddings_dim, inner_dim, bias=True)
        self.to_out = nn.Linear(inner_dim, embeddings_dim, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 2 * embeddings_dim),
            nn.LayerNorm(2 * embeddings_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embeddings_dim, embeddings_dim),
        )

    def forward(self, descriptors0: torch.Tensor, descriptors1: torch.Tensor):
        qk0 = self.to_qk(descriptors0)
        qk1 = self.to_qk(descriptors1)
        v0 = self.to_v(descriptors0)
        v1 = self.to_v(descriptors1)

        qk0 = qk0.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        qk1 = qk1.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        v0 = v0.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        v1 = v1.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

        m0 = nn.functional.scaled_dot_product_attention(qk0, qk1, v1)
        m1 = nn.functional.scaled_dot_product_attention(qk1, qk0, v0)

        m0 = m0.transpose(1, 2).flatten(start_dim=-2)
        m1 = m1.transpose(1, 2).flatten(start_dim=-2)
        descriptors0 = descriptors0 + self.ffn(torch.cat([descriptors0, m0], -1))
        descriptors1 = descriptors1 + self.ffn(torch.cat([descriptors1, m1], -1))
        return descriptors0, descriptors1


class LightGlueTransformerLayer(nn.Module):
    def __init__(self, config: LightGlueConfig):
        super().__init__()

        self.self_attention_block = LightGlueSelfAttentionBlock(config)
        self.cross_attention_block = LightGlueCrossAttentionBlock(config)

    def forward(
        self,
        keypoints0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors0: torch.Tensor,
        descriptors1: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        descriptors0 = self.self_attention_block(descriptors0, keypoints0, output_attentions=output_attentions)
        descriptors1 = self.self_attention_block(descriptors1, keypoints1, output_attentions=output_attentions)
        output = self.cross_attention_block(descriptors0, descriptors1, output_attentions=output_attentions)
        return output


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

        self.token = nn.Sequential(nn.Linear(config.descriptor_dim, 1), nn.Sigmoid())

    def forward(self, descriptors_0: torch.Tensor, descriptors_1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token0 = self.token(descriptors_0.detach()).squeeze(-1)
        token1 = self.token(descriptors_1.detach()).squeeze(-1)
        return token0, token1


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

        self.input_dim = config.input_dim
        self.descriptor_dim = config.descriptor_dim
        self.num_layers = config.num_layers
        self.filter_threshold = config.filter_threshold

        if self.input_dim != self.descriptor_dim:
            self.input_projection = nn.Linear(self.input_dim, self.descriptor_dim, bias=True)
        else:
            self.input_projection = nn.Identity()

        self.positional_encoding = LightGluePositionalEncoding(config)

        self.transformer_layers = nn.ModuleList([LightGlueTransformerLayer(config) for _ in range(config.num_layers)])
        self.match_assignment_layers = nn.ModuleList(
            [LightGlueMatchAssignmentLayer(config) for _ in range(config.num_match_assignment_layers)]
        )
        self.token_confidence = nn.ModuleList(
            [LightGlueTokenConfidenceLayer(config) for _ in range(config.num_layers - 1)]
        )

        self.post_init()

    def keypoint_processing(self, height, image_descriptors, image_keypoints, width):
        keypoints = normalize_keypoints(image_keypoints, height, width)
        descriptors = image_descriptors.contiguous()
        descriptors = self.input_projection(descriptors)
        encoded_keypoints = self.positional_encoding(keypoints)
        return descriptors, encoded_keypoints

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
        _matches = torch.full((batch_size, num_keypoints), device=device, dtype=matches.dtype)
        _matches[:, indices_self] = torch.where(matches == -1, -1, indices_other.gather(1, matches.clamp(min=0)))
        _matching_scores = torch.zeros((batch_size, num_keypoints), device=device)
        _matching_scores[:, indices_self] = matching_scores
        return _matches, _matching_scores

    def do_layer_point_pruning(self, descriptors, i, indices, prune, token):
        scores = self.match_assignment_layers[i].get_matchability(descriptors)
        prune_mask = self.get_pruning_mask(token, scores, i)
        keep = torch.where(prune_mask)[1]
        indices = indices.index_select(1, keep)
        descriptors = descriptors.index_select(-2, keep)
        prune[:, indices] += 1
        return descriptors, indices

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
        descriptors_0, encoded_keypoints_0 = self.keypoint_processing(height, descriptors_0, keypoints_0, width)

        descriptors_1, encoded_keypoints_1 = self.keypoint_processing(height, descriptors_1, keypoints_1, width)

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
                output_attentions=output_attentions,
            )
            descriptors_0, descriptors_1 = transformer_output[:2]

            if do_early_stop:
                assert batch_size == 1
                token_0, token_1 = self.token_confidence[i](descriptors_0, descriptors_1)
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
                descriptors_0, indices_0 = self.do_layer_point_pruning(descriptors_0, i, indices_0, prune_0, token_0)
                descriptors_1, indices_1 = self.do_layer_point_pruning(descriptors_1, i, indices_1, prune_1, token_1)

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
        )
        #
        # encoded_keypoints0 = self.keypoint_encoder(
        #     keypoints_0, image0_scores, output_hidden_states=output_hidden_states
        # )
        # encoded_keypoints1 = self.keypoint_encoder(
        #     keypoints1, image1_scores, output_hidden_states=output_hidden_states
        # )
        #
        # last_hidden_state_0 = encoded_keypoints0[0]
        # last_hidden_state_1 = encoded_keypoints1[0]
        #
        # # Keypoint MLP encoder.
        # descriptors_0 = descriptors_0 + last_hidden_state_0
        # descriptors_1 = descriptors_1 + last_hidden_state_1
        #
        # # Multi-layer Transformer network.
        # gnn_outputs = self.gnn(
        #     descriptors_0,
        #     descriptors_1,
        #     output_hidden_states=output_hidden_states,
        #     output_attentions=output_attentions,
        # )
        # descriptors_0, descriptors_1 = gnn_outputs[:2]
        #
        # # Final MLP projection.
        # projected_descriptors_0 = self.final_projection(descriptors_0)
        # projected_descriptors_1 = self.final_projection(descriptors_1)
        #
        # # Compute matching descriptor distance.
        # scores = torch.einsum("bdn,bdm->bnm", projected_descriptors_0, projected_descriptors_1)
        # scores = scores / self.config.descriptor_dim**0.5
        #
        # # Run the optimal transport.
        # scores = log_optimal_transport(scores, self.bin_score, iterations=self.config.sinkhorn_iterations)
        #
        # # Get the matches with score above "match_threshold".
        # max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        # indices0, indices1 = max0.indices, max1.indices
        # mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        # mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        # zero = scores.new_tensor(0)
        # matching_scores_0 = torch.where(mutual0, max0.values.exp(), zero)
        # matching_scores_1 = torch.where(mutual1, matching_scores_0.gather(1, indices1), zero)
        # valid0 = mutual0 & (matching_scores_0 > self.config.matching_threshold)
        # valid1 = mutual1 & valid0.gather(1, indices1)
        # matches_0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        # matches_1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        #
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + concat_hidden_states_tuples_pair(
        #         encoded_keypoints0[1], encoded_keypoints1[1]
        #     )
        #     all_hidden_states = all_hidden_states + gnn_outputs[2]
        #     all_hidden_states = all_hidden_states + concat_hidden_states_tuples_pair(
        #         (projected_descriptors_0,), (projected_descriptors_1,)
        #     )
        #
        # if output_attentions:
        #     all_attentions = all_attentions + gnn_outputs[3]
        #
        # return (
        #     matches_0,
        #     matches_1,
        #     matching_scores_0,
        #     matching_scores_1,
        #     all_hidden_states,
        #     all_attentions,
        # )

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
            image0_scores = torch.unsqueeze(scores[0][image0_indices], dim=0)
            image1_indices = torch.nonzero(mask[1]).squeeze()
            image1_keypoints = torch.unsqueeze(keypoints[1][image1_indices], dim=0)
            image1_descriptors = torch.unsqueeze(descriptors[1][image1_indices], dim=0)
            image1_scores = torch.unsqueeze(scores[1][image1_indices], dim=0)

            match_image_output = self.match_image_pair(
                image0_keypoints,
                image0_descriptors,
                image0_scores,
                image1_keypoints,
                image1_descriptors,
                image1_scores,
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
                hidden_states,
                attentions,
            ) = match_image_output
            list_matches_0.append(matches_0)
            list_matches_1.append(matches_1)
            list_matching_scores_0.append(matching_scores_0)
            list_matching_scores_1.append(matching_scores_1)
            list_keypoints_0.append(image0_keypoints)
            list_keypoints_1.append(image1_keypoints)
            list_hidden_states.append(hidden_states)
            list_attentions.append(attentions)

        attentions, hidden_states, keypoints, matches, matches_mask, matching_scores = self.batch_output(
            batch_size,
            list_attentions,
            list_hidden_states,
            list_keypoints_0,
            list_keypoints_1,
            list_matches_0,
            list_matches_1,
            list_matching_scores_0,
            list_matching_scores_1,
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
            _keypoints_0,
            _keypoints_1,
        ) in enumerate(
            zip(
                list_matches_0,
                list_matches_1,
                list_matching_scores_0,
                list_matching_scores_1,
                list_keypoints_0,
                list_keypoints_1,
            )
        ):
            matches[i, 0, : _matches_0.shape[1]] = _matches_0
            matches[i, 1, : _matches_1.shape[1]] = _matches_1
            matching_scores[i, 0, : _matching_scores_0.shape[1]] = _matching_scores_0
            matching_scores[i, 1, : _matching_scores_1.shape[1]] = _matching_scores_1
            matches_mask[i, 0, : _matches_0.shape[1]] = 1
            matches_mask[i, 1, : _matches_1.shape[1]] = 1
            keypoints[i, 0, : _keypoints_0.shape[1], :] = _keypoints_0
            keypoints[i, 1, : _keypoints_1.shape[1], :] = _keypoints_1

        if output_hidden_states:
            hidden_states = batch_hidden_states(list_hidden_states)
        else:
            hidden_states = None
        if output_attentions:
            attentions = batch_attention_probs_list(list_attentions)
        else:
            attentions = None
        return attentions, hidden_states, keypoints, matches, matches_mask, matching_scores

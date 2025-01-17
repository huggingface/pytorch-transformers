# coding=utf-8
# Copyright 2024 Om Research Lab and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch OmDet-Turbo model."""

import math
import os
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...activations import ACT2CLS, ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_cuda_available,
    replace_return_docstrings,
)
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_utils import PreTrainedModel
from ...utils import is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_omdet_turbo import OmDetTurboConfig


MultiScaleDeformableAttention = None

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "OmDetTurboConfig"


@dataclass
class OmDetTurboEncoderOutput(ModelOutput):
    """
    Base class for outputs of the OmDetTurboHybridEncoder.

    Args:
        last_hidden_state (`torch.FloatTensor`):
            Last hidden states of the encoder.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        extracted_states (`Tuple[torch.FloatTensor]`):
            The extracted states from the Feature Pyramid Network (FPN) and Path Aggregation Network (PAN) of the encoder.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    extracted_states: Tuple[torch.FloatTensor] = None


@dataclass
class OmDetTurboDecoderOutput(ModelOutput):
    """
    Base class for outputs of the OmDetTurboDecoder.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder.
        decoder_coords (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            The predicted coordinates of the objects.
        decoder_classes (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes)`):
            The predicted classes of the objects.
        encoder_coord_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            The predicted coordinates of the objects from the encoder.
        encoder_class_logits (`Tuple[torch.FloatTensor]`) of shape `(batch_size, num_queries, num_classes)`:
            The predicted class of the objects from the encoder.
        init_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            The initial reference points.
        intermediate_reference_points (`Tuple[Tuple[torch.FloatTensor]]`):
            The intermediate reference points.
        hidden_states (`Optional[Tuple[torch.FloatTensor]]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`Optional[Tuple[Tuple[torch.FloatTensor]]]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_coords: torch.FloatTensor = None
    decoder_classes: torch.FloatTensor = None
    encoder_coord_logits: torch.FloatTensor = None
    encoder_class_logits: Tuple[torch.FloatTensor] = None
    init_reference_points: torch.FloatTensor = None
    intermediate_reference_points: Tuple[Tuple[torch.FloatTensor]] = None


@dataclass
class OmDetTurboObjectDetectionOutput(ModelOutput):
    """
    Output type of [`OmDetTurboObjectDetectionOutput`].

    Args:
        loss (`torch.FloatTensor`):
            The loss value.
        decoder_coord_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            The predicted coordinates logits of the objects.
        decoder_class_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes)`):
            The predicted class of the objects.
        init_reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            The initial reference points.
        intermediate_reference_points (`Tuple[Tuple[torch.FloatTensor]]`):
            The intermediate reference points.
        encoder_coord_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            The predicted coordinates of the objects from the encoder.
        encoder_class_logits (`Tuple[torch.FloatTensor]`):
            The predicted class of the objects from the encoder.
        encoder_extracted_states (`torch.FloatTensor`):
            The extracted states from the Feature Pyramid Network (FPN) and Path Aggregation Network (PAN) of the encoder.
        decoder_hidden_states (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`Tuple[Tuple[torch.FloatTensor]]`, *optional*):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
        encoder_hidden_states (`Tuple[torch.FloatTensor]`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        encoder_attentions (`Tuple[Tuple[torch.FloatTensor]]`, *optional*):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
        classes_structure (`torch.LongTensor`, *optional*):
            The number of queried classes for each image.
    """

    loss: torch.FloatTensor = None
    decoder_coord_logits: torch.FloatTensor = None
    decoder_class_logits: torch.FloatTensor = None
    init_reference_points: torch.FloatTensor = None
    intermediate_reference_points: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_coord_logits: torch.FloatTensor = None
    encoder_class_logits: Tuple[torch.FloatTensor] = None
    encoder_extracted_states: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    classes_structure: Optional[torch.LongTensor] = None


# Copied from models.deformable_detr.load_cuda_kernels
def load_cuda_kernels():
    from torch.utils.cpp_extension import load

    global MultiScaleDeformableAttention

    root = Path(__file__).resolve().parent.parent.parent / "kernels" / "deformable_detr"
    src_files = [
        root / filename
        for filename in [
            "vision.cpp",
            os.path.join("cpu", "ms_deform_attn_cpu.cpp"),
            os.path.join("cuda", "ms_deform_attn_cuda.cu"),
        ]
    ]

    MultiScaleDeformableAttention = load(
        "MultiScaleDeformableAttention",
        src_files,
        with_cuda=True,
        extra_include_paths=[str(root)],
        extra_cflags=["-DWITH_CUDA=1"],
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    )


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention
def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: Union[Tensor, List[Tuple]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # Ignore copy
    value_list = value.split([height * width for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class OmDetTurboLRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.current_load = 0

    def has(self, key) -> bool:
        return key in self.cache

    def get(self, key):
        """
        Get the value of the key if the key exists in the cache, otherwise return None.
        Move the key to the end of the cache to show that it was recently used.
        """
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value) -> None:
        """
        Add the key-value pair to the cache.
        Move the key to the end of the cache to show that it was recently used.
        If the cache is full, remove the first key (least recently used).
        """
        if key not in self.cache:
            self.current_load += 1
            if self.current_load > self.capacity:
                self.cache.popitem(last=False)
                self.current_load -= 1

        self.cache[key] = value
        self.cache.move_to_end(key)


class OmDetTurboLanguageBackbone(nn.Module):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__()
        self.model = AutoModel.from_config(config.text_config)
        self.text_projection = nn.Parameter(torch.zeros(config.text_projection_in_dim, config.text_projection_out_dim))

    def forward(self, hidden_states, mask=None, encode_type="task"):
        text_outputs = self.model(hidden_states)
        pooled_output = text_outputs[0]
        if encode_type == "task":
            if mask is None:
                raise ValueError("mask is required for task encoding")
            max_len = (mask != 0).sum(1).max().item()
            truncated_mask = mask[:, :max_len]
            truncated_output = pooled_output[:, :max_len, :]
            return truncated_output.transpose(0, 1), truncated_mask
        elif encode_type == "class":
            max_pooled_output = pooled_output[torch.arange(pooled_output.shape[0]), hidden_states.argmax(dim=-1)]
            projected_output = max_pooled_output @ self.text_projection
            return projected_output
        else:
            raise ValueError(f"encode_type {encode_type} is not supported")


class OmDetTurboVisionBackbone(nn.Module):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__()
        self.apply_layernorm_after_vision_backbone = config.apply_layernorm_after_vision_backbone
        self.vision_backbone = load_backbone(config)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(in_channel_dim, eps=config.layer_norm_eps) for in_channel_dim in config.encoder_in_channels]
        )

    def forward(self, pixel_values):
        outputs = self.vision_backbone(pixel_values).feature_maps
        if self.apply_layernorm_after_vision_backbone:
            outputs = [
                layer_norm(output).permute(0, 3, 1, 2).contiguous()
                for layer_norm, output in zip(self.layer_norms, outputs)
            ]

        return outputs


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.MultiScaleDeformableAttentionFunction
class MultiScaleDeformableAttentionFunction(Function):
    @staticmethod
    def forward(
        context,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        context.im2col_step = im2col_step
        output = MultiScaleDeformableAttention.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            context.im2col_step,
        )
        context.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(context, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = context.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MultiScaleDeformableAttention.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            context.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention with DeformableDetr->OmDetTurbo, Deformable DETR->OmDet-Turbo
class OmDetTurboMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, config: OmDetTurboConfig, num_heads: int, n_points: int):
        super().__init__()

        kernel_loaded = MultiScaleDeformableAttention is not None
        if is_torch_cuda_available() and is_ninja_available() and not kernel_loaded:
            try:
                load_cuda_kernels()
            except Exception as e:
                logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in OmDetTurboMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.disable_custom_kernels = config.disable_custom_kernels

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        # Ignore copy
        total_elements = sum([shape[0] * shape[1] for shape in spatial_shapes_list])
        if total_elements != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(~attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        num_coordinates = reference_points.shape[-1]
        if num_coordinates == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif num_coordinates == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        if self.disable_custom_kernels:
            # PyTorch implementation
            output = multi_scale_deformable_attention(
                value, spatial_shapes_list, sampling_locations, attention_weights
            )
        else:
            try:
                # custom kernel
                output = MultiScaleDeformableAttentionFunction.apply(
                    value,
                    spatial_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            except Exception:
                # PyTorch implementation
                output = multi_scale_deformable_attention(
                    value, spatial_shapes_list, sampling_locations, attention_weights
                )
        output = self.output_proj(output)

        return output, attention_weights


# Copied from transformers.models.rt_detr.modeling_rt_detr.RTDetrConvNormLayer with RTDetr->OmDetTurbo
class OmDetTurboConvNormLayer(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, padding=None, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels, config.batch_norm_eps)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


# Copied from transformers.models.rt_detr.modeling_rt_detr.RTDetrRepVggBlock with RTDetr->OmDetTurbo, activation_function->csp_activation
class OmDetTurboRepVggBlock(nn.Module):
    """
    RepVGG architecture block introduced by the work "RepVGG: Making VGG-style ConvNets Great Again".
    """

    def __init__(self, config: OmDetTurboConfig):
        super().__init__()

        activation = config.csp_activation
        hidden_channels = int(config.encoder_hidden_dim * config.hidden_expansion)
        self.conv1 = OmDetTurboConvNormLayer(config, hidden_channels, hidden_channels, 3, 1, padding=1)
        self.conv2 = OmDetTurboConvNormLayer(config, hidden_channels, hidden_channels, 1, 1, padding=0)
        self.activation = nn.Identity() if activation is None else ACT2CLS[activation]()

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.activation(y)


# Copied from transformers.models.rt_detr.modeling_rt_detr.RTDetrCSPRepLayer with RTDetr->OmDetTurbo, activation_function->csp_activation
class OmDetTurboCSPRepLayer(nn.Module):
    """
    Cross Stage Partial (CSP) network layer with RepVGG blocks.
    """

    def __init__(self, config: OmDetTurboConfig):
        super().__init__()

        in_channels = config.encoder_hidden_dim * 2
        out_channels = config.encoder_hidden_dim
        num_blocks = 3
        activation = config.csp_activation

        hidden_channels = int(out_channels * config.hidden_expansion)
        self.conv1 = OmDetTurboConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.conv2 = OmDetTurboConvNormLayer(config, in_channels, hidden_channels, 1, 1, activation=activation)
        self.bottlenecks = nn.Sequential(*[OmDetTurboRepVggBlock(config) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = OmDetTurboConvNormLayer(config, hidden_channels, out_channels, 1, 1, activation=activation)
        else:
            self.conv3 = nn.Identity()

    def forward(self, hidden_state):
        device = hidden_state.device
        hidden_state_1 = self.conv1(hidden_state)
        hidden_state_1 = self.bottlenecks(hidden_state_1).to(device)
        hidden_state_2 = self.conv2(hidden_state).to(device)
        return self.conv3(hidden_state_1 + hidden_state_2)


class OmDetTurboMultiheadAttention(nn.Module):
    """Equivalent implementation of nn.MultiheadAttention with `batch_first=True`."""

    def __init__(self, config, hidden_size, num_attention_heads, dropout):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        query_layer = self.transpose_for_scores(self.query(queries))
        key_layer = self.transpose_for_scores(self.key(keys))
        value_layer = self.transpose_for_scores(self.value(values))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.out_proj(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class OmDetTurboEncoderLayer(nn.Module):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__()
        self.self_attn = OmDetTurboMultiheadAttention(
            config,
            hidden_size=config.encoder_hidden_dim,
            num_attention_heads=config.num_attention_heads,
            dropout=config.encoder_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.encoder_dropout)
        self.activation_fn = ACT2FN[config.encoder_feedforward_activation]
        self.encoder_feedforward_dropout = nn.Dropout(config.encoder_feedforward_dropout)
        self.fc1 = nn.Linear(config.encoder_hidden_dim, config.encoder_dim_feedforward)
        self.fc2 = nn.Linear(config.encoder_dim_feedforward, config.encoder_hidden_dim)
        self.final_layer_norm = nn.LayerNorm(config.encoder_hidden_dim, eps=config.layer_norm_eps)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Object queries (also called content embeddings), to be added to the hidden states.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        query = key = self.with_pos_embed(hidden_states, position_embeddings)

        hidden_states = self.self_attn(
            queries=query,
            keys=key,
            values=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states, attentions = hidden_states if output_attentions else (hidden_states[0], None)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.encoder_feedforward_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        if output_attentions:
            return hidden_states, attentions

        return (hidden_states,)


class OmDetTurboEncoder(nn.Module):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__()

        self.layers = nn.ModuleList([OmDetTurboEncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(
        self, src, src_mask=None, pos_embed=None, output_attentions: bool = False
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        hidden_states = src
        attention = () if output_attentions else None
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=src_mask,
                position_embeddings=pos_embed,
                output_attentions=output_attentions,
            )
            if output_attentions:
                attention = attention + (hidden_states[1],)
            hidden_states = hidden_states[0]

        return hidden_states, attention


class OmDetTurboHybridEncoder(nn.Module):
    """
    Encoder consisting of channel projection layers, a set of `OmDetTurboEncoder`, a top-down Feature Pyramid Network
    (FPN) and a bottom-up Path Aggregation Network (PAN). More details on the paper: https://arxiv.org/abs/2304.08069

    Args:
        config: OmDetTurboConfig
    """

    def __init__(self, config: OmDetTurboConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encoder_projection_indices = config.encoder_projection_indices
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.encoder_hidden_dim for _ in self.in_channels]

        self.channel_projection_layers = nn.ModuleList()
        for in_channel in self.in_channels:
            self.channel_projection_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, self.encoder_hidden_dim, kernel_size=(1, 1), bias=False),
                    nn.BatchNorm2d(self.encoder_hidden_dim),
                )
            )

        # encoder transformer
        self.encoder = nn.ModuleList([OmDetTurboEncoder(config) for _ in range(len(self.encoder_projection_indices))])
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                OmDetTurboConvNormLayer(
                    config,
                    in_channels=self.encoder_hidden_dim,
                    out_channels=self.encoder_hidden_dim,
                    kernel_size=1,
                    stride=1,
                    activation=config.conv_norm_activation,
                )
            )
            self.fpn_blocks.append(OmDetTurboCSPRepLayer(config))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(
                OmDetTurboConvNormLayer(
                    config,
                    in_channels=self.encoder_hidden_dim,
                    out_channels=self.encoder_hidden_dim,
                    kernel_size=3,
                    stride=2,
                    activation=config.conv_norm_activation,
                )
            )
            self.pan_blocks.append(OmDetTurboCSPRepLayer(config))

    @staticmethod
    def build_2d_sincos_position_embedding(
        width, height, embed_dim=256, temperature=10000.0, device="cpu", dtype=torch.float32
    ):
        grid_w = torch.arange(int(width), dtype=dtype, device=device)
        grid_h = torch.arange(int(height), dtype=dtype, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D sin-cos position embedding")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=dtype, device=device) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(
        self,
        inputs_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layers) that is passed to the encoder.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeddings

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # get projection features
        projected_features = [self.channel_projection_layers[i](feature) for i, feature in enumerate(hidden_states)]
        # encoder
        for encoder_layer_index, feature_to_project_index in enumerate(self.encoder_projection_indices):
            if output_hidden_states:
                encoder_states = encoder_states + (projected_features[feature_to_project_index],)
            height, width = projected_features[feature_to_project_index].shape[2:]
            # flatten [batch, channel, height, width] to [batch, height*width, channel]
            src_flatten = projected_features[feature_to_project_index].flatten(2).permute(0, 2, 1)
            if self.training or self.eval_size is None:
                pos_embed = self.build_2d_sincos_position_embedding(
                    width,
                    height,
                    self.encoder_hidden_dim,
                    self.positional_encoding_temperature,
                    device=src_flatten.device,
                    dtype=src_flatten.dtype,
                ).to(src_flatten.device, src_flatten.dtype)
            else:
                pos_embed = None
            layer_outputs = self.encoder[encoder_layer_index](
                src_flatten,
                pos_embed=pos_embed,
                output_attentions=output_attentions,
            )
            projected_features[feature_to_project_index] = (
                layer_outputs[0].permute(0, 2, 1).reshape(-1, self.encoder_hidden_dim, height, width).contiguous()
            )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (projected_features[feature_to_project_index],)

        # Feature Pyramid Network (FPN)
        fpn_feature_maps = [projected_features[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = fpn_feature_maps[0]
            feat_low = projected_features[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            fpn_feature_maps[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            fps_map = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            fpn_feature_maps.insert(0, fps_map)

        # Path Aggregation Network (PAN)
        fpn_states = [fpn_feature_maps[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = fpn_states[-1]
            feat_high = fpn_feature_maps[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            hidden_states = self.pan_blocks[idx](
                torch.concat([downsample_feat, feat_high.to(downsample_feat.device)], dim=1)
            )
            fpn_states.append(hidden_states)
        if not return_dict:
            return (fpn_states[-1], encoder_states, all_attentions, fpn_states)
        return OmDetTurboEncoderOutput(
            last_hidden_state=fpn_states[-1],
            hidden_states=encoder_states,
            attentions=all_attentions,
            extracted_states=fpn_states,
        )


class OmDetTurboMLPWithDropout(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.class_embed_dim, config.task_encoder_hidden_dim)
        self.activation = ACT2FN[config.decoder_activation]
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.linear2 = nn.Linear(config.task_encoder_hidden_dim, config.class_embed_dim)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class OmDetTurboMLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        hidden_layers_dims = [hidden_dim] * (num_layers - 1)
        layers_dims = [input_dim] + hidden_layers_dims + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(layers_dims[:-1], layers_dims[1:])]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class OmDetTurboResidualLayer(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.class_embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.decoder_dropout)

    def forward(self, x, y):
        return self.norm1(x + self.dropout(y))


class OmDetTurboTaskEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = OmDetTurboMLPWithDropout(config)
        self.res1 = OmDetTurboResidualLayer(config)

    def forward(self, x):
        mlp_out = self.mlp(x)
        x = self.res1(x, mlp_out)
        return x


class OmDetTurboDeformableTransformerDecoderLayer(nn.Module):
    """
    A single layer of the Deformable Transformer Decoder.
    """

    def __init__(self, config):
        super().__init__()
        # self attention
        self.self_attn = OmDetTurboMultiheadAttention(
            config,
            hidden_size=config.decoder_hidden_dim,
            num_attention_heads=config.decoder_num_heads,
            dropout=config.decoder_dropout,
        )
        self.dropout1 = nn.Dropout(config.decoder_dropout)
        self.norm1 = nn.LayerNorm(config.decoder_hidden_dim, eps=config.layer_norm_eps)

        # cross attention
        self.cross_attn = OmDetTurboMultiscaleDeformableAttention(
            config, num_heads=config.decoder_num_heads, n_points=config.decoder_num_points
        )
        self.dropout2 = nn.Dropout(config.decoder_dropout)
        self.norm2 = nn.LayerNorm(config.decoder_hidden_dim, eps=config.layer_norm_eps)

        # feed forward network
        self.linear1 = nn.Linear(config.decoder_hidden_dim, config.decoder_dim_feedforward)
        self.act = ACT2FN[config.decoder_activation]
        self.dropout3 = nn.Dropout(config.decoder_dropout)
        self.linear2 = nn.Linear(config.decoder_dim_feedforward, config.decoder_hidden_dim)
        self.dropout4 = nn.Dropout(config.decoder_dropout)
        self.norm3 = nn.LayerNorm(config.decoder_hidden_dim, eps=config.layer_norm_eps)

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        decoder_embeddings,
        task_features,
        reference_points,
        vision_features,
        vision_shapes,
        vision_shapes_list,
        level_start_index=None,
        attention_mask=None,
        padding_mask=None,
        query_position=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states

        origin_embedding_len = decoder_embeddings.shape[1]

        # self attention
        query = key = self.with_pos_embed(decoder_embeddings, query_position)
        # combine task_features with query, key, value
        task_features = task_features.transpose(0, 1)
        query = torch.cat((query, task_features), dim=1)
        key = torch.cat((key, task_features), dim=1)
        decoder_embeddings = torch.cat((decoder_embeddings, task_features), dim=1)

        outputs = self.self_attn(
            query,
            key,
            decoder_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        context, self_attention = outputs if output_attentions else (outputs[0], None)
        decoder_embeddings = decoder_embeddings + self.dropout1(context)
        decoder_embeddings = self.norm1(decoder_embeddings)

        task_features = decoder_embeddings[:, origin_embedding_len:, :].transpose(0, 1)
        decoder_embeddings = decoder_embeddings[:, :origin_embedding_len, :]

        # cross attention
        hidden_states = self.with_pos_embed(decoder_embeddings, query_position)
        reference_points = reference_points.unsqueeze(2)
        outputs, cross_attention = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=padding_mask,
            encoder_hidden_states=vision_features,
            reference_points=reference_points,
            spatial_shapes=vision_shapes,
            spatial_shapes_list=vision_shapes_list,
            level_start_index=level_start_index,
        )
        decoder_embeddings = decoder_embeddings + self.dropout2(outputs)
        residual = self.norm2(decoder_embeddings)

        # feed forward network
        decoder_embeddings = self.linear2(self.dropout3(self.act(self.linear1(residual))))
        decoder_embeddings = residual + self.dropout4(decoder_embeddings)
        decoder_embeddings = self.norm3(decoder_embeddings)

        return (
            decoder_embeddings,
            task_features,
            self_attention if output_attentions else None,
            cross_attention if output_attentions else None,
        )


class OmDetTurboPreTrainedModel(PreTrainedModel):
    config_class = OmDetTurboConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        def linear_init_(module_to_init):
            bound = 1 / math.sqrt(module_to_init.weight.shape[0])
            nn.init.uniform_(module_to_init.weight, -bound, bound)
            if hasattr(module_to_init, "bias") and module_to_init.bias is not None:
                nn.init.uniform_(module_to_init.bias, -bound, bound)

        if isinstance(module, OmDetTurboEncoderLayer):
            linear_init_(module.fc1)
            linear_init_(module.fc2)
        elif isinstance(module, OmDetTurboDecoder):
            nn.init.constant_(module.encoder_bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(module.encoder_bbox_head.layers[-1].bias, 0.0)
            for mlp in module.decoder_bbox_head:
                nn.init.constant_(mlp.layers[-1].weight, 0.0)
                nn.init.constant_(mlp.layers[-1].bias, 0.0)
            linear_init_(module.encoder_vision_features[0])
            nn.init.xavier_uniform_(module.encoder_vision_features[0].weight)
            if module.learn_initial_query:
                nn.init.xavier_uniform_(module.tgt_embed.weight)
            nn.init.xavier_uniform_(module.query_position_head.layers[0].weight)
            nn.init.xavier_uniform_(module.query_position_head.layers[1].weight)
            for layer in module.channel_projection_layers:
                nn.init.xavier_uniform_(layer[0].weight)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, OmDetTurboDecoder):
            module.gradient_checkpointing = value

    @staticmethod
    def _get_cache_key_at_index(input_ids, attention_mask, index):
        input_ids = input_ids[index]
        input_mask = attention_mask[index]
        cache_key = tuple(input_ids[input_mask != 0].tolist())
        return cache_key

    def get_cached_class_embeddings(self, classes_input_ids, classes_attention_mask):
        not_cached_index = []
        not_cached_classes = []
        total_embeddings = []
        for idx, _ in enumerate(classes_input_ids):
            cache_key = self._get_cache_key_at_index(classes_input_ids, classes_attention_mask, idx)
            if self.language_cache_class.has(cache_key):
                total_embeddings.append(self.language_cache_class.get(cache_key))
            else:
                total_embeddings.append(None)
                not_cached_index.append(idx)
                not_cached_classes.append(cache_key)

        if not_cached_classes:
            not_cached_classes_ids = torch.stack([classes_input_ids[idx] for idx in not_cached_index])
            embeddings = self.language_backbone(not_cached_classes_ids, encode_type="class")
            for idx, emb in enumerate(embeddings):
                idx_to_put = not_cached_index[idx]
                total_embeddings[idx_to_put] = emb
                self.language_cache_class.put(not_cached_classes[idx], emb)

        total_class_embs = torch.stack(total_embeddings).to(self.device)
        return total_class_embs

    def get_cached_task_embeddings(self, tasks_input_ids, tasks_attention_mask):
        not_cached_index = []
        not_cached_tasks = []
        total_task_features = []
        total_task_masks = []
        for idx, _ in enumerate(tasks_input_ids):
            cache_key = self._get_cache_key_at_index(tasks_input_ids, tasks_attention_mask, idx)
            if self.language_cache_prompt.has(cache_key):
                task_feature, task_mask = self.language_cache_prompt.get(cache_key)
                total_task_features.append(task_feature)
                total_task_masks.append(task_mask)
            else:
                total_task_features.append(None)
                total_task_masks.append(None)
                not_cached_index.append(idx)
                not_cached_tasks.append(cache_key)

        if not_cached_tasks:
            not_cached_index_ids = torch.stack([tasks_input_ids[idx] for idx in not_cached_index])
            not_cached_mask = torch.stack([tasks_attention_mask[idx] for idx in not_cached_index])
            embeddings, masks = self.language_backbone(not_cached_index_ids, mask=not_cached_mask, encode_type="task")

            for idx in range(embeddings.shape[1]):
                emb = embeddings[:, [idx], :]
                idx_to_put = not_cached_index[idx]
                cur_mask = torch.unsqueeze(masks[idx], dim=0).to(self.device)
                total_task_features[idx_to_put] = emb
                total_task_masks[idx_to_put] = cur_mask
                self.language_cache_prompt.put(not_cached_tasks[idx], (emb, cur_mask))

        # pad before concat if needed
        max_len = max([task.shape[0] for task in total_task_features])
        for idx, task in enumerate(total_task_features):
            if task.shape[0] < max_len:
                pad_size = max_len - task.shape[0]
                total_task_features[idx] = F.pad(task, (0, 0, 0, 0, 0, pad_size))
                total_task_masks[idx] = F.pad(total_task_masks[idx], (0, pad_size))

        total_task_features = torch.cat(total_task_features, dim=1).to(self.device)
        total_task_masks = torch.cat(total_task_masks, dim=0).to(self.device)

        return total_task_features, total_task_masks

    def get_language_embedding(
        self,
        classes_input_ids,
        classes_attention_mask,
        tasks_input_ids,
        tasks_attention_mask,
        classes_structure,
    ):
        batched_classes_embeddings = self.get_cached_class_embeddings(classes_input_ids, classes_attention_mask)
        # regroup class embeddings using saved structure
        max_class_size = torch.max(classes_structure)
        class_embeddings_regrouped = []
        start = 0
        for size in classes_structure:
            pad_size = max_class_size - size
            class_embeddings_regrouped.append(
                F.pad(batched_classes_embeddings[start : start + size], (0, 0, 0, pad_size)).unsqueeze(1)
            )
            start += size
        class_embeddings = torch.cat(class_embeddings_regrouped, dim=1)

        task_embeddings, task_mask = self.get_cached_task_embeddings(tasks_input_ids, tasks_attention_mask)

        return class_embeddings, task_embeddings, task_mask


OMDET_TURBO_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OmDetTurboConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

OMDET_TURBO_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`DetrImageProcessor.__call__`] for
            details.

        classes_input_ids (`torch.LongTensor` of shape `(total_classes (>= batch_size), sequence_length)`):
            Indices of input classes sequence tokens in the vocabulary of the language model.
            Several classes can be provided for each tasks, thus the tokenized classes are flattened
            and the structure of the classes is provided in the `classes_structure` argument.

            Indices can be obtained using [`OmDetTurboProcessor`]. See [`OmDetTurboProcessor.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)

        classes_attention_mask (`torch.BoolTensor` of shape `(total_classes (>= batch_size), num_classes, sequence_length)`):
            Attention mask for the classes. This is a binary mask that indicates which tokens should be attended to,
            and which should not.

        tasks_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input tasks sequence tokens in the vocabulary of the language model.

            Indices can be obtained using [`OmDetTurboProcessor`]. See [`OmDetTurboProcessor.__call__`] for
            details.

            [What are input IDs?](../glossary#input-ids)

        tasks_attention_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Attention mask for the tasks. This is a binary mask that indicates which tokens should be attended to,
            and which should not.

        classes_structure (torch.LongTensor of shape `(batch_size)`):
            Structure of the classes. This tensor indicates the number of classes for each task.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """


def _cosine_similarity_scaled(a, b, logit_scale):
    a = a / a.norm(dim=2, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * torch.bmm(a, b)
    return logits_per_image


def get_class_similarity(class_distance_type, cls_feature, class_proj):
    logit_scale = torch.tensor(1 / 0.07).log()
    if class_distance_type == "cosine":
        class_logits = _cosine_similarity_scaled(cls_feature, class_proj, logit_scale)
    elif class_distance_type == "dot":
        class_logits = torch.bmm(cls_feature, class_proj)
    else:
        raise Exception("Unknown class_distance_type {}".format(class_distance_type))
    return class_logits


def _inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class OmDetTurboDecoder(OmDetTurboPreTrainedModel):
    def __init__(self, config: OmDetTurboConfig):
        self.config = config
        super().__init__(config)
        self.gradient_checkpointing = False

        hidden_dim = config.decoder_hidden_dim
        self.num_queries = config.num_queries
        self.class_distance_type = config.class_distance_type
        self.learn_initial_query = config.learn_initial_query

        # backbone feature projection
        self.channel_projection_layers = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim))
            for x in config.vision_features_channels
        )
        self.task_encoder = OmDetTurboTaskEncoder(config)
        if config.class_embed_dim != hidden_dim:
            self.task_project = nn.Linear(config.class_embed_dim, hidden_dim)

        # Transformer module
        self.layers = nn.ModuleList(
            [OmDetTurboDeformableTransformerDecoderLayer(config) for _ in range(config.decoder_num_layers)]
        )
        self.decoder_num_layers = config.decoder_num_layers
        # decoder embedding
        if self.learn_initial_query:
            self.tgt_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.query_position_head = OmDetTurboMLP(
            input_dim=4, hidden_dim=2 * hidden_dim, output_dim=hidden_dim, num_layers=2
        )

        # encoder head
        self.encoder_vision_features = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim, eps=config.layer_norm_eps)
        )
        self.encoder_class_head = nn.Linear(config.class_embed_dim, hidden_dim)
        self.encoder_bbox_head = OmDetTurboMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=4, num_layers=3)

        # decoder head
        self.decoder_class_head = nn.ModuleList(
            [nn.Linear(config.class_embed_dim, hidden_dim) for _ in range(config.decoder_num_layers)]
        )
        self.decoder_bbox_head = nn.ModuleList(
            [OmDetTurboMLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(config.decoder_num_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    @lru_cache(maxsize=32)
    def generate_anchors(self, spatial_shapes=None, grid_size=0.05, device="cpu", dtype=torch.float32):
        # We always generate anchors in float32 to preserve equivalence between
        # dynamic and static anchor inference
        # Ignore copy
        if spatial_shapes is None:
            raise ValueError("spatial_shapes must be provided")

        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=height, dtype=dtype, device=device),
                torch.arange(end=width, dtype=dtype, device=device),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_wh = torch.tensor([width, height], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**level)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, height * width, 4))
        # define the valid range for anchor coordinates
        eps = 1e-2
        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_encoder_input(self, vision_features):
        # get projection features
        vision_features = [self.channel_projection_layers[i](feat) for i, feat in enumerate(vision_features)]
        # get encoder inputs
        new_vision_features = []
        new_vision_shapes_list = []
        for feat in vision_features:
            height, width = feat.shape[2:]
            # [batch_size, channels, height, width] -> [batch_size, height*width, channels]
            new_vision_features.append(feat.flatten(2).permute(0, 2, 1))
            # [num_feature_levels, 2]
            new_vision_shapes_list.append((height, width))

        # [batch_size, height*width, channels]
        new_vision_features = torch.cat(new_vision_features, 1)
        new_vision_shapes = torch.tensor(new_vision_shapes_list, dtype=torch.int64).to(vision_features[0].device)
        level_start_index = torch.cat((new_vision_shapes.new_zeros((1,)), new_vision_shapes.prod(1).cumsum(0)[:-1]))

        return new_vision_features, new_vision_shapes, new_vision_shapes_list, level_start_index

    def _get_decoder_input(
        self, vision_features, vision_shapes, class_features, denoise_embeddings=None, denoise_bboxes=None
    ):
        batch_size = len(vision_features)
        # prepare input for decoder
        anchors, valid_mask = self.generate_anchors(
            vision_shapes, device=vision_features.device, dtype=vision_features.dtype
        )
        predicted_class_features = self.encoder_vision_features(
            torch.where(
                valid_mask, vision_features, torch.tensor(0.0, dtype=vision_features.dtype).to(vision_features.device)
            )
        )

        original_class_projected = self.encoder_class_head(class_features).permute(1, 2, 0)
        encoder_class_similarity = get_class_similarity(
            self.class_distance_type, predicted_class_features, original_class_projected
        )

        # dynamic anchors + static content
        # (batch_size, height*width, 4)
        encoder_outputs_bboxes = self.encoder_bbox_head(predicted_class_features) + anchors

        # query selection
        # (batch_size, num_queries)
        topk_ind = torch.topk(encoder_class_similarity.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (batch_size, num_queries)
        batch_ind = (
            torch.arange(end=batch_size, dtype=topk_ind.dtype, device=topk_ind.device)
            .unsqueeze(-1)
            .repeat(1, self.num_queries)
            .view(-1)
        )

        reference_points = encoder_outputs_bboxes[batch_ind, topk_ind].view(batch_size, self.num_queries, -1)
        encoder_bboxes = reference_points.sigmoid()
        if denoise_bboxes is not None:
            reference_points = torch.cat([denoise_bboxes, reference_points], 1)
        if self.training:
            reference_points = reference_points.detach()
        encoder_class_similarity = encoder_class_similarity[batch_ind, topk_ind].view(batch_size, self.num_queries, -1)

        if self.learn_initial_query:
            embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            embeddings = predicted_class_features[batch_ind, topk_ind].view(batch_size, self.num_queries, -1)
            if self.training:
                embeddings = embeddings.detach()
        if denoise_embeddings is not None:
            embeddings = torch.cat([denoise_embeddings, embeddings], 1)

        return embeddings, reference_points, encoder_bboxes, encoder_class_similarity, anchors

    def forward(
        self,
        vision_features,
        class_features,
        task_features,
        task_mask,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Args:
            vision_features (`torch.FloatTensor`): The sequence of vision features. shape depends on the vision
                backbone.
            class_features (`torch.FloatTensor`): The sequence of class features of shape
                `(class_sequence_length, batch_size, class_embed_dim)`.
            task_features (`torch.FloatTensor`): The sequence of task features of shape
                `(task_sequence_length, batch_size, decoder_hidden_dim)`.
            task_mask (`torch.LongTensor`): The mask for the task features of shape `(batch_size, task_sequence_length)`.
            output_attentions (`bool`, *optional*): Whether or not to return the attentions tensors of all attention
                layers. See `attentions` under returned tensors for more detail.
            output_hidden_states (`bool`, *optional*): Whether or not to return the hidden states of all layers. See
                `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*): Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain
                tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_features, vision_shapes, vision_shapes_list, level_start_index = self._get_encoder_input(
            vision_features
        )

        # todo add denoising for training
        denoise_embeddings, denoise_bboxes, key_padding_mask = None, None, None
        batch_size = task_mask.shape[0]

        # compose attn_mask for vision_emb and task_emb fusion
        task_features = self.task_encoder(task_features)
        if self.task_project is not None:
            task_features = self.task_project(task_features)
        src_key_mask = (task_mask == 0).detach()
        attn_mask_len = self.num_queries
        fusion_size = attn_mask_len + task_features.shape[0]
        key_padding_mask = torch.zeros([batch_size, fusion_size], dtype=torch.bool).to(task_features.device)
        key_padding_mask[:, attn_mask_len:] = src_key_mask
        attention_mask = _prepare_4d_attention_mask(~key_padding_mask, dtype=vision_features.dtype)
        decoder_embeddings, reference_points, encoder_bboxes, encoder_class_similarity, init_reference_points = (
            self._get_decoder_input(
                vision_features, tuple(vision_shapes_list), class_features, denoise_embeddings, denoise_bboxes
            )
        )

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions else None
        predicted_class_features = decoder_embeddings

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (predicted_class_features,)
        decoder_bboxes = []
        decoder_classes = []
        last_refined_bbox = None
        reference_points = reference_points.sigmoid()
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                predicted_class_features, task_features, self_attention, cross_attention = (
                    self._gradient_checkpointing_func(
                        layer.__call__,
                        predicted_class_features,
                        task_features,
                        reference_points,
                        vision_features,
                        vision_shapes,
                        vision_shapes_list,
                        level_start_index=level_start_index,
                        attention_mask=attention_mask,
                        query_position=self.query_position_head(reference_points),
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                )
            else:
                predicted_class_features, task_features, self_attention, cross_attention = layer(
                    predicted_class_features,
                    task_features,
                    reference_points,
                    vision_features,
                    vision_shapes,
                    vision_shapes_list,
                    level_start_index=level_start_index,
                    attention_mask=attention_mask,
                    query_position=self.query_position_head(reference_points),
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            if output_attentions:
                all_self_attns = all_self_attns + (self_attention,)
                all_cross_attns = all_cross_attns + (cross_attention,)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (predicted_class_features,)

            refined_bbox = torch.sigmoid(
                self.decoder_bbox_head[i](predicted_class_features) + _inverse_sigmoid(reference_points)
            )
            original_class_projected = self.decoder_class_head[i](class_features).permute(1, 2, 0)
            if self.training:
                decoder_classes.append(
                    get_class_similarity(
                        class_distance_type=self.class_distance_type,
                        cls_feature=predicted_class_features,
                        class_proj=original_class_projected,
                    )
                )
                if i == 0:
                    decoder_bboxes.append(refined_bbox)
                else:
                    decoder_bboxes.append(
                        torch.sigmoid(
                            self.decoder_bbox_head[i](predicted_class_features) + _inverse_sigmoid(last_refined_bbox)
                        )
                    )
            elif i == self.decoder_num_layers - 1:
                decoder_classes.append(
                    get_class_similarity(self.class_distance_type, predicted_class_features, original_class_projected)
                )
                decoder_bboxes.append(refined_bbox)
                break
            last_refined_bbox = refined_bbox
            reference_points = refined_bbox.detach() if self.training else refined_bbox
        if output_attentions:
            all_attns += (all_self_attns, all_cross_attns)

        last_hidden_state = predicted_class_features
        decoder_bboxes = torch.stack(decoder_bboxes)
        decoder_classes = torch.stack(decoder_classes)

        if not return_dict:
            return (
                last_hidden_state,
                all_hidden_states,
                all_attns,
                decoder_bboxes,
                decoder_classes,
                encoder_bboxes,
                encoder_class_similarity,
                init_reference_points,
                reference_points,
            )

        return OmDetTurboDecoderOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            decoder_coords=decoder_bboxes,
            decoder_classes=decoder_classes,
            encoder_coord_logits=encoder_bboxes,
            encoder_class_logits=encoder_class_similarity,
            init_reference_points=init_reference_points,
            intermediate_reference_points=reference_points,
        )


@add_start_docstrings(
    """
    OmDetTurbo Model (consisting of a vision and a text backbone, and encoder-decoder architecture) outputting
    bounding boxes and classes scores for tasks such as COCO detection.
    """,
    OMDET_TURBO_START_DOCSTRING,
)
class OmDetTurboForObjectDetection(OmDetTurboPreTrainedModel):
    def __init__(self, config: OmDetTurboConfig):
        super().__init__(config)
        self.vision_backbone = OmDetTurboVisionBackbone(config)
        self.language_backbone = OmDetTurboLanguageBackbone(config)
        self.encoder = OmDetTurboHybridEncoder(config)
        self.decoder = OmDetTurboDecoder(config)
        self.num_queries = config.num_queries

        self.language_cache_class = OmDetTurboLRUCache(config.cache_size)
        self.language_cache_prompt = OmDetTurboLRUCache(config.cache_size)
        self.vocab_size = config.text_config.vocab_size
        self.post_init()

    def get_input_embeddings(self):
        return self.language_backbone.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_backbone.model.set_input_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_backbone.model.resize_token_embeddings(
            new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of
        )
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    @add_start_docstrings_to_model_forward(OMDET_TURBO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OmDetTurboObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        classes_input_ids: torch.LongTensor,
        classes_attention_mask: torch.LongTensor,
        tasks_input_ids: torch.LongTensor,
        tasks_attention_mask: torch.LongTensor,
        classes_structure: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], OmDetTurboObjectDetectionOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import requests
        >>> from PIL import Image

        >>> from transformers import AutoProcessor, OmDetTurboForObjectDetection

        >>> processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
        >>> model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> classes = ["cat", "remote"]
        >>> task = "Detect {}.".format(", ".join(classes))
        >>> inputs = processor(image, text=classes, task=task, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits)
        >>> results = processor.post_process_grounded_object_detection(
        ...     outputs,
        ...     classes=classes,
        ...     target_sizes=[image.size[::-1]],
        ...     score_threshold=0.3,
        ...     nms_threshold=0.3,
        >>> )[0]
        >>> for score, class_name, box in zip(results["scores"], results["classes"], results["boxes"]):
        ...     box = [round(i, 1) for i in box.tolist()]
        ...     print(
        ...         f"Detected {class_name} with confidence "
        ...         f"{round(score.item(), 2)} at location {box}"
        ...     )
        Detected remote with confidence 0.76 at location [39.9, 71.3, 176.5, 117.9]
        Detected cat with confidence 0.72 at location [345.1, 22.5, 639.7, 371.9]
        Detected cat with confidence 0.65 at location [12.7, 53.8, 315.5, 475.3]
        Detected remote with confidence 0.57 at location [333.4, 75.6, 370.7, 187.0]
        ```"""
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss = None
        image_features = self.vision_backbone(pixel_values)
        encoder_outputs = self.encoder(
            image_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        class_features, task_features, task_mask = self.get_language_embedding(
            classes_input_ids,
            classes_attention_mask,
            tasks_input_ids,
            tasks_attention_mask,
            classes_structure,
        )
        encoder_extracted_states = encoder_outputs.extracted_states if return_dict else encoder_outputs[-1]
        decoder_outputs = self.decoder(
            encoder_extracted_states,
            class_features,
            task_features,
            task_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return tuple(
                output
                for output in [
                    loss,
                    decoder_outputs[3][-1],
                    decoder_outputs[4][-1],
                    decoder_outputs[7],
                    decoder_outputs[8],
                    decoder_outputs[5],
                    decoder_outputs[6],
                    encoder_outputs[-1],
                    decoder_outputs[1],
                    decoder_outputs[2],
                    encoder_outputs[1],
                    encoder_outputs[2],
                    classes_structure,
                ]
                if output is not None
            )

        return OmDetTurboObjectDetectionOutput(
            loss=loss,
            decoder_coord_logits=decoder_outputs.decoder_coords[-1],
            decoder_class_logits=decoder_outputs.decoder_classes[-1],
            init_reference_points=decoder_outputs.init_reference_points,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            encoder_coord_logits=decoder_outputs.encoder_coord_logits,
            encoder_class_logits=decoder_outputs.encoder_class_logits,
            encoder_extracted_states=encoder_outputs.extracted_states,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            classes_structure=classes_structure,
        )


__all__ = ["OmDetTurboForObjectDetection", "OmDetTurboPreTrainedModel"]

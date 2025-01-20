# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Relation DETR model."""

import copy
import functools
import math
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torchvision.ops import Conv2dNormActivation

from ...activations import ACT2CLS, ACT2FN
from ...image_transforms import center_to_corners_format, corners_to_center_format
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import load_backbone
from .configuration_relation_detr import RelationDetrConfig


logger = logging.get_logger(__name__)

MultiScaleDeformableAttention = None


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.load_cuda_kernels
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


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RelationDetrConfig"
_CHECKPOINT_FOR_DOC = "xiuqhou/relation-detr"


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


@dataclass
class RelationDetrDecoderOutput(ModelOutput):
    """
    Base class for outputs of the RelationDetrDecoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    pred_logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RelationDetrDenoisingGeneratorOutput(ModelOutput):
    """
    Base class for outputs of the RelationDetrDenoisingGenerator.

    Args:

        noised_label_query (`torch.FloatTensor` of shape `(batch_size, num_queries, num_labels)`):
            Noised label queries.
        noised_box_query (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Noised box queries.
        denoise_attn_mask (`torch.FloatTensor` of shape `(num_queries, num_queries)`):
            Attention mask for denoising.
        denoising_groups (`int`):
            Number of denoising groups.
        max_gt_num_per_image (`int`):
            Maximum number of ground truth boxes per image.
    """

    noised_label_query: Optional[torch.FloatTensor] = None
    noised_box_query: Optional[torch.FloatTensor] = None
    denoise_attn_mask: Optional[torch.FloatTensor] = None
    denoising_groups: Optional[int] = None
    max_gt_num_per_image: Optional[int] = None


@dataclass
class RelationDetrModelOutput(ModelOutput):
    """
    Base class for outputs of the Relation DETR encoder-decoder model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        hybrid_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, hybrid_queries, hidden_size)`, *optional*, returned only for train mode):
            Sequence of hidden-states at the output of the last layer of the hybrid branch of the model.
        hybrid_intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, hybrid_queries, hidden_size)`, *optional*, returned only for train mode):
            Stacked intermediate hidden states (output of each layer of the hybrid branch).
        hybrid_intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, hybrid_queries, 4)`, *optional*, returned only for train mode):
            Stacked intermediate reference points (reference points of each layer of the hybrid branch).
        hybrid_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned only for train mode and when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, hybrid_queries, hidden_size)`. Hidden-states of the hybrid model at the output of each
            layer plus the initial embedding outputs.
        hybrid_attentions (`tuple(torch.FloatTensor)`, *optional*, returned only for train mode and when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, hybrid_queries,
            hybrid_queries)`. Attentions weights of the hybrid model, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        hybrid_cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned only for train mode and when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, hybrid_queries, num_heads, 4, 4)`.
            Attentions weights of the hybrid model's cross-attention layer, after the attention softmax, used to compute
            the weighted average in the cross-attention heads.
        hybrid_outputs_class (`torch.FloatTensor` of shape `(batch_size, hybrid_queries, config.num_labels)`, *optional*, returned only for train mode):
            Classification logits (including no-object) for all queries.
        hybrid_outputs_coord (`torch.FloatTensor` of shape `(batch_size, hybrid_queries, 4)`, *optional*, returned only for train mode):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding).
        hybrid_enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned only for train mode):
            Predicted bounding boxes scores with top `config.hybrid_queries` scores in the first stage. Output of bounding box binary classification (i.e. foreground and background).
        hybrid_enc_outputs_coord (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned only for train mode):
            Normalized boxes coordinates for queries with top `config.hybrid_queries` scores in the first stage, represented
            as (center_x, center_y, width, height). These values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding possible padding).
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        dec_outputs_class (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_labels)`):
            Classification logits (including no-object) for all queries.
        dec_outputs_coord (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding).
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Predicted bounding boxes scores with top `config.num_queries` scores in the first stage. Output of bounding
            box binary classification (i.e. foreground and background).
        enc_outputs_coord (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`):
            Normalized boxes coordinates for queries with top `config.num_queries` scores in the first stage, represented
            as (center_x, center_y, width, height). These values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding possible padding).
    """

    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    hybrid_last_hidden_state: Optional[torch.FloatTensor] = None
    hybrid_intermediate_hidden_states: Optional[torch.FloatTensor] = None
    hybrid_intermediate_reference_points: Optional[torch.FloatTensor] = None
    hybrid_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hybrid_attentions: Optional[Tuple[torch.FloatTensor]] = None
    hybrid_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    hybrid_outputs_class: Optional[torch.FloatTensor] = None
    hybrid_outputs_coord: Optional[torch.FloatTensor] = None
    hybrid_enc_outputs_class: Optional[torch.FloatTensor] = None
    hybrid_enc_outputs_coord: Optional[torch.FloatTensor] = None
    init_reference_points: torch.FloatTensor = None
    dec_outputs_class: torch.FloatTensor = None
    dec_outputs_coord: torch.FloatTensor = None
    enc_outputs_class: torch.FloatTensor = None
    enc_outputs_coord: torch.FloatTensor = None


@dataclass
class RelationDetrObjectDetectionOutput(ModelOutput):
    """
    Output type of [`RelationDetrForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~RelationDetrProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        auxiliary_outputs (`list[Dict]`):
            It is a list of dictionaries containing the two above keys (`logits` and `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_heads, 4,
            4)`. Attentions weights of the encoder, after the attention softmax, used to compute the weighted average
            in the self-attention heads.
        hybrid_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, hybrid_queries, hidden_size)`, *optional*, returned only for train mode):
            Sequence of hidden-states at the output of the last layer of the hybrid branch of the model.
        hybrid_intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, hybrid_queries, hidden_size)`, *optional*, returned only for train mode):
            Stacked intermediate hidden states (output of each layer of the hybrid branch).
        hybrid_intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, hybrid_queries, 4)`, *optional*, returned only for train mode):
            Stacked intermediate reference points (reference points of each layer of the hybrid branch).
        hybrid_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned only for train mode and when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, hybrid_queries, hidden_size)`. Hidden-states of the hybrid model at the output of each
            layer plus the initial embedding outputs.
        hybrid_attentions (`tuple(torch.FloatTensor)`, *optional*, returned only for train mode and when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, hybrid_queries,
            hybrid_queries)`. Attentions weights of the hybrid model, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        hybrid_cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned only for train mode and when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, hybrid_queries, num_heads, 4, 4)`.
            Attentions weights of the hybrid model's cross-attention layer, after the attention softmax, used to compute
            the weighted average in the cross-attention heads.
        hybrid_outputs_class (`torch.FloatTensor` of shape `(batch_size, hybrid_queries, config.num_labels)`, *optional*, returned only for train mode):
            Classification logits (including no-object) for all queries.
        hybrid_outputs_coord (`torch.FloatTensor` of shape `(batch_size, hybrid_queries, 4)`, *optional*, returned only for train mode):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding).
        hybrid_enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned only for train mode):
            Predicted bounding boxes scores with top `config.hybrid_queries` scores in the first stage. Output of bounding box binary classification (i.e. foreground and background).
        hybrid_enc_outputs_coord (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned only for train mode):
            Normalized boxes coordinates for queries with top `config.hybrid_queries` scores in the first stage, represented
            as (center_x, center_y, width, height). These values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding possible padding).
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        dec_outputs_class (`torch.FloatTensor` of shape `(batch_size, num_queries, config.num_labels)`):
            Classification logits (including no-object) for all queries.
        dec_outputs_coord (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding).
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Predicted bounding boxes scores with top `config.num_queries` scores in the first stage. Output of bounding
            box binary classification (i.e. foreground and background).
        enc_outputs_coord (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`):
            Normalized boxes coordinates for queries with top `config.num_queries` scores in the first stage, represented
            as (center_x, center_y, width, height). These values are normalized in [0, 1], relative to the size of each
            individual image in the batch (disregarding possible padding).
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: List[Dict] = None
    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    hybrid_last_hidden_state: Optional[torch.FloatTensor] = None
    hybrid_intermediate_hidden_states: Optional[torch.FloatTensor] = None
    hybrid_intermediate_reference_points: Optional[torch.FloatTensor] = None
    hybrid_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hybrid_attentions: Optional[Tuple[torch.FloatTensor]] = None
    hybrid_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    hybrid_outputs_class: Optional[torch.FloatTensor] = None
    hybrid_outputs_coord: Optional[torch.FloatTensor] = None
    hybrid_enc_outputs_class: Optional[torch.FloatTensor] = None
    hybrid_enc_outputs_coord: Optional[torch.FloatTensor] = None
    init_reference_points: torch.FloatTensor = None
    dec_outputs_class: torch.FloatTensor = None
    dec_outputs_coord: torch.FloatTensor = None
    enc_outputs_class: torch.FloatTensor = None
    enc_outputs_coord: torch.FloatTensor = None


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.inverse_sigmoid
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->RelationDetr
class RelationDetrFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it user-friendly
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->RelationDetr
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `RelationDetrFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = RelationDetrFrozenBatchNorm2d(module.num_features)

            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class RelationDetrConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by RelationDetrFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config: RelationDetrConfig):
        super().__init__()

        self.config = config
        backbone = load_backbone(config)

        # replace batch norm by frozen batch norm
        replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.channels

        backbone_model_type = None
        if config.backbone is not None:
            backbone_model_type = config.backbone
        elif config.backbone_config is not None:
            backbone_model_type = config.backbone_config.model_type
        else:
            raise ValueError("Either `backbone` or `backbone_config` should be provided in the config")

        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)

        self.backbone_features_format = config.backbone_features_format
        self.post_layer_norm = config.backbone_post_layer_norm

        if self.post_layer_norm:
            self.norms = nn.ModuleList(
                [nn.LayerNorm(channel, eps=config.layer_norm_eps) for channel in self.intermediate_channel_sizes]
            )

    def check_image_format(self, features):
        # do some check on the input according to backbone_features_format
        channel_dim = 1 if self.backbone_features_format == "channels_first" else -1
        for prev_feat, feat in zip(features[:-1], features[1:]):
            if not (prev_feat.shape[channel_dim] < feat.shape[channel_dim]):
                raise ValueError(
                    "Feature maps should be in increasing order of channels, make sure `backbone_features_format` is right"
                )

    def forward(self, pixel_values: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values).feature_maps

        self.check_image_format(features)

        # convert (batch_size, channels, height, width) -> (batch_size, height, width, channels)
        if self.backbone_features_format == "channels_first":
            features = [feature.permute(0, 2, 3, 1) for feature in features]

        # apply layer_norm if needed
        if self.post_layer_norm:
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        # convert (batch_size, height, width, channels) -> (batch_size, channels, height, width)
        features = [feature.permute(0, 3, 1, 2) for feature in features]

        return features


class RelationDetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, config: RelationDetrConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.d_model // 2
        self.temperature = config.sin_cos_temperature
        self.normalize = config.sin_cos_normalize
        self.scale = config.sin_cos_scale
        self.eps = 1e-6
        self.offset = config.sin_cos_offset
        assert (
            isinstance(self.temperature, int) or len(self.temperature) == 2
        ), "Only support (t_x, t_y) or an integer t for temperature"

    def get_dim_t(self, device: torch.device):
        if isinstance(self.temperature, int):
            dim_t = get_dim_t(self.embedding_dim, self.temperature, device)
            return dim_t, dim_t
        return (get_dim_t(self.embedding_dim, t, device) for t in self.temperature)

    def forward(self, pixel_mask: torch.Tensor) -> torch.Tensor:
        y_embed = pixel_mask.cumsum(1)
        x_embed = pixel_mask.cumsum(2)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        else:
            y_embed = y_embed + self.offset
            x_embed = x_embed + self.offset

        dim_tx, dim_ty = self.get_dim_t(pixel_mask.device)

        pos_x = x_embed.unsqueeze(-1) / dim_tx
        pos_y = y_embed.unsqueeze(-1) / dim_ty
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# Copied from transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding
class RelationDetrLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    def forward(self, pixel_values, pixel_mask=None):
        height, width = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention
def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: Union[Tensor, List[Tuple]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
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


# Modified from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention
class RelationDetrMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Relation DETR.
    """

    def __init__(self, config: RelationDetrConfig, num_heads: int, n_points: int):
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
                "You'd better set embed_dim (d_model) in RelationDetrMultiscaleDeformableAttention to make the"
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

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.FloatTensor] = None,
        reference_points: Optional[torch.FloatTensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        spatial_shapes_list: Optional[List[Tuple]] = None,
        level_start_index: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        total_elements = sum(height * width for height, width in spatial_shapes_list)
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

        if self.disable_custom_kernels or MultiScaleDeformableAttention is None:
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


# Modified from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiheadAttention
class RelationDetrMultiheadAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the Relation DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, target_len, embed_dim = hidden_states.size()
        # add position embeddings to the hidden states before projecting to queries and keys
        hidden_states_original = hidden_states
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        # get queries, keys and values
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.size()}"
            )

        # expand attention_mask
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                float_attention_mask = torch.zeros_like(attention_mask, dtype=hidden_states.dtype)
                float_attention_mask.masked_fill_(~attention_mask, float("-inf"))
                attention_mask = float_attention_mask

            if attention_mask.dim() == 2:
                if attention_mask.size() != (target_len, source_len):
                    raise ValueError(
                        f"2D-Attention mask should be of size {(target_len, source_len)}, but is {attention_mask.size()}"
                    )
                # [seq_len, seq_len] -> [batch_size, self.num_heads, target_seq_len, source_seq_len]
                attention_mask = attention_mask.expand(batch_size, self.num_heads, *attention_mask.size())
            elif attention_mask.dim() == 3:
                if attention_mask.size() != (batch_size * self.num_heads, target_len, source_len):
                    raise ValueError(
                        f"3D-Attention mask should be of size {(batch_size, self.num_heads, target_len, source_len)}, but"
                        f" is {attention_mask.size()}"
                    )
                attention_mask = attention_mask.view(batch_size, self.num_heads, target_len, source_len)
            else:
                raise ValueError(
                    f"Attention mask should be of size {(target_len, source_len)} of {(batch_size * self.num_heads, target_len, source_len)}, but is"
                    f" {attention_mask.size()}"
                )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, self.num_heads, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, self.num_heads, target_len, source_len)}, but is"
                    f" {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_len, source_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_len, source_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Modified from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrEncoderLayer with DeformableDetr->RelationDetr
class RelationDetrEncoderLayer(nn.Module):
    def __init__(self, config: RelationDetrConfig):
        super().__init__()
        self.d_model = config.d_model
        self.self_attn = RelationDetrMultiscaleDeformableAttention(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_embeddings: torch.FloatTensor = None,
        reference_points: Optional[torch.FloatTensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        spatial_shapes_list: Optional[List[Tuple]] = None,
        level_start_index: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            spatial_shapes_list (`List[Tuple]`, *optional*):
                List of spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Modified from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrDecoderLayer
class RelationDetrDecoderLayer(nn.Module):
    def __init__(self, config: RelationDetrConfig):
        super().__init__()
        self.d_model = config.d_model

        # self-attention
        self.self_attn = RelationDetrMultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)
        # cross-attention
        self.encoder_attn = RelationDetrMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)
        # feedforward neural networks
        self.fc1 = nn.Linear(self.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        position_embeddings: Optional[torch.FloatTensor] = None,
        reference_points: Optional[torch.FloatTensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        spatial_shapes_list: Optional[List[Tuple]] = None,
        level_start_index: Optional[torch.LongTensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            self_attention_mask (`torch.FloatTensor`):
                Self attention mask of size `(batch, 1, target_len, target_len)` where padding elements are indicated
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, d_model)` where padding elements are indicated.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=self_attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states

        # Cross-Attention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=encoder_attention_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states

        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


# Modified from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrPreTrainedModel
class RelationDetrPreTrainedModel(PreTrainedModel):
    config_class = RelationDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = [r"RelationDetrConvEncoder", r"RelationDetrEncoderLayer", r"RelationDetrDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        xavier_gain = self.config.init_xavier_std

        prior_prob = self.config.init_bias_prior_prob or 1 / (self.config.num_labels + 1)
        bias = float(-math.log((1 - prior_prob) / prior_prob))

        if isinstance(module, RelationDetrDecoder):
            for layer in module.class_head:
                nn.init.xavier_uniform_(layer.weight, xavier_gain)
                nn.init.constant_(layer.bias, bias)

            for layer in module.bbox_head:
                nn.init.constant_(layer.layers[-1].weight, 0)
                nn.init.constant_(layer.layers[-1].bias, 0)

        if isinstance(module, RelationDetrModel):
            nn.init.xavier_uniform_(module.encoder_class_head.weight, xavier_gain)
            nn.init.xavier_uniform_(module.hybrid_class_head.weight, xavier_gain)
            nn.init.constant_(module.encoder_class_head.bias, bias)
            nn.init.constant_(module.hybrid_class_head.bias, bias)
            nn.init.constant_(module.encoder_bbox_head.layers[-1].weight, 0)
            nn.init.constant_(module.encoder_bbox_head.layers[-1].bias, 0)
            nn.init.constant_(module.hybrid_bbox_head.layers[-1].weight, 0)
            nn.init.constant_(module.hybrid_bbox_head.layers[-1].bias, 0)

        if isinstance(module, RelationDetrLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, RelationDetrMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.n_heads, dtype=torch.int64).to(default_dtype) * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data, xavier_gain)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data, xavier_gain)
            nn.init.constant_(module.output_proj.bias.data, 0.0)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if hasattr(module, "reference_points") and not self.config.two_stage:
            nn.init.xavier_uniform_(module.reference_points.weight.data, xavier_gain)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        if hasattr(module, "level_embed"):
            nn.init.normal_(module.level_embed)


RELATION_DETR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RelationDetrConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RELATION_DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`RelationDetrImageProcessor.__call__`]
            for details.

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            Not used by default. Can be used to mask object queries.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            can choose to directly pass a flattened representation of an image.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            embedded representation.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


class RelationDetrEncoder(RelationDetrPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`RelationDetrEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: RelationDetrConfig
    """

    def __init__(self, config: RelationDetrConfig):
        super().__init__(config)
        self.gradient_checkpointing = False

        self.dropout = config.dropout
        self.layers = nn.ModuleList([RelationDetrEncoderLayer(config) for _ in range(config.encoder_layers)])

        self.memory_fusion = nn.Sequential(
            nn.Linear(config.d_model * (config.encoder_layers + 1), config.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device),
                indexing="ij",
            )
            # TODO: valid_ratios could be useless here. check https://github.com/fundamentalvision/Deformable-DETR/issues/36
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        position_embeddings: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        spatial_shapes_list: List,
        level_start_index: torch.LongTensor,
        valid_ratios: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            spatial_shapes_list (`List`):
                Spatial shapes of each feature map with list data type.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points for `RelationDetrMultiscaleDeformableAttention`.
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

        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        spatial_shapes_tuple = tuple(spatial_shapes_list)

        if reference_points is None:
            reference_points = self.get_reference_points(
                spatial_shapes_tuple, valid_ratios, device=inputs_embeds.device
            )

        encoder_states = ()
        all_attentions = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    reference_points,
                    spatial_shapes,
                    spatial_shapes_list,
                    level_start_index,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    position_embeddings=position_embeddings,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    spatial_shapes_list=spatial_shapes_list,
                    level_start_index=level_start_index,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        encoder_states = encoder_states + (hidden_states,)

        hidden_states = torch.cat(encoder_states, dim=-1)
        hidden_states = self.memory_fusion(hidden_states)

        if not output_hidden_states:
            encoder_states = None

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


def box_rel_encoding(src_boxes: torch.FloatTensor, tgt_boxes: torch.FloatTensor, eps: float = 1e-5):
    # construct position relation
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

    return pos_embed


def compile_compatible_lru_cache(*lru_args, **lru_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(torch.compiler, "is_compiling") and not torch.compiler.is_compiling():
                # Cache the function only if the model is not being compiled
                # check if the function is already cached, otherwise create it
                if not hasattr(self, f"_cached_{func.__name__}"):
                    self.__setattr__(
                        f"_cached_{func.__name__}", lru_cache(*lru_args, **lru_kwargs)(func.__get__(self))
                    )
                return self.__getattribute__(f"_cached_{func.__name__}")(*args, **kwargs)
            else:
                # Otherwise, just call the original function
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


@compile_compatible_lru_cache(maxsize=32)
def get_dim_t(num_pos_feats: int, temperature: int, device: torch.device):
    dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32, device=device)
    dim_t = temperature ** (dim_t * 2 / num_pos_feats)
    return dim_t  # (0, 2, 4, ..., ⌊n/2⌋*2)


def exchange_xy_fn(pos_res: torch.FloatTensor):
    index = torch.cat(
        [
            torch.arange(1, -1, -1, device=pos_res.device),
            torch.arange(2, pos_res.shape[-2], device=pos_res.device),
        ]
    )
    pos_res = torch.index_select(pos_res, -2, index)
    return pos_res


def get_sine_pos_embed(
    pos_tensor: torch.FloatTensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
    exchange_xy: bool = True,
) -> Tensor:
    """
    Generate sinusoidal position embedding for a tensor.

    Args:
        pos_tensor (`torch.FloatTensor`):
            Input tensor to be encoded through sinusoidal position encoding.
        num_pos_feats (`int`, *optional*, defaults to 128):
            Projected shape for each element in the tensor.
        temperature (`int`, *optional*, defaults to 10000):
            The temperature used for scaling the position embedding.
        scale (`float`, *optional*, defaults to `2 * math.pi`):
            The scale used for scaling the position embedding.
        exchange_xy (`bool`, *optional*, defaults to True):
            Exchange pos x and pos. For example, if the last dimension of input tensor denotes [x, y, ...],
            the results will be [pos(y), pos(x), ...].
    Returns:
        `torch.FloatTensor` with the last dimension expanded by `num_pos_feats` times.
    """
    dim_t = get_dim_t(num_pos_feats, temperature, pos_tensor.device)

    pos_res = pos_tensor.unsqueeze(-1) * scale / dim_t
    pos_res = torch.stack((pos_res.sin(), pos_res.cos()), dim=-1).flatten(-2)
    if exchange_xy:
        pos_res = exchange_xy_fn(pos_res)
    pos_res = pos_res.flatten(-2)
    return pos_res


class PositionRelationEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 16,
        num_heads: int = 8,
        temperature: float = 10000.0,
        scale: float = 100.0,
        activation_layer: str = "relu",
        inplace: bool = True,
    ):
        super().__init__()
        activation_layer = ACT2CLS[activation_layer]
        self.pos_proj = Conv2dNormActivation(
            embedding_dim * 4,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=embedding_dim,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )

    def forward(self, src_boxes: torch.FloatTensor, tgt_boxes: Optional[torch.FloatTensor] = None):
        """
        Args:
            src_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
                Source boxes.
            tgt_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`, *optional*):
                Target boxes, if not provided, `tgt_boxes` will be set to `src_boxes`.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_heads, num_boxes, num_boxes)`
        """

        if tgt_boxes is None:
            tgt_boxes = src_boxes

        if src_boxes.shape[-1] != 4 or tgt_boxes.shape[-1] != 4:
            raise ValueError("src_boxes and tgt_boxes must have 4 coordinates.")

        with torch.no_grad():
            pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
            pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
        pos_embed = self.pos_proj(pos_embed)

        return pos_embed.clone()


class RelationDetrDecoder(RelationDetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`RelationDetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some tweaks for Relation DETR:

    - `position_embeddings`, `reference_points`, `spatial_shapes` and `valid_ratios` are added to the forward pass.
    - it also returns a stack of intermediate outputs and reference points from all decoding layers.

    Args:
        config: RelationDetrConfig
    """

    def __init__(self, config: RelationDetrConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([RelationDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.ref_point_head = RelationDetrMLPPredictionHead(
            input_dim=2 * config.d_model,
            hidden_dim=config.d_model,
            output_dim=config.d_model,
            num_layers=2,
        )
        self.query_scale = RelationDetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=config.d_model,
            num_layers=2,
        )
        self.gradient_checkpointing = False

        # hack implementation for iterative bounding box refinement and two-stage Relation DETR
        self.class_head = nn.ModuleList(
            [nn.Linear(config.d_model, config.num_labels) for _ in range(config.decoder_layers)]
        )
        self.bbox_head = nn.ModuleList(
            [
                RelationDetrMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                )
                for _ in range(config.decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.position_relation_embedding = PositionRelationEmbedding(
            embedding_dim=config.d_relation,
            num_heads=config.num_attention_heads,
            temperature=config.rel_temperature,
            scale=config.rel_scale,
            activation_layer=config.activation_function,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        reference_points: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        spatial_shapes_list: List,
        level_start_index: torch.LongTensor,
        valid_ratios: torch.FloatTensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        skip_relation: Optional[bool] = False,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
                Reference point in range `[0, 1]`, represented as (center_x, center_y, width, height), top-left (0,0),
                bottom-right (1, 1), not including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            spatial_shapes_list (`List`):
                Spatial shapes of each feature map with list data type.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.
            self_attention_mask (`torch.LongTensor` or `torch.FloatTensor` of shape `(num_queries, num_queries)` or
                `(batch_size * num_heads, num_queries, num_queries)`, *optional*):
                Mask to avoid performing self-attention on padding pixel_values of the decoder. Mask values selected in
                `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
                If input `torch.FloatTensor`, it will be added to self attention weight directly.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            skip_relation (`bool`, *optional*):
                Whether or not to skip the position relation embedding. Defaults to `False`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()
        outputs_classes = ()
        outputs_coords = ()

        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        position_relation = self_attention_mask
        src_boxes = reference_points
        for idx, decoder_layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :], self.config.d_model // 2)
            position_embeddings = self.ref_point_head(query_sine_embed)
            if idx != 0:
                position_embeddings = position_embeddings * self.query_scale(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    reference_points_input,
                    spatial_shapes,
                    spatial_shapes_list,
                    level_start_index,
                    position_relation,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    spatial_shapes_list=spatial_shapes_list,
                    level_start_index=level_start_index,
                    self_attention_mask=position_relation,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            # get output, reference_points are not detached for look_forward_twice
            inv_reference_points = inverse_sigmoid(reference_points)
            outputs_class = self.class_head[idx](self.norm(hidden_states))
            outputs_coord = self.bbox_head[idx](self.norm(hidden_states))
            outputs_coord_logits = outputs_coord + inv_reference_points
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes += (outputs_class,)
            outputs_coords += (outputs_coord,)

            if not skip_relation:
                position_relation = self.position_relation_embedding(src_boxes, outputs_coord).flatten(0, 1)
                # note that True is valid, False is invalid, which is opposite to the official implementation
                if self_attention_mask is not None:
                    position_relation.masked_fill_(~self_attention_mask, float("-inf"))
                src_boxes = outputs_coord

            # hack implementation for iterative bounding box refinement
            reference_points = self.bbox_head[idx](hidden_states) + inv_reference_points.detach()
            reference_points = reference_points.sigmoid()

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        outputs_classes = torch.stack(outputs_classes, dim=1)
        outputs_coords = torch.stack(outputs_coords, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    outputs_classes,
                    outputs_coords,
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return RelationDetrDecoderOutput(
            pred_logits=outputs_classes,
            pred_boxes=outputs_coords,
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class RelationDetrChannelMapper(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        norm_layer=functools.partial(nn.GroupNorm, 32),
        activation_layer: Optional[str] = None,
        dilation: int = 1,
        inplace: bool = True,
        bias: Optional[bool] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.convs = nn.ModuleList()
        self.num_channels = [out_channels] * num_outs
        if activation_layer is not None:
            activation_layer = ACT2CLS[activation_layer]
        for in_channel in in_channels:
            self.convs.append(
                Conv2dNormActivation(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                    groups=groups,
                    dilation=dilation,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    inplace=inplace,
                )
            )
        for _ in range(num_outs - len(in_channels)):
            self.convs.append(
                Conv2dNormActivation(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=bias,
                    groups=groups,
                    dilation=dilation,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    inplace=inplace,
                )
            )
            in_channel = out_channels

    def forward(self, inputs: List[torch.FloatTensor]):
        """
        Map inputs into specific embed_dim. If the length of inputs is smaller than convolution modules,
        only the last `len(inputs) + extra` will be used for mapping.

        Args:
            inputs: List of torch.FloatTensor.

        Returns:
            List of mapped torch.FloatTensor
        """
        assert len(inputs) <= len(self.in_channels)
        start = len(self.in_channels) - len(inputs)
        convs = self.convs[start:]
        outs = [convs[i](inputs[i]) for i in range(len(inputs))]
        for i in range(len(inputs), len(convs)):
            if i == len(inputs):
                outs.append(convs[i](inputs[-1]))
            else:
                outs.append(convs[i](outs[-1]))
        return outs


@add_start_docstrings(
    """
    The bare Relation DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    RELATION_DETR_START_DOCSTRING,
)
class RelationDetrModel(RelationDetrPreTrainedModel):
    def __init__(self, config: RelationDetrConfig):
        super().__init__(config)

        # Create backbone + positional encoding
        self.backbone = RelationDetrConvEncoder(config)
        self.position_embeddings = RelationDetrSinePositionEmbedding(config)

        # Create input projection layers
        self.neck = RelationDetrChannelMapper(
            in_channels=self.backbone.intermediate_channel_sizes,
            out_channels=config.d_model,
            num_outs=config.num_feature_levels,
        )

        self.encoder = RelationDetrEncoder(config)
        self.decoder = RelationDetrDecoder(config)

        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))
        self.target_embed = nn.Embedding(config.num_queries, config.d_model)
        self.hybrid_target_embed = nn.Embedding(config.hybrid_queries, config.d_model)

        self.encoder_output = nn.Linear(config.d_model, config.d_model)
        self.encoder_output_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.encoder_class_head = nn.Linear(config.d_model, config.num_labels)
        self.hybrid_class_head = nn.Linear(config.d_model, config.num_labels)

        self.encoder_bbox_head = RelationDetrMLPPredictionHead(config.d_model, config.d_model, 4, 3)
        self.hybrid_bbox_head = RelationDetrMLPPredictionHead(config.d_model, config.d_model, 4, 3)

        # initilize encoder and hybrid classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        nn.init.constant_(self.hybrid_class_head.bias, bias_value)
        # initiailize encoder and hybrid regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)
        nn.init.constant_(self.hybrid_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.hybrid_bbox_head.layers[-1].bias, 0.0)

        self.two_stage_num_proposals = config.num_queries
        self.hybrid_num_proposals = config.hybrid_queries
        self.num_labels = config.num_labels

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    @staticmethod
    def flatten_multi_level(multi_level_elements: torch.Tensor):
        multi_level_elements = torch.cat(tensors=[e.flatten(-2) for e in multi_level_elements], dim=-1)  # (b, [c], s)
        if multi_level_elements.ndim == 3:
            multi_level_elements.transpose_(1, 2)
        return multi_level_elements

    def multi_level_misc(self, multi_level_masks):
        spatial_shapes = [m.shape[-2:] for m in multi_level_masks]
        spatial_shapes = multi_level_masks[0].new_tensor(spatial_shapes, dtype=torch.int64)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = self.multi_level_valid_ratios(multi_level_masks)
        return spatial_shapes, level_start_index, valid_ratios

    @staticmethod
    def get_valid_ratios(mask: torch.BoolTensor):
        b, h, w = mask.shape
        if h == 0 or w == 0:  # for empty Tensor
            return mask.new_ones((b, 2)).float()
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)  # [n, 2]
        return valid_ratio

    def multi_level_valid_ratios(self, multi_level_masks: List[torch.BoolTensor]):
        # note that True is valid and False is invalid for multi_level_masks
        # which is opposite to the official implementation
        return torch.stack([self.get_valid_ratios(~m) for m in multi_level_masks], 1)

    @staticmethod
    def get_full_reference_points(spatial_shapes: torch.LongTensor, valid_ratios: torch.FloatTensor):
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.arange(0.5, h + 0.5, device=spatial_shapes.device),
                torch.arange(0.5, w + 0.5, device=spatial_shapes.device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)  # [n, h*w, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [n, s, 2]
        return reference_points

    def get_reference(self, spatial_shapes: torch.LongTensor, valid_ratios: torch.FloatTensor):
        # get full_reference_points, should be transferred using valid_ratios
        full_reference_points = self.get_full_reference_points(spatial_shapes, valid_ratios)
        reference_points = full_reference_points[:, :, None] * valid_ratios[:, None]
        # get proposals, reuse full_reference_points to speed up
        level_wh = full_reference_points.new_tensor([[i] for i in range(spatial_shapes.shape[0])])
        level_wh = 0.05 * 2.0 ** level_wh.repeat_interleave(spatial_shapes.prod(-1), 0)
        level_wh = level_wh.expand_as(full_reference_points)
        proposals = torch.cat([full_reference_points, level_wh], -1)
        return reference_points, proposals

    def get_lvl_pos_embed(self, multi_level_pos_embeds: List[torch.FloatTensor]):
        multi_level_pos_embeds = [p + l.view(1, -1, 1, 1) for p, l in zip(multi_level_pos_embeds, self.level_embed)]
        return self.flatten_multi_level(multi_level_pos_embeds)

    def get_encoder_output(
        self, memory: torch.FloatTensor, proposals: torch.FloatTensor, memory_padding_mask: torch.BoolTensor
    ):
        output_proposals_valid = ((proposals > 0.01) & (proposals < 0.99)).all(-1, keepdim=True)
        proposals = torch.log(proposals / (1 - proposals))  # inverse_sigmoid
        invalid = memory_padding_mask.unsqueeze(-1) | ~output_proposals_valid
        proposals.masked_fill_(invalid, float("inf"))

        output_memory = memory * (~memory_padding_mask.unsqueeze(-1)) * (output_proposals_valid)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, proposals

    def get_multi_levels(self, pixel_values: torch.FloatTensor, pixel_mask: torch.LongTensor):
        # extract higher features matching proto_levels
        multi_level_feats = self.backbone(pixel_values)

        # apply neck to get multi_level_feats
        multi_level_feats = self.neck(multi_level_feats)

        # extract multi_level masks and pos_embeds
        pixel_mask = pixel_mask.to(pixel_values.dtype)
        multi_level_masks = [
            F.interpolate(pixel_mask[None], size=feat.shape[-2:])[0].to(torch.bool) for feat in multi_level_feats
        ]

        # extract multi_level_pos_embeds
        multi_level_pos_embeds = [self.position_embeddings(m) for m in multi_level_masks]

        return multi_level_feats, multi_level_masks, multi_level_pos_embeds

    @add_start_docstrings_to_model_forward(RELATION_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RelationDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        self_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        noised_label_query: Optional[torch.FloatTensor] = None,
        noised_box_query: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, RelationDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("xiuqhou/relation-detr-resnet50")
        >>> model = RelationDetrModel.from_pretrained("xiuqhou/relation-detr-resnet50")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 900, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model`
        sources, masks, position_embeddings_list = self.get_multi_levels(pixel_values, pixel_mask)

        source_flatten = self.flatten_multi_level(sources)
        mask_flatten = self.flatten_multi_level(masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(position_embeddings_list)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(masks)
        reference_points, proposals = self.get_reference(spatial_shapes, valid_ratios)
        spatial_shapes_list = spatial_shapes.tolist()

        # Sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=source_flatten,
                attention_mask=mask_flatten,
                reference_points=reference_points,
                position_embeddings=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Prepare decoder inputs
        batch_size, _, num_channels = encoder_outputs[0].shape
        enc_outputs_class = None
        enc_outputs_coord_logits = None
        object_query_embedding, output_proposals = self.get_encoder_output(
            encoder_outputs[0], proposals, ~mask_flatten
        )

        # Linear projection for bounding box binary classification (i.e. foreground and background)
        enc_outputs_class = self.encoder_class_head(object_query_embedding)
        delta_bbox = self.encoder_bbox_head(object_query_embedding)
        enc_outputs_coord_logits = delta_bbox + output_proposals

        # Only keep top scoring `config.num_queries` proposals
        topk = min(self.config.num_queries, enc_outputs_class.shape[1])
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
        topk_class = torch.gather(enc_outputs_class, 1, topk_proposals.repeat(1, 1, self.num_labels))
        topk_coords_logits = torch.gather(enc_outputs_coord_logits, 1, topk_proposals.repeat(1, 1, 4))
        topk_coords = topk_coords_logits.sigmoid()

        # Get target and reference points
        reference_points = topk_coords.detach()
        target = self.target_embed.weight.expand(batch_size, -1, -1)[:, :topk]

        if self.training:
            # Get hybrid classes and coordinates, target and reference points
            hybrid_enc_class = self.hybrid_class_head(object_query_embedding)
            hybrid_enc_coord = self.hybrid_bbox_head(object_query_embedding) + output_proposals
            hybrid_enc_coord = hybrid_enc_coord.sigmoid()
            topk = min(self.hybrid_num_proposals, hybrid_enc_class.shape[1])
            topk_index = torch.topk(hybrid_enc_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
            hybrid_enc_class = hybrid_enc_class.gather(1, topk_index.expand(-1, -1, self.num_labels))
            hybrid_enc_coord = hybrid_enc_coord.gather(1, topk_index.expand(-1, -1, 4))
            hybrid_reference_points = hybrid_enc_coord.detach()
            hybrid_target = self.hybrid_target_embed.weight.expand(batch_size, -1, -1)[:, :topk]
        else:
            hybrid_enc_class = hybrid_enc_coord = None

        # Combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=encoder_outputs[0],
            self_attention_mask=self_attention_mask,
            encoder_attention_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs_classes = decoder_outputs.pred_logits if return_dict else decoder_outputs[0]
        outputs_coords = decoder_outputs.pred_boxes if return_dict else decoder_outputs[1]

        if self.training:
            hybrid_outputs = self.decoder(
                inputs_embeds=hybrid_target,
                encoder_hidden_states=encoder_outputs[0],
                self_attention_mask=None,
                encoder_attention_mask=mask_flatten,
                reference_points=hybrid_reference_points,
                spatial_shapes=spatial_shapes,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                skip_relation=True,
            )
            hybrid_outputs_classes = hybrid_outputs.pred_logits if return_dict else hybrid_outputs[0]
            hybrid_outputs_coords = hybrid_outputs.pred_boxes if return_dict else hybrid_outputs[1]

        else:
            hybrid_outputs_classes = hybrid_outputs_coords = None
            hybrid_outputs = RelationDetrDecoderOutput() if return_dict else ()

        if not return_dict:
            decoder_mediate_outputs = decoder_outputs[2:]
            hybrid_mediate_output = hybrid_outputs[2:]
            # We need to reorganize the outputs sequence so that optional outputs will not affect
            # the indexing of the final outputs
            outputs = (reference_points, outputs_classes, outputs_coords, topk_class, topk_coords)
            hybrid_outputs = tuple(
                value
                for value in [
                    hybrid_outputs_classes,
                    hybrid_outputs_coords,
                    hybrid_enc_class,
                    hybrid_enc_coord,
                ]
                if value is not None
            )
            tuple_outputs = (
                decoder_mediate_outputs + encoder_outputs + hybrid_mediate_output + hybrid_outputs + outputs
            )

            return tuple_outputs

        return RelationDetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,  # optional
            decoder_attentions=decoder_outputs.attentions,  # optional
            cross_attentions=decoder_outputs.cross_attentions,  # optional
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,  # optional
            encoder_attentions=encoder_outputs.attentions,  # optional
            hybrid_last_hidden_state=hybrid_outputs.last_hidden_state,
            hybrid_intermediate_hidden_states=hybrid_outputs.intermediate_hidden_states,  # optional
            hybrid_intermediate_reference_points=hybrid_outputs.intermediate_reference_points,  # optional
            hybrid_hidden_states=hybrid_outputs.hidden_states,  # optional
            hybrid_attentions=hybrid_outputs.attentions,  # optional
            hybrid_cross_attentions=hybrid_outputs.cross_attentions,  # optional
            hybrid_outputs_class=hybrid_outputs_classes,
            hybrid_outputs_coord=hybrid_outputs_coords,
            hybrid_enc_outputs_class=hybrid_enc_class,
            hybrid_enc_outputs_coord=hybrid_enc_coord,
            init_reference_points=reference_points,
            dec_outputs_class=outputs_classes,
            dec_outputs_coord=outputs_coords,
            enc_outputs_class=topk_class,
            enc_outputs_coord=topk_coords,
        )


# Modified from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class RelationDetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.FloatTensor):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GenerateDNQueries(nn.Module):
    """
    Generate denoising queries as introduced in https://arxiv.org/abs/2203.01305

    Args:
        num_queries (`int`, *optional*, defaults to 300):
            Number of total denoising queries.
        num_classes (`int`, *optional*, defaults to 80):
            Number of total categories.
        label_embed_dim (`int`, *optional*, defaults to 256):
            The embedding dimension for label encoding.
        denoising_groups (`int`, *optional*, defaults to 5):
            Number of noised ground truth groups.
        label_noise_ratio (`float`, *optional*, defaults to 0.2):
            The probability of the label being noised.
        box_noise_scale (`float`, *optional*, defaults to 0.4):
            Scaling factor for box noising.
        with_indicator (`bool`, *optional*, defaults to `False`):
            Whether to add indicator in noised label/box queries.
    """

    def __init__(
        self,
        num_queries: int = 300,
        num_classes: int = 80,
        label_embed_dim: int = 256,
        denoising_groups: int = 5,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 0.4,
        with_indicator: bool = False,
        return_dict: bool = True,
    ):
        super(GenerateDNQueries, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.denoising_groups = denoising_groups
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.with_indicator = with_indicator
        self.return_dict = return_dict

        # leave one dim for indicator mentioned in DN-DETR
        if with_indicator:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim - 1)
        else:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    @staticmethod
    def apply_label_noise(labels: torch.Tensor, label_noise_ratio: float = 0.2, num_classes: int = 80):
        if label_noise_ratio > 0:
            mask = torch.rand_like(labels.float()) < label_noise_ratio
            noised_labels = torch.randint_like(labels, 0, num_classes)
            noised_labels = torch.where(mask, noised_labels, labels)
            return noised_labels
        else:
            return labels

    @staticmethod
    def apply_box_noise(boxes: torch.FloatTensor, box_noise_scale: float = 0.4):
        """
        Args:
            boxes (`torch.FloatTensor` of shape `(num_boxes, 4)`):
                Bounding boxes in format `(center_x, center_y, width, height)` with shape `(num_boxes, 4)`.
            box_noise_scale (`float`, *optional*, defaults to 0.4):
                Scaling factor for box noising.

        Returns:
            `torch.FloatTensor`, noised bounding boxes with shape `(num_boxes, 4)`.
        """
        if box_noise_scale > 0:
            diff = torch.zeros_like(boxes)
            diff[:, :2] = boxes[:, 2:] / 2
            diff[:, 2:] = boxes[:, 2:]
            boxes += torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff) * box_noise_scale
            boxes = boxes.clamp(min=0.0, max=1.0)
        return boxes

    def generate_query_masks(self, max_gt_num_per_image: int, device: torch.device):
        noised_query_nums = max_gt_num_per_image * self.denoising_groups
        tgt_size = noised_query_nums + self.num_queries
        attn_mask = torch.zeros(tgt_size, tgt_size, device=device, dtype=torch.bool)
        # match query cannot see the reconstruct
        attn_mask[noised_query_nums:, :noised_query_nums] = True
        for i in range(self.denoising_groups):
            start_col = start_row = max_gt_num_per_image * i
            end_col = end_row = max_gt_num_per_image * (i + 1)
            assert noised_query_nums >= end_col and start_col >= 0, "check attn_mask"
            attn_mask[start_row:end_row, :start_col] = True
            attn_mask[start_row:end_row, end_col:noised_query_nums] = True
        return attn_mask

    def forward(
        self,
        gt_labels_list: List[torch.LongTensor],
        gt_boxes_list: List[torch.FloatTensor],
        return_dict: Optional[bool] = None,
    ):
        """
        Args:
            gt_labels_list (`List[torch.LongTensor]` of len `(batch_size,)`):
                Classification labels for images in a batch.
            gt_boxes_list (`List[torch.FloatTensor]` of len `(batch_size,)`):
                Ground truth bounding boxes for images in a batch, in format `(center_x, center_y, width, height)`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """

        return_dict = return_dict if return_dict is not None else self.return_dict

        # concat ground truth labels and boxes in one batch
        # e.g. [tensor([0, 1, 2]), tensor([2, 3, 4])] -> tensor([0, 1, 2, 2, 3, 4])
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)

        # For efficient denoising, repeat the original ground truth labels and boxes to
        # create more training denoising samples.
        # e.g. tensor([0, 1, 2, 2, 3, 4]) -> tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4]) if group = 2.
        gt_labels = gt_labels.repeat(self.denoising_groups, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.denoising_groups, 1)

        # set the device as "gt_labels"
        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)

        batch_size = len(gt_labels_list)

        # the number of ground truth per image in one batch
        # e.g. [tensor([0, 1]), tensor([2, 3, 4])] -> gt_nums_per_image: [2, 3]
        # means there are 2 instances in the first image and 3 instances in the second image
        gt_nums_per_image = [x.numel() for x in gt_labels_list]

        # Add noise on labels and boxes
        noised_labels = self.apply_label_noise(gt_labels, self.label_noise_ratio, self.num_classes)
        noised_boxes = self.apply_box_noise(gt_boxes, self.box_noise_scale)
        noised_boxes = inverse_sigmoid(noised_boxes)

        # encoding labels
        label_embedding = self.label_encoder(noised_labels)
        query_num = label_embedding.shape[0]

        # add indicator to label encoding if with_indicator == True
        if self.with_indicator:
            label_embedding = torch.cat([label_embedding, torch.ones([query_num, 1], device=device)], 1)

        # calculate the max number of ground truth in one image inside the batch.
        # e.g. gt_nums_per_image = [2, 3] which means
        # the first image has 2 instances and the second image has 3 instances
        # then the max_gt_num_per_image should be 3.
        max_gt_num_per_image = max(gt_nums_per_image)

        # the total denoising queries is depended on denoising groups and max number of instances.
        noised_query_nums = max_gt_num_per_image * self.denoising_groups

        # initialize the generated noised queries to zero.
        # And the zero initialized queries will be assigned with noised embeddings later.
        noised_label_queries = torch.zeros(batch_size, noised_query_nums, self.label_embed_dim, device=device)
        noised_box_queries = torch.zeros(batch_size, noised_query_nums, 4, device=device)

        # batch index per image: [0, 1, 2, 3] for batch_size == 4
        batch_idx = torch.arange(0, batch_size)

        # e.g. gt_nums_per_image = [2, 3]
        # batch_idx = [0, 1]
        # then the "batch_idx_per_instance" equals to [0, 0, 1, 1, 1]
        # which indicates which image the instance belongs to.
        # cuz the instances has been flattened before.
        batch_idx_per_instance = torch.repeat_interleave(batch_idx, torch.tensor(gt_nums_per_image).long())

        # indicate which image the noised labels belong to. For example:
        # noised label: tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4])
        # batch_idx_per_group: tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        # which means the first label "tensor([0])"" belongs to "image_0".
        batch_idx_per_group = batch_idx_per_instance.repeat(self.denoising_groups, 1).flatten()

        # Cuz there might be different numbers of ground truth in each image of the same batch.
        # So there might be some padding part in noising queries.
        # Here we calculate the indexes for the valid queries and
        # fill them with the noised embeddings.
        # And leave the padding part to zeros.
        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat([torch.arange(num) for num in gt_nums_per_image])
            valid_index_per_group = torch.cat(
                [valid_index_per_group + max_gt_num_per_image * i for i in range(self.denoising_groups)]
            ).long()
        if len(batch_idx_per_group):
            noised_label_queries[(batch_idx_per_group, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group, valid_index_per_group)] = noised_boxes

        # generate attention masks for transformer layers
        attn_mask = self.generate_query_masks(max_gt_num_per_image, device)

        if not return_dict:
            output = (noised_label_queries, noised_box_queries, attn_mask, self.denoising_groups, max_gt_num_per_image)
            return output

        return RelationDetrDenoisingGeneratorOutput(
            noised_label_query=noised_label_queries,
            noised_box_query=noised_box_queries,
            denoise_attn_mask=attn_mask,
            denoising_groups=self.denoising_groups,
            max_gt_num_per_image=max_gt_num_per_image,
        )


class GenerateCDNQueries(GenerateDNQueries):
    def __init__(
        self,
        num_queries: int = 300,
        num_classes: int = 80,
        label_embed_dim: int = 256,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        return_dict: bool = True,
    ):
        super().__init__(
            num_queries=num_queries,
            num_classes=num_classes,
            label_embed_dim=label_embed_dim,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            denoising_groups=1,
            return_dict=return_dict,
        )

        self.num_denoising = num_denoising
        self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    def apply_box_noise(self, boxes: torch.FloatTensor, box_noise_scale: float = 0.4):
        """
        Args:
            boxes (`torch.FloatTensor` of shape `(num_boxes, 4)`):
                Bounding boxes in format `(center_x, center_y, width, height)` with shape `(num_boxes, 4)`.
            box_noise_scale (`float`, *optional*, defaults to 0.4):
                Scaling factor for box noising.

        Returns:
            `torch.FloatTensor`, noised bounding boxes with shape `(num_boxes, 4)`.
        """
        num_boxes = len(boxes) // self.denoising_groups // 2
        positive_idx = torch.arange(num_boxes, dtype=torch.long, device=boxes.device)
        positive_idx = positive_idx.unsqueeze(0).repeat(self.denoising_groups, 1)
        positive_idx += (
            torch.arange(self.denoising_groups, dtype=torch.long, device=boxes.device).unsqueeze(1) * num_boxes * 2
        )
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + num_boxes
        if box_noise_scale > 0:
            diff = torch.zeros_like(boxes)
            diff[:, :2] = boxes[:, 2:] / 2
            diff[:, 2:] = boxes[:, 2:] / 2
            rand_sign = torch.randint_like(boxes, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(boxes)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            xyxy_boxes = center_to_corners_format(boxes)
            xyxy_boxes += torch.mul(rand_part, diff) * box_noise_scale
            xyxy_boxes = xyxy_boxes.clamp(min=0.0, max=1.0)
            boxes = corners_to_center_format(xyxy_boxes)

        return boxes

    def forward(
        self,
        gt_labels_list: List[torch.LongTensor],
        gt_boxes_list: List[torch.FloatTensor],
        return_dict: Optional[bool] = None,
    ):
        """
        Args:
            gt_labels_list (`List[torch.LongTensor]` of len `(batch_size,)`):
                Classification labels for images in a batch.
            gt_boxes_list (`List[torch.FloatTensor]` of len `(batch_size,)`):
                Ground truth bounding boxes for images in a batch, in format `(center_x, center_y, width, height)`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.return_dict

        # the number of ground truth per image in one batch
        # e.g. [tensor([0, 1]), tensor([2, 3, 4])] -> gt_nums_per_image: [2, 3]
        # means there are 2 instances in the first image and 3 instances in the second image
        gt_nums_per_image = [x.numel() for x in gt_labels_list]

        # calculate the max number of ground truth in one image inside the batch.
        # e.g. gt_nums_per_image = [2, 3] which means
        # the first image has 2 instances and the second image has 3 instances
        # then the max_gt_num_per_image should be 3.
        max_gt_num_per_image = max(gt_nums_per_image)

        # get denoising_groups, which is 1 for empty ground truth
        denoising_groups = self.num_denoising * max_gt_num_per_image // max(max_gt_num_per_image**2, 1)
        self.denoising_groups = max(denoising_groups, 1)

        # concat ground truth labels and boxes in one batch
        # e.g. [tensor([0, 1, 2]), tensor([2, 3, 4])] -> tensor([0, 1, 2, 2, 3, 4])
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)

        # For efficient denoising, repeat the original ground truth labels and boxes to
        # create more training denoising samples.
        # each group has positive and negative. e.g. if group = 2, tensor([0, 1, 2, 2, 3, 4]) ->
        # tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4]).
        gt_labels = gt_labels.repeat(self.denoising_groups * 2, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.denoising_groups * 2, 1)

        # set the device as "gt_labels"
        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)

        batch_size = len(gt_labels_list)

        # Add noise on labels and boxes
        noised_labels = self.apply_label_noise(gt_labels, self.label_noise_ratio * 0.5, self.num_classes)
        noised_boxes = self.apply_box_noise(gt_boxes, self.box_noise_scale)
        noised_boxes = inverse_sigmoid(noised_boxes)

        # encoding labels
        label_embedding = self.label_encoder(noised_labels)

        # the total denoising queries is depended on denoising groups and max number of instances.
        noised_query_nums = max_gt_num_per_image * self.denoising_groups * 2

        # initialize the generated noised queries to zero.
        # And the zero initialized queries will be assigned with noised embeddings later.
        noised_label_queries = torch.zeros(batch_size, noised_query_nums, self.label_embed_dim, device=device)
        noised_box_queries = torch.zeros(batch_size, noised_query_nums, 4, device=device)

        # batch index per image: [0, 1, 2, 3] for batch_size == 4
        batch_idx = torch.arange(0, batch_size)

        # e.g. gt_nums_per_image = [2, 3]
        # batch_idx = [0, 1]
        # then the "batch_idx_per_instance" equals to [0, 0, 1, 1, 1]
        # which indicates which image the instance belongs to.
        # cuz the instances has been flattened before.
        batch_idx_per_instance = torch.repeat_interleave(batch_idx, torch.tensor(gt_nums_per_image, dtype=torch.long))

        # indicate which image the noised labels belong to. For example:
        # noised label: tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4， 0, 1, 2, 2, 3, 4])
        # batch_idx_per_group: tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        # which means the first label "tensor([0])"" belongs to "image_0".
        batch_idx_per_group = batch_idx_per_instance.repeat(self.denoising_groups * 2, 1).flatten()

        # Cuz there might be different numbers of ground truth in each image of the same batch.
        # So there might be some padding part in noising queries.
        # Here we calculate the indexes for the valid queries and
        # fill them with the noised embeddings.
        # And leave the padding part to zeros.
        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat([torch.arange(num) for num in gt_nums_per_image])
            valid_index_per_group = torch.cat(
                [valid_index_per_group + max_gt_num_per_image * i for i in range(self.denoising_groups * 2)]
            ).long()
        if len(batch_idx_per_group):
            noised_label_queries[(batch_idx_per_group, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group, valid_index_per_group)] = noised_boxes

        # generate attention masks for transformer layers
        attn_mask = self.generate_query_masks(2 * max_gt_num_per_image, device)

        if not return_dict:
            output = (
                noised_label_queries,
                noised_box_queries,
                attn_mask,
                self.denoising_groups,
                2 * max_gt_num_per_image,
            )
            return output

        return RelationDetrDenoisingGeneratorOutput(
            noised_label_query=noised_label_queries,
            noised_box_query=noised_box_queries,
            denoise_attn_mask=attn_mask,
            denoising_groups=self.denoising_groups,
            max_gt_num_per_image=2 * max_gt_num_per_image,
        )


@add_start_docstrings(
    """
    Relation DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    RELATION_DETR_START_DOCSTRING,
)
class RelationDetrForObjectDetection(RelationDetrPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    # _tied_weights_keys = [r"bbox_head\.[1-9]\d*", r"class_head\.[1-9]\d*"]
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    def __init__(self, config: RelationDetrConfig):
        super().__init__(config)

        # Relation DETR encoder-decoder model
        self.model = RelationDetrModel(config)

        self.denoising_generator = GenerateCDNQueries(
            num_queries=config.num_queries,
            num_classes=config.num_labels,
            label_embed_dim=config.d_model,
            num_denoising=config.num_denoising,
            label_noise_ratio=config.label_noise_ratio,
            box_noise_scale=config.box_noise_scale,
            return_dict=config.use_return_dict,
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RELATION_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RelationDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        # decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], RelationDetrObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, RelationDetrForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("xiuqhou/relation-detr-resnet50")
        >>> model = RelationDetrForObjectDetection.from_pretrained("xiuqhou/relation-detr-resnet50")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)
        >>> result = results[0]
        >>> for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected cat with confidence 0.956 at location [343.8, 24.9, 639.52, 371.71]
        Detected cat with confidence 0.953 at location [12.6, 54.34, 316.37, 471.86]
        Detected remote with confidence 0.946 at location [40.09, 73.49, 175.52, 118.06]
        Detected remote with confidence 0.897 at location [333.09, 76.71, 369.77, 187.4]
        Detected couch with confidence 0.896 at location [0.44, 0.53, 640.44, 475.54]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.config.num_denoising > 0 and labels is not None:
            # collect ground truth for denoising generation
            gt_labels_list = [t["class_labels"] for t in labels]
            gt_boxes_list = [t["boxes"] for t in labels]
            noised_embeddings = self.denoising_generator(gt_labels_list, gt_boxes_list)
            noised_label_query = noised_embeddings.noised_label_query if return_dict else noised_embeddings[0]
            noised_box_query = noised_embeddings.noised_box_query if return_dict else noised_embeddings[1]
            self_attention_mask = noised_embeddings.denoise_attn_mask if return_dict else noised_embeddings[2]
            self_attention_mask = (
                ~self_attention_mask
            )  # note multi-head attention mask is opposite to that of torchvision
            denoising_groups = noised_embeddings.denoising_groups if return_dict else noised_embeddings[3]
            max_gt_num_per_image = noised_embeddings.max_gt_num_per_image if return_dict else noised_embeddings[4]

            denoising_meta_values = {
                "dn_num_group": denoising_groups,
                "max_gt_num_per_image": max_gt_num_per_image,
                "dn_num_split": [denoising_groups * max_gt_num_per_image, self.config.num_queries],
            }
        else:
            noised_label_query = noised_box_query = self_attention_mask = None
            denoising_meta_values = None

        # First, sent images through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            self_attention_mask=self_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            noised_label_query=noised_label_query,
            noised_box_query=noised_box_query,
        )

        outputs_class = outputs.dec_outputs_class if return_dict else outputs[-4]
        outputs_coord = outputs.dec_outputs_coord if return_dict else outputs[-3]

        # layer is at the second dimension
        logits = outputs_class[:, -1]
        pred_boxes = outputs_coord[:, -1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        hybrid_loss = hybrid_loss_dict = None
        if labels is not None:
            enc_topk_logits = outputs.enc_outputs_class if return_dict else outputs[-2]
            enc_topk_coords = outputs.enc_outputs_coord if return_dict else outputs[-1]

            loss, loss_dict, auxiliary_outputs = self.loss_function(
                labels=labels,
                config=self.config,
                outputs_class=outputs_class,
                outputs_coord=outputs_coord,
                enc_topk_logits=enc_topk_logits,
                enc_topk_bboxes=enc_topk_coords,
                denoising_meta_values=denoising_meta_values,
            )

            if self.training:
                hybrid_outputs_class = outputs.hybrid_outputs_class if return_dict else outputs[-9]
                hybrid_outputs_coord = outputs.hybrid_outputs_coord if return_dict else outputs[-8]
                hybrid_enc_topk_logits = outputs.hybrid_enc_outputs_class if return_dict else outputs[-7]
                hybrid_enc_topk_coords = outputs.hybrid_enc_outputs_coord if return_dict else outputs[-6]
                hybrid_labels = copy.deepcopy(labels)
                for label in hybrid_labels:
                    if "boxes" in label:
                        label["boxes"] = label["boxes"].repeat(self.config.hybrid_assign, 1)
                    if "class_labels" in label:
                        label["class_labels"] = label["class_labels"].repeat(self.config.hybrid_assign)
                hybrid_loss, hybrid_loss_dict, _ = self.loss_function(
                    labels=hybrid_labels,
                    config=self.config,
                    outputs_class=hybrid_outputs_class,
                    outputs_coord=hybrid_outputs_coord,
                    enc_topk_logits=hybrid_enc_topk_logits,
                    enc_topk_bboxes=hybrid_enc_topk_coords,
                    denoising_meta_values=None,
                )
                loss = loss + hybrid_loss
                loss_dict.update({k + "_hybrid": v for k, v in hybrid_loss_dict.items()})

        if not return_dict:
            outputs = (loss, loss_dict, logits, pred_boxes, auxiliary_outputs) + outputs
            outputs = tuple(output for output in outputs if output is not None)

            return outputs

        dict_outputs = RelationDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            hybrid_last_hidden_state=outputs.hybrid_last_hidden_state,
            hybrid_intermediate_hidden_states=outputs.hybrid_intermediate_hidden_states,
            hybrid_intermediate_reference_points=outputs.hybrid_intermediate_reference_points,
            hybrid_hidden_states=outputs.hybrid_hidden_states,
            hybrid_attentions=outputs.hybrid_attentions,
            hybrid_cross_attentions=outputs.hybrid_cross_attentions,
            hybrid_outputs_class=outputs.hybrid_outputs_class,
            hybrid_outputs_coord=outputs.hybrid_outputs_coord,
            hybrid_enc_outputs_class=outputs.hybrid_enc_outputs_class,
            hybrid_enc_outputs_coord=outputs.hybrid_enc_outputs_coord,
            init_reference_points=outputs.init_reference_points,
            dec_outputs_class=outputs.dec_outputs_class,
            dec_outputs_coord=outputs.dec_outputs_coord,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord=outputs.enc_outputs_coord,
        )

        return dict_outputs


__all__ = [
    "RelationDetrForObjectDetection",
    "RelationDetrModel",
    "RelationDetrPreTrainedModel",
]

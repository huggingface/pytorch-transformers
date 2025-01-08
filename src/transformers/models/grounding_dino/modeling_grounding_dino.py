# coding=utf-8
# Copyright 2024 IDEA Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Grounding DINO model."""

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_timm_available,
    is_torch_cuda_available,
    replace_return_docstrings,
    requires_backends,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_grounding_dino import GroundingDinoConfig


if is_timm_available():
    from timm import create_model


logger = logging.get_logger(__name__)

MultiScaleDeformableAttention = None


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


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GroundingDinoConfig"
_CHECKPOINT_FOR_DOC = "IDEA-Research/grounding-dino-tiny"


@dataclass
class GroundingDinoDecoderOutput(ModelOutput):
    """
    Base class for outputs of the GroundingDinoDecoder. This class adds two attributes to
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
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GroundingDinoEncoderOutput(ModelOutput):
    """
    Base class for outputs of the GroundingDinoEncoder. This class extends BaseModelOutput, due to:
    - vision and text last hidden states
    - vision and text intermediate hidden states

    Args:
        last_hidden_state_vision (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the vision encoder.
        last_hidden_state_text (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the text encoder.
        vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the vision embeddings + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the vision encoder at the
            output of each layer plus the initial embedding outputs.
        text_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the text embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the text encoder at the output of
            each layer plus the initial embedding outputs.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the text-vision attention, vision-text attention, text-enhancer (self-attention) and
            multi-scale deformable attention heads.
    """

    last_hidden_state_vision: torch.FloatTensor = None
    last_hidden_state_text: torch.FloatTensor = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GroundingDinoModelOutput(ModelOutput):
    """
    Base class for outputs of the Grounding DINO encoder-decoder model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
        encoder_last_hidden_state_vision (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_last_hidden_state_text (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the vision embeddings + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the vision encoder at the
            output of each layer plus the initial embedding outputs.
        encoder_text_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the text embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the text encoder at the output of
            each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the text-vision attention, vision-text attention, text-enhancer (self-attention) and
            multi-scale deformable attention heads. attention softmax, used to compute the weighted average in the
            bi-attention heads.
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.num_queries` scoring bounding boxes are picked as
            region proposals in the first stage. Output of bounding box binary classification (i.e. foreground and
            background).
        enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
    """

    last_hidden_state: torch.FloatTensor = None
    init_reference_points: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_last_hidden_state_vision: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state_text: Optional[torch.FloatTensor] = None
    encoder_vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    enc_outputs_class: Optional[torch.FloatTensor] = None
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None


@dataclass
class GroundingDinoObjectDetectionOutput(ModelOutput):
    """
    Output type of [`GroundingDinoForObjectDetection`].

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
            possible padding). You can use [`~GroundingDinoProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        auxiliary_outputs (`List[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention, cross-attention and multi-scale deformable attention heads.
        encoder_last_hidden_state_vision (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_last_hidden_state_text (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the vision embeddings + one for the output of each
            layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the vision encoder at the
            output of each layer plus the initial embedding outputs.
        encoder_text_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the text embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the text encoder at the output of
            each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of tuples of `torch.FloatTensor` (one for attention for each layer) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the text-vision attention, vision-text attention, text-enhancer (self-attention) and
            multi-scale deformable attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.two_stage=True`):
            Predicted bounding boxes scores where the top `config.num_queries` scoring bounding boxes are picked as
            region proposals in the first stage. Output of bounding box binary classification (i.e. foreground and
            background).
        enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.two_stage=True`):
            Logits of predicted bounding boxes coordinates in the first stage.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    init_reference_points: Optional[torch.FloatTensor] = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_last_hidden_state_vision: Optional[torch.FloatTensor] = None
    encoder_last_hidden_state_text: Optional[torch.FloatTensor] = None
    encoder_vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    enc_outputs_class: Optional[torch.FloatTensor] = None
    enc_outputs_coord_logits: Optional[torch.FloatTensor] = None


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->GroundingDino
class GroundingDinoFrozenBatchNorm2d(nn.Module):
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


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->GroundingDino
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `GroundingDinoFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = GroundingDinoFrozenBatchNorm2d(module.num_features)

            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class GroundingDinoConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by GroundingDinoFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.use_timm_backbone:
            requires_backends(self, ["timm"])
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                **config.backbone_kwargs,
            )
        else:
            backbone = load_backbone(config)

        # replace batch norm by frozen batch norm
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

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

    # Copied from transformers.models.detr.modeling_detr.DetrConvEncoder.forward with Detr->GroundingDino
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


# Copied from transformers.models.detr.modeling_detr.DetrConvModel with Detr->GroundingDino
class GroundingDinoConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(self, pixel_values, pixel_mask):
        # send pixel_values and pixel_mask through backbone to get list of (feature_map, pixel_mask) tuples
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # position encoding
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


class GroundingDinoSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.d_model // 2
        self.temperature = config.positional_embedding_temperature
        self.scale = 2 * math.pi

    def forward(self, pixel_values, pixel_mask):
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class GroundingDinoLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, config):
        super().__init__()

        embedding_dim = config.d_model // 2
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


def build_position_encoding(config):
    if config.position_embedding_type == "sine":
        position_embedding = GroundingDinoSinePositionEmbedding(config)
    elif config.position_embedding_type == "learned":
        position_embedding = GroundingDinoLearnedPositionEmbedding(config)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


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


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention with DeformableDetr->GroundingDino, Deformable DETR->Grounding DINO
class GroundingDinoMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, config: GroundingDinoConfig, num_heads: int, n_points: int):
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
                "You'd better set embed_dim (d_model) in GroundingDinoMultiscaleDeformableAttention to make the"
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
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
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
            output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
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
                output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output, attention_weights


class GroundingDinoTextEnhancerLayer(nn.Module):
    """Vanilla Transformer with text embeddings as input"""

    def __init__(self, config):
        super().__init__()
        self.self_attn = GroundingDinoMultiheadAttention(
            config, num_attention_heads=config.encoder_attention_heads // 2
        )

        # Implementation of Feedforward model
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim // 2)
        self.fc2 = nn.Linear(config.encoder_ffn_dim // 2, config.d_model)

        self.layer_norm_before = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layer_norm_after = nn.LayerNorm(config.d_model, config.layer_norm_eps)

        self.activation = ACT2FN[config.activation_function]
        self.num_heads = config.encoder_attention_heads // 2
        self.dropout = config.text_enhancer_dropout

    def with_pos_embed(self, hidden_state: Tensor, position_embeddings: Optional[Tensor]):
        return hidden_state if position_embeddings is None else hidden_state + position_embeddings

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_masks: Optional[torch.BoolTensor] = None,
        position_embeddings: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Text self-attention to enhance projection of text features generated by
        the text encoder (AutoModel based on text_config) within GroundingDinoEncoderLayer

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                Text features generated by the text encoder.
            attention_masks (`torch.BoolTensor`, *optional*):
                Attention mask for text self-attention. False for real tokens and True for padding tokens.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings to be added to the hidden states.

        Returns:
            `tuple(torch.FloatTensor)` comprising two elements:
            - **hidden_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) --
                Output of the text self-attention layer.
            - **attention_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length,
              sequence_length)`) --
                Attention weights of the text self-attention layer.
        """

        # repeat attn mask
        if attention_masks.dim() == 3 and attention_masks.shape[0] == hidden_states.shape[0]:
            # batch_size, num_queries, num_keys
            attention_masks = attention_masks[:, None, :, :]
            attention_masks = attention_masks.repeat(1, self.num_heads, 1, 1)

            dtype = hidden_states.dtype
            attention_masks = attention_masks.to(dtype=dtype)  # fp16 compatibility
            attention_masks = (1.0 - attention_masks) * torch.finfo(dtype).min

        queries = keys = self.with_pos_embed(hidden_states, position_embeddings)
        attention_output, attention_weights = self.self_attn(
            queries=queries,
            keys=keys,
            values=hidden_states,
            attention_mask=attention_masks,
            output_attentions=True,
        )
        attention_output = nn.functional.dropout(attention_output, p=self.dropout, training=self.training)
        hidden_states = hidden_states + attention_output
        hidden_states = self.layer_norm_before(hidden_states)

        residual = hidden_states
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states + residual
        hidden_states = self.layer_norm_after(hidden_states)

        return hidden_states, attention_weights


class GroundingDinoBiMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        vision_dim = text_dim = config.d_model
        embed_dim = config.encoder_ffn_dim // 2
        num_heads = config.encoder_attention_heads // 2
        dropout = config.fusion_dropout

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by `num_heads` (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.vision_proj = nn.Linear(self.vision_dim, self.embed_dim)
        self.text_proj = nn.Linear(self.text_dim, self.embed_dim)
        self.values_vision_proj = nn.Linear(self.vision_dim, self.embed_dim)
        self.values_text_proj = nn.Linear(self.text_dim, self.embed_dim)

        self.out_vision_proj = nn.Linear(self.embed_dim, self.vision_dim)
        self.out_text_proj = nn.Linear(self.embed_dim, self.text_dim)

    def _reshape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        vision_attention_mask: Optional[torch.BoolTensor] = None,
        text_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Image-to-text and text-to-image cross-attention

        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_dim)`):
                Projected flattened image features generated by the vision backbone.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`):
                Projected text features generated by the text encoder.
            vision_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for image-to-text cross-attention. False for real tokens and True for padding tokens.
            text_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for text-to-image cross-attention. False for real tokens and True for padding tokens.

        Returns:
            `tuple(tuple(torch.FloatTensor), tuple(torch.FloatTensor))` where each inner tuple comprises an attention
            output and weights:
            - **vision_attn_output** (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_din)`)
              --
                Output of the image-to-text cross-attention layer.
            - **vision_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, vision_sequence_length,
              vision_sequence_length)`) --
                Attention weights of the image-to-text cross-attention layer.
            - **text_attn_output** (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`) --
                Output of the text-to-image cross-attention layer.
            - **text_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, text_sequence_length,
              text_sequence_length)`) --
                Attention weights of the text-to-image cross-attention layer.
        """
        batch_size, tgt_len, _ = vision_features.size()

        vision_query_states = self.vision_proj(vision_features) * self.scale
        vision_query_states = self._reshape(vision_query_states, tgt_len, batch_size)

        text_key_states = self.text_proj(text_features)
        text_key_states = self._reshape(text_key_states, -1, batch_size)

        vision_value_states = self.values_vision_proj(vision_features)
        vision_value_states = self._reshape(vision_value_states, -1, batch_size)

        text_value_states = self.values_text_proj(text_features)
        text_value_states = self._reshape(text_value_states, -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)

        vision_query_states = vision_query_states.view(*proj_shape)
        text_key_states = text_key_states.view(*proj_shape)
        vision_value_states = vision_value_states.view(*proj_shape)
        text_value_states = text_value_states.view(*proj_shape)

        src_len = text_key_states.size(1)
        attn_weights = torch.bmm(vision_query_states, text_key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (batch_size * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        attn_weights = attn_weights - attn_weights.max()
        # Do not increase -50000/50000, data type half has quite limited range
        attn_weights = torch.clamp(attn_weights, min=-50000, max=50000)

        attn_weights_transposed = attn_weights.transpose(1, 2)
        text_attn_weights = attn_weights_transposed - torch.max(attn_weights_transposed, dim=-1, keepdim=True)[0]

        # Do not increase -50000/50000, data type half has quite limited range
        text_attn_weights = torch.clamp(text_attn_weights, min=-50000, max=50000)

        # mask vision for language
        if vision_attention_mask is not None:
            vision_attention_mask = (
                vision_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            text_attn_weights.masked_fill_(vision_attention_mask, float("-inf"))

        text_attn_weights = text_attn_weights.softmax(dim=-1)

        # mask language for vision
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_weights.masked_fill_(text_attention_mask, float("-inf"))
        vision_attn_weights = attn_weights.softmax(dim=-1)

        vision_attn_probs = F.dropout(vision_attn_weights, p=self.dropout, training=self.training)
        text_attn_probs = F.dropout(text_attn_weights, p=self.dropout, training=self.training)

        vision_attn_output = torch.bmm(vision_attn_probs, text_value_states)
        text_attn_output = torch.bmm(text_attn_probs, vision_value_states)

        if vision_attn_output.size() != (batch_size * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`vision_attn_output` should be of size {(batch_size, self.num_heads, tgt_len, self.head_dim)}, but is {vision_attn_output.size()}"
            )

        if text_attn_output.size() != (batch_size * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`text_attn_output` should be of size {(batch_size, self.num_heads, src_len, self.head_dim)}, but is {text_attn_output.size()}"
            )

        vision_attn_output = vision_attn_output.view(batch_size, self.num_heads, tgt_len, self.head_dim)
        vision_attn_output = vision_attn_output.transpose(1, 2)
        vision_attn_output = vision_attn_output.reshape(batch_size, tgt_len, self.embed_dim)

        text_attn_output = text_attn_output.view(batch_size, self.num_heads, src_len, self.head_dim)
        text_attn_output = text_attn_output.transpose(1, 2)
        text_attn_output = text_attn_output.reshape(batch_size, src_len, self.embed_dim)

        vision_attn_output = self.out_vision_proj(vision_attn_output)
        text_attn_output = self.out_text_proj(text_attn_output)

        return (vision_attn_output, vision_attn_weights), (text_attn_output, text_attn_weights)


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->GroundingDino
class GroundingDinoDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class GroundingDinoFusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        drop_path = config.fusion_droppath

        # pre layer norm
        self.layer_norm_vision = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layer_norm_text = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.attn = GroundingDinoBiMultiHeadAttention(config)

        # add layer scale for training stability
        self.drop_path = GroundingDinoDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        init_values = 1e-4
        self.vision_param = nn.Parameter(init_values * torch.ones((config.d_model)), requires_grad=True)
        self.text_param = nn.Parameter(init_values * torch.ones((config.d_model)), requires_grad=True)

    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        attention_mask_vision: Optional[torch.BoolTensor] = None,
        attention_mask_text: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Image and text features fusion

        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_dim)`):
                Projected flattened image features generated by the vision backbone.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`):
                Projected text features generated by the text encoder.
            attention_mask_vision (`torch.BoolTensor`, **optional**):
                Attention mask for image-to-text cross-attention. False for real tokens and True for padding tokens.
            attention_mask_text (`torch.BoolTensor`, **optional**):
                Attention mask for text-to-image cross-attention. False for real tokens and True for padding tokens.

        Returns:
            `tuple(tuple(torch.FloatTensor), tuple(torch.FloatTensor))` where each inner tuple comprises an enhanced
            feature and attention output and weights:
            - **vision_features** (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, vision_dim)`) --
                Updated vision features with attention output from image-to-text cross-attention layer.
            - **vision_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, vision_sequence_length,
              vision_sequence_length)`) --
                Attention weights of the image-to-text cross-attention layer.
            - **text_features** (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, text_dim)`) --
                Updated text features with attention output from text-to-image cross-attention layer.
            - **text_attn_weights** (`torch.FloatTensor` of shape `(batch_size, num_heads, text_sequence_length,
              text_sequence_length)`) --
                Attention weights of the text-to-image cross-attention layer.
        """
        vision_features = self.layer_norm_vision(vision_features)
        text_features = self.layer_norm_text(text_features)
        (delta_v, vision_attn), (delta_t, text_attn) = self.attn(
            vision_features,
            text_features,
            vision_attention_mask=attention_mask_vision,
            text_attention_mask=attention_mask_text,
        )
        vision_features = vision_features + self.drop_path(self.vision_param * delta_v)
        text_features = text_features + self.drop_path(self.text_param * delta_t)

        return (vision_features, vision_attn), (text_features, text_attn)


class GroundingDinoDeformableLayer(nn.Module):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = GroundingDinoMultiscaleDeformableAttention(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
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

        return hidden_states, attn_weights


# Based on https://github.com/IDEA-Research/GroundingDINO/blob/2b62f419c292ca9c518daae55512fabc3fead4a4/groundingdino/models/GroundingDINO/utils.py#L24
def get_sine_pos_embed(
    pos_tensor: torch.Tensor, num_pos_feats: int = 128, temperature: int = 10000, exchange_xy: bool = True
) -> Tensor:
    """
    Generate sine position embeddings from a position tensor.

    Args:
        pos_tensor (torch.Tensor):
            Tensor containing positions. Shape: [..., n].
        num_pos_feats (`int`, *optional*, defaults to 128):
            Projected shape for each float in the tensor.
        temperature (`int`, *optional*, defaults to 10000):
            Temperature in the sine/cosine function.
        exchange_xy (`bool`, *optional*, defaults to `True`):
            Exchange pos x and pos y. For example, input tensor is [x,y], the results will be [pos(y), pos(x)].

    Returns:
        position_embeddings (torch.Tensor): shape: [..., n * hidden_size].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_tensor = pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)
    position_embeddings = [sine_func(x) for x in pos_tensor]
    if exchange_xy:
        position_embeddings[0], position_embeddings[1] = position_embeddings[1], position_embeddings[0]
    position_embeddings = torch.cat(position_embeddings, dim=-1)
    return position_embeddings


class GroundingDinoEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.d_model = config.d_model

        self.text_enhancer_layer = GroundingDinoTextEnhancerLayer(config)
        self.fusion_layer = GroundingDinoFusionLayer(config)
        self.deformable_layer = GroundingDinoDeformableLayer(config)

    def get_text_position_embeddings(
        self,
        text_features: Tensor,
        text_position_embedding: Optional[torch.Tensor],
        text_position_ids: Optional[torch.Tensor],
    ) -> Tensor:
        batch_size, seq_length, _ = text_features.shape
        if text_position_embedding is None and text_position_ids is None:
            text_position_embedding = torch.arange(seq_length, device=text_features.device)
            text_position_embedding = text_position_embedding.float()
            text_position_embedding = text_position_embedding.unsqueeze(0).unsqueeze(-1)
            text_position_embedding = text_position_embedding.repeat(batch_size, 1, 1)
            text_position_embedding = get_sine_pos_embed(
                text_position_embedding, num_pos_feats=self.d_model, exchange_xy=False
            )
        if text_position_ids is not None:
            text_position_embedding = get_sine_pos_embed(
                text_position_ids[..., None], num_pos_feats=self.d_model, exchange_xy=False
            )

        return text_position_embedding

    def forward(
        self,
        vision_features: Tensor,
        vision_position_embedding: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        key_padding_mask: Tensor,
        reference_points: Tensor,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_position_embedding: Optional[Tensor] = None,
        text_self_attention_masks: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,
    ):
        text_position_embedding = self.get_text_position_embeddings(
            text_features, text_position_embedding, text_position_ids
        )

        (vision_features, vision_fused_attn), (text_features, text_fused_attn) = self.fusion_layer(
            vision_features=vision_features,
            text_features=text_features,
            attention_mask_vision=key_padding_mask,
            attention_mask_text=text_attention_mask,
        )

        (text_features, text_enhanced_attn) = self.text_enhancer_layer(
            hidden_states=text_features,
            attention_masks=~text_self_attention_masks,  # note we use ~ for mask here
            position_embeddings=(text_position_embedding if text_position_embedding is not None else None),
        )

        (vision_features, vision_deformable_attn) = self.deformable_layer(
            hidden_states=vision_features,
            attention_mask=~key_padding_mask,
            position_embeddings=vision_position_embedding,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        return (
            (vision_features, text_features),
            (vision_fused_attn, text_fused_attn, text_enhanced_attn, vision_deformable_attn),
        )


class GroundingDinoMultiheadAttention(nn.Module):
    """Equivalent implementation of nn.MultiheadAttention with `batch_first=True`."""

    def __init__(self, config, num_attention_heads=None):
        super().__init__()
        if config.hidden_size % num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(config.hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout)

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
            # Apply the attention mask is (precomputed for all layers in GroundingDinoModel forward() function)
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


class GroundingDinoDecoderLayer(nn.Module):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self-attention
        self.self_attn = GroundingDinoMultiheadAttention(config, num_attention_heads=config.decoder_attention_heads)

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # cross-attention text
        self.encoder_attn_text = GroundingDinoMultiheadAttention(
            config, num_attention_heads=config.decoder_attention_heads
        )
        self.encoder_attn_text_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # cross-attention
        self.encoder_attn = GroundingDinoMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        vision_encoder_hidden_states: Optional[torch.Tensor] = None,
        vision_encoder_attention_mask: Optional[torch.Tensor] = None,
        text_encoder_hidden_states: Optional[torch.Tensor] = None,
        text_encoder_attention_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states

        # Self Attention
        queries = keys = self.with_pos_embed(hidden_states, position_embeddings)
        hidden_states, self_attn_weights = self.self_attn(
            queries=queries,
            keys=keys,
            values=hidden_states,
            attention_mask=self_attn_mask,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states

        # Cross-Attention Text
        queries = self.with_pos_embed(hidden_states, position_embeddings)
        hidden_states, text_cross_attn_weights = self.encoder_attn_text(
            queries=queries,
            keys=text_encoder_hidden_states,
            values=text_encoder_hidden_states,
            attention_mask=text_encoder_attention_mask,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = second_residual + hidden_states
        hidden_states = self.encoder_attn_text_layer_norm(hidden_states)

        third_residual = hidden_states

        # Cross-Attention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            attention_mask=vision_encoder_attention_mask,
            encoder_hidden_states=vision_encoder_hidden_states,
            encoder_attention_mask=vision_encoder_attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = third_residual + hidden_states
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
            outputs += (self_attn_weights, text_cross_attn_weights, cross_attn_weights)

        return outputs


class GroundingDinoContrastiveEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_text_len = config.max_text_len

    def forward(
        self,
        vision_hidden_state: torch.FloatTensor,
        text_hidden_state: torch.FloatTensor,
        text_token_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        output = vision_hidden_state @ text_hidden_state.transpose(-1, -2)
        output = output.masked_fill(~text_token_mask[:, None, :], float("-inf"))

        # padding to max_text_len
        new_output = torch.full((*output.shape[:-1], self.max_text_len), float("-inf"), device=output.device)
        new_output[..., : output.shape[-1]] = output

        return new_output


class GroundingDinoPreTrainedModel(PreTrainedModel):
    config_class = GroundingDinoConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        std = self.config.init_std

        if isinstance(module, GroundingDinoLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, GroundingDinoMultiscaleDeformableAttention):
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
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)
        elif isinstance(module, GroundingDinoBiMultiHeadAttention):
            nn.init.xavier_uniform_(module.vision_proj.weight)
            module.vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.text_proj.weight)
            module.text_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.values_vision_proj.weight)
            module.values_vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.values_text_proj.weight)
            module.values_text_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.out_vision_proj.weight)
            module.out_vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.out_text_proj.weight)
            module.out_text_proj.bias.data.fill_(0)
        elif isinstance(module, (GroundingDinoEncoderLayer, GroundingDinoDecoderLayer)):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=std)
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
        elif isinstance(module, GroundingDinoMLPPredictionHead):
            nn.init.constant_(module.layers[-1].weight.data, 0)
            nn.init.constant_(module.layers[-1].bias.data, 0)

        if hasattr(module, "reference_points") and not self.config.two_stage:
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        if hasattr(module, "level_embed"):
            nn.init.normal_(module.level_embed)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GroundingDinoDecoder):
            module.gradient_checkpointing = value


GROUNDING_DINO_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GroundingDinoConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GROUNDING_DINO_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`GroundingDinoImageProcessor.__call__`] for
            details.

        input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`BertTokenizer.__call__`] for details.

        token_type_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`: 0 corresponds to a `sentence A` token, 1 corresponds to a `sentence B` token

            [What are token type IDs?](../glossary#token-type-ids)

        attention_mask (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are real (i.e. **not masked**),
            - 0 for tokens that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state_vision`, *optional*: `last_hidden_state_text`, *optional*:
            `vision_hidden_states`, *optional*: `text_hidden_states`, *optional*: `attentions`)
            `last_hidden_state_vision` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence
            of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the
            decoder.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


class GroundingDinoEncoder(GroundingDinoPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`GroundingDinoEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: GroundingDinoConfig
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([GroundingDinoEncoderLayer(config) for _ in range(config.encoder_layers)])

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map.

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
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
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
        vision_features: Tensor,
        vision_attention_mask: Tensor,
        vision_position_embedding: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios=None,
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_position_embedding: Optional[Tensor] = None,
        text_self_attention_masks: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            vision_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 0 for pixel features that are real (i.e. **not masked**),
                - 1 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            vision_position_embedding (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_seq_len, hidden_size)`):
                Flattened text features that are passed to the encoder.
            text_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Mask to avoid performing attention on padding text features. Mask values selected in `[0, 1]`:
                - 0 for text features that are real (i.e. **not masked**),
                - 1 for text features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            text_position_embedding (`torch.FloatTensor` of shape `(batch_size, text_seq_len)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            text_self_attention_masks (`torch.BoolTensor` of shape `(batch_size, text_seq_len, text_seq_len)`):
                Masks to avoid performing attention between padding text features. Mask values selected in `[0, 1]`:
                - 1 for text features that are real (i.e. **not masked**),
                - 0 for text features that are padding (i.e. **masked**).
            text_position_ids (`torch.LongTensor` of shape `(batch_size, num_queries)`):
                Position ids for text features.
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

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=vision_features.device)

        encoder_vision_states = () if output_hidden_states else None
        encoder_text_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        all_attn_fused_text = () if output_attentions else None
        all_attn_fused_vision = () if output_attentions else None
        all_attn_enhanced_text = () if output_attentions else None
        all_attn_deformable = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_vision_states += (vision_features,)
                encoder_text_states += (text_features,)

            (vision_features, text_features), attentions = encoder_layer(
                vision_features=vision_features,
                vision_position_embedding=vision_position_embedding,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=vision_attention_mask,
                reference_points=reference_points,
                text_features=text_features,
                text_attention_mask=text_attention_mask,
                text_position_embedding=text_position_embedding,
                text_self_attention_masks=text_self_attention_masks,
                text_position_ids=text_position_ids,
            )

            if output_attentions:
                all_attn_fused_vision += (attentions[0],)
                all_attn_fused_text += (attentions[1],)
                all_attn_enhanced_text += (attentions[2],)
                all_attn_deformable += (attentions[3],)

        if output_hidden_states:
            encoder_vision_states += (vision_features,)
            encoder_text_states += (text_features,)

        if output_attentions:
            all_attns = (all_attn_fused_vision, all_attn_fused_text, all_attn_enhanced_text, all_attn_deformable)

        if not return_dict:
            enc_outputs = [vision_features, text_features, encoder_vision_states, encoder_text_states, all_attns]
            return tuple(v for v in enc_outputs if v is not None)
        return GroundingDinoEncoderOutput(
            last_hidden_state_vision=vision_features,
            last_hidden_state_text=text_features,
            vision_hidden_states=encoder_vision_states,
            text_hidden_states=encoder_text_states,
            attentions=all_attns,
        )


class GroundingDinoDecoder(GroundingDinoPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`GroundingDinoDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some tweaks for Grounding DINO:

    - `position_embeddings`, `reference_points`, `spatial_shapes` and `valid_ratios` are added to the forward pass.
    - it also returns a stack of intermediate outputs and reference points from all decoding layers.

    Args:
        config: GroundingDinoConfig
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layers = nn.ModuleList([GroundingDinoDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.reference_points_head = GroundingDinoMLPPredictionHead(
            config.query_dim // 2 * config.d_model, config.d_model, config.d_model, 2
        )
        self.gradient_checkpointing = False

        # hack implementation for iterative bounding box refinement as in two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.query_scale = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds,
        vision_encoder_hidden_states,
        vision_encoder_attention_mask=None,
        text_encoder_hidden_states=None,
        text_encoder_attention_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        self_attn_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            vision_encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Last hidden state from encoder related to vision feature map.
            vision_encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
            text_encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, text_seq_len, hidden_size)`):
                Last hidden state from encoder related to text features.
            text_encoder_attention_mask (`torch.Tensor` of shape `(batch_size, text_seq_len)`, *optional*):
                Mask to avoid performing attention on padding text features. Mask values selected in `[0, 1]`:
                - 0 for text features that are real (i.e. **not masked**),
                - 1 for text features that are padding (i.e. **masked**).
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.
            self_attn_mask (`torch.BoolTensor` of shape `(batch_size, text_seq_len)`):
                Masks to avoid performing self-attention between vision hidden state. Mask values selected in `[0, 1]`:
                - 1 for queries that are real (i.e. **not masked**),
                - 0 for queries that are padding (i.e. **masked**).
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

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_attns = () if output_attentions else None
        all_cross_attns_vision = () if (output_attentions and vision_encoder_hidden_states is not None) else None
        all_cross_attns_text = () if (output_attentions and text_encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()

        if text_encoder_attention_mask is not None:
            dtype = text_encoder_hidden_states.dtype

            text_encoder_attention_mask = text_encoder_attention_mask[:, None, None, :]
            text_encoder_attention_mask = text_encoder_attention_mask.repeat(
                1, self.config.decoder_attention_heads, self.config.num_queries, 1
            )
            text_encoder_attention_mask = text_encoder_attention_mask.to(dtype=dtype)
            text_encoder_attention_mask = text_encoder_attention_mask * torch.finfo(dtype).min

        for idx, decoder_layer in enumerate(self.layers):
            num_coordinates = reference_points.shape[-1]
            if num_coordinates == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            elif num_coordinates == 2:
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            else:
                raise ValueError("Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
            query_pos = get_sine_pos_embed(reference_points_input[:, :, 0, :], num_pos_feats=self.config.d_model // 2)
            query_pos = self.reference_points_head(query_pos)

            # In original implementation they apply layer norm before outputting intermediate hidden states
            # Though that's not through between layers so the layers use as input the output of the previous layer
            # withtout layer norm
            if output_hidden_states:
                all_hidden_states += (self.layer_norm(hidden_states),)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    query_pos,
                    reference_points_input,
                    spatial_shapes,
                    level_start_index,
                    vision_encoder_hidden_states,
                    vision_encoder_attention_mask,
                    text_encoder_hidden_states,
                    text_encoder_attention_mask,
                    self_attn_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=query_pos,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    vision_encoder_hidden_states=vision_encoder_hidden_states,
                    vision_encoder_attention_mask=vision_encoder_attention_mask,
                    text_encoder_hidden_states=text_encoder_hidden_states,
                    text_encoder_attention_mask=text_encoder_attention_mask,
                    self_attn_mask=self_attn_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                num_coordinates = reference_points.shape[-1]
                if num_coordinates == 4:
                    new_reference_points = tmp + torch.special.logit(reference_points, eps=1e-5)
                    new_reference_points = new_reference_points.sigmoid()
                elif num_coordinates == 2:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + torch.special.logit(reference_points, eps=1e-5)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    raise ValueError(
                        f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
                    )
                reference_points = new_reference_points.detach()

            intermediate += (self.layer_norm(hidden_states),)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if text_encoder_hidden_states is not None:
                    all_cross_attns_text += (layer_outputs[2],)

                if vision_encoder_hidden_states is not None:
                    all_cross_attns_vision += (layer_outputs[3],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if output_attentions:
            all_attns += (all_self_attns, all_cross_attns_text, all_cross_attns_vision)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_attns,
                ]
                if v is not None
            )
        return GroundingDinoDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


# these correspond to [CLS], [SEP], . and ?
SPECIAL_TOKENS = [101, 102, 1012, 1029]


def generate_masks_with_special_tokens_and_transfer_map(input_ids: torch.LongTensor) -> Tuple[Tensor, Tensor]:
    """Generate attention mask between each pair of special tokens and positional ids.
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
    Returns:
        `tuple(torch.Tensor)` comprising attention mask between each special tokens and position_ids:
        - **attention_mask** (`torch.BoolTensor` of shape `(batch_size, sequence_length, sequence_length)`)
        - **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
    """
    batch_size, num_token = input_ids.shape
    # special_tokens_mask: batch_size, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((batch_size, num_token), device=input_ids.device).bool()
    for special_token in SPECIAL_TOKENS:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(batch_size, 1, 1)
    position_ids = torch.zeros((batch_size, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )

        previous_col = col

    return attention_mask, position_ids.to(torch.long)


@add_start_docstrings(
    """
    The bare Grounding DINO Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    GROUNDING_DINO_START_DOCSTRING,
)
class GroundingDinoModel(GroundingDinoPreTrainedModel):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        # Create backbone + positional encoding
        backbone = GroundingDinoConvEncoder(config)
        position_embeddings = build_position_encoding(config)
        self.backbone = GroundingDinoConvModel(backbone, position_embeddings)

        # Create input projection layers
        if config.num_feature_levels > 1:
            num_backbone_outs = len(backbone.intermediate_channel_sizes)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = backbone.intermediate_channel_sizes[i]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
                in_channels = config.d_model
            self.input_proj_vision = nn.ModuleList(input_proj_list)
        else:
            self.input_proj_vision = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                ]
            )

        # Create text backbone
        self.text_backbone = AutoModel.from_config(
            config.text_config, add_pooling_layer=False, attn_implementation=config._attn_implementation
        )
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.d_model)

        if config.embedding_init_target or not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        self.encoder = GroundingDinoEncoder(config)
        self.decoder = GroundingDinoDecoder(config)

        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
            if (
                config.two_stage_bbox_embed_share
                and config.decoder_bbox_embed_share
                and self.decoder.bbox_embed is not None
            ):
                self.encoder_output_bbox_embed = self.decoder.bbox_embed
            else:
                self.encoder_output_bbox_embed = GroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                )

            self.encoder_output_class_embed = GroundingDinoContrastiveEmbedding(config)
        else:
            self.reference_points = nn.Embedding(config.num_queries, 4)

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

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def generate_encoder_output_proposals(self, enc_output, padding_mask, spatial_shapes):
        """Generate the encoder output proposals from encoded enc_output.

        Args:
            enc_output (`torch.Tensor[batch_size, sequence_length, hidden_size]`): Output of the encoder.
            padding_mask (`torch.Tensor[batch_size, sequence_length]`): Padding mask for `enc_output`.
            spatial_shapes (`torch.Tensor[num_feature_levels, 2]`): Spatial shapes of the feature maps.

        Returns:
            `tuple(torch.FloatTensor)`: A tuple of feature map and bbox prediction.
                - object_query (Tensor[batch_size, sequence_length, hidden_size]): Object query features. Later used to
                  directly predict a bounding box. (without the need of a decoder)
                - output_proposals (Tensor[batch_size, sequence_length, 4]): Normalized proposals, after an inverse
                  sigmoid.
        """
        batch_size = enc_output.shape[0]
        proposals = []
        current_position = 0
        for level, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, current_position : (current_position + height * width)]
            mask_flatten_ = mask_flatten_.view(batch_size, height, width, 1)
            valid_height = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_width = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = meshgrid(
                torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device),
                torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], 1).view(batch_size, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / scale
            width_heigth = torch.ones_like(grid) * 0.05 * (2.0**level)
            proposal = torch.cat((grid, width_heigth), -1).view(batch_size, -1, 4)
            proposals.append(proposal)
            current_position += height * width

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse sigmoid
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # assign each pixel as an object query
        object_query = enc_output
        object_query = object_query.masked_fill(padding_mask.unsqueeze(-1), float(0))
        object_query = object_query.masked_fill(~output_proposals_valid, float(0))
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return object_query, output_proposals

    @add_start_docstrings_to_model_forward(GROUNDING_DINO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GroundingDinoModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_mask: Optional[Tensor] = None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "a cat."

        >>> processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        >>> model = AutoModel.from_pretrained("IDEA-Research/grounding-dino-tiny")

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
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

        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        text_token_mask = attention_mask.bool()  # just to avoid renaming everywhere

        max_text_len = self.config.max_text_len
        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
            position_ids = position_ids[:, :max_text_len]
            input_ids = input_ids[:, :max_text_len]
            token_type_ids = token_type_ids[:, :max_text_len]
            text_token_mask = text_token_mask[:, :max_text_len]

        # Extract text features from text backbone
        text_outputs = self.text_backbone(
            input_ids, text_self_attention_masks, token_type_ids, position_ids, return_dict=return_dict
        )
        text_features = text_outputs.last_hidden_state if return_dict else text_outputs[0]
        text_features = self.text_projection(text_features)

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # which is a list of tuples
        vision_features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        feature_maps = []
        masks = []
        for level, (source, mask) in enumerate(vision_features):
            feature_maps.append(self.input_proj_vision[level](source))
            masks.append(mask)

        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(feature_maps):
            _len_sources = len(feature_maps)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj_vision[level](vision_features[-1][0])
                else:
                    source = self.input_proj_vision[level](feature_maps[-1])
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone.position_embedding(source, mask).to(source.dtype)
                feature_maps.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)

        # Create queries
        query_embeds = None
        if self.config.embedding_init_target or self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(zip(feature_maps, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = valid_ratios.float()

        # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                vision_features=source_flatten,
                vision_attention_mask=~mask_flatten,
                vision_position_embedding=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                text_features=text_features,
                text_attention_mask=~text_token_mask,
                text_position_embedding=None,
                text_self_attention_masks=~text_self_attention_masks,
                text_position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a GroundingDinoEncoderOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, GroundingDinoEncoderOutput):
            encoder_outputs = GroundingDinoEncoderOutput(
                last_hidden_state_vision=encoder_outputs[0],
                last_hidden_state_text=encoder_outputs[1],
                vision_hidden_states=encoder_outputs[2] if output_hidden_states else None,
                text_hidden_states=encoder_outputs[3] if output_hidden_states else None,
                attentions=encoder_outputs[-1] if output_attentions else None,
            )

        # Fifth, prepare decoder inputs
        enc_outputs_class = None
        enc_outputs_coord_logits = None
        if self.config.two_stage:
            object_query_embedding, output_proposals = self.generate_encoder_output_proposals(
                encoder_outputs[0], ~mask_flatten, spatial_shapes
            )

            # hack implementation as in two-stage Deformable DETR
            # apply a detection head to each pixel (A.4 in paper)
            # linear projection for bounding box binary classification (i.e. foreground and background)
            enc_outputs_class = self.encoder_output_class_embed(
                object_query_embedding, encoder_outputs[1], text_token_mask
            )
            # 3-layer FFN to predict bounding boxes coordinates (bbox regression branch)
            delta_bbox = self.encoder_output_bbox_embed(object_query_embedding)
            enc_outputs_coord_logits = delta_bbox + output_proposals

            # only keep top scoring `config.num_queries` proposals
            topk = self.config.num_queries
            topk_logits = enc_outputs_class.max(-1)[0]
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]
            topk_coords_logits = torch.gather(
                enc_outputs_coord_logits, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )

            topk_coords_logits = topk_coords_logits.detach()
            reference_points = topk_coords_logits.sigmoid()
            init_reference_points = reference_points
            if query_embeds is not None:
                target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            else:
                target = torch.gather(
                    object_query_embedding, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
                ).detach()
        else:
            target = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            reference_points = self.reference_points.weight.unsqueeze(0).repeat(batch_size, 1, 1).sigmoid()
            init_reference_points = reference_points

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            vision_encoder_hidden_states=encoder_outputs[0],
            vision_encoder_attention_mask=mask_flatten,
            text_encoder_hidden_states=encoder_outputs[1],
            text_encoder_attention_mask=~text_token_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            self_attn_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            enc_outputs = tuple(value for value in [enc_outputs_class, enc_outputs_coord_logits] if value is not None)
            tuple_outputs = (
                (decoder_outputs[0], init_reference_points) + decoder_outputs[1:] + encoder_outputs + enc_outputs
            )

            return tuple_outputs

        return GroundingDinoModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            init_reference_points=init_reference_points,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state_vision=encoder_outputs.last_hidden_state_vision,
            encoder_last_hidden_state_text=encoder_outputs.last_hidden_state_text,
            encoder_vision_hidden_states=encoder_outputs.vision_hidden_states,
            encoder_text_hidden_states=encoder_outputs.text_hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
        )


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class GroundingDinoMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@add_start_docstrings(
    """
    Grounding DINO Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top,
    for tasks such as COCO detection.
    """,
    GROUNDING_DINO_START_DOCSTRING,
)
class GroundingDinoForObjectDetection(GroundingDinoPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    # the bbox_embed in the decoder are all clones though
    _tied_weights_keys = [r"bbox_embed\.[1-9]\d*", r"model\.decoder\.bbox_embed\.[0-9]\d*"]

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.model = GroundingDinoModel(config)
        _class_embed = GroundingDinoContrastiveEmbedding(config)

        if config.decoder_bbox_embed_share:
            _bbox_embed = GroundingDinoMLPPredictionHead(
                input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
            )
            self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        else:
            for _ in range(config.decoder_layers):
                _bbox_embed = GroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                )
                self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        self.class_embed = nn.ModuleList([_class_embed for _ in range(config.decoder_layers)])
        # hack for box-refinement
        self.model.decoder.bbox_embed = self.bbox_embed
        # hack implementation for two-stage
        self.model.decoder.class_embed = self.class_embed

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(GROUNDING_DINO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GroundingDinoObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        pixel_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Union[GroundingDinoEncoderOutput, Tuple]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: List[Dict[str, Union[torch.LongTensor, torch.FloatTensor]]] = None,
    ):
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, GroundingDinoForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "a cat."

        >>> processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        >>> model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to COCO API
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = processor.image_processor.post_process_object_detection(
        ...     outputs, threshold=0.35, target_sizes=target_sizes
        ... )[0]
        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 1) for i in box.tolist()]
        ...     print(f"Detected {label.item()} with confidence " f"{round(score.item(), 2)} at location {box}")
        Detected 1 with confidence 0.45 at location [344.8, 23.2, 637.4, 373.8]
        Detected 1 with confidence 0.41 at location [11.9, 51.6, 316.6, 472.9]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # First, sent images through Grounding DINO base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        idx = 5 + (1 if output_attentions else 0) + (1 if output_hidden_states else 0)
        enc_text_hidden_state = outputs.encoder_last_hidden_state_text if return_dict else outputs[idx]
        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        init_reference_points = outputs.init_reference_points if return_dict else outputs[1]
        inter_references_points = outputs.intermediate_reference_points if return_dict else outputs[3]

        # class logits + predicted bounding boxes
        outputs_classes = []
        outputs_coords = []

        # hidden_states are of shape (batch_size, num_stages, height, width)
        # predict class and bounding box deltas for each stage
        num_levels = hidden_states.shape[1]
        for level in range(num_levels):
            if level == 0:
                reference = init_reference_points
            else:
                reference = inter_references_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)
            outputs_class = self.class_embed[level](
                vision_hidden_state=hidden_states[:, level],
                text_hidden_state=enc_text_hidden_state,
                text_token_mask=attention_mask.bool(),
            )
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])

            reference_coordinates = reference.shape[-1]
            if reference_coordinates == 4:
                outputs_coord_logits = delta_bbox + reference
            elif reference_coordinates == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 4 or 2, but got {reference.shape[-1]}")
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, self.device, pred_boxes, self.config, outputs_class, outputs_coord
            )

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            tuple_outputs = ((loss, loss_dict) + output) if loss is not None else output

            return tuple_outputs

        dict_outputs = GroundingDinoObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            last_hidden_state=outputs.last_hidden_state,
            auxiliary_outputs=auxiliary_outputs,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state_vision=outputs.encoder_last_hidden_state_vision,
            encoder_last_hidden_state_text=outputs.encoder_last_hidden_state_text,
            encoder_vision_hidden_states=outputs.encoder_vision_hidden_states,
            encoder_text_hidden_states=outputs.encoder_text_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=outputs.enc_outputs_class,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
        )

        return dict_outputs


__all__ = ["GroundingDinoForObjectDetection", "GroundingDinoModel", "GroundingDinoPreTrainedModel"]

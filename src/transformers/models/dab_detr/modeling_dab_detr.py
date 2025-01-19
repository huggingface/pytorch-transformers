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
"""PyTorch DAB-DETR model."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import load_backbone
from .configuration_dab_detr import DabDetrConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DabDetrConfig"
_CHECKPOINT_FOR_DOC = "IDEA-Research/dab_detr-base"


@dataclass
# Copied from transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrDecoderOutput with ConditionalDetr->DabDetr,Conditional DETR->DAB-DETR,2 (anchor points)->4 (anchor points)
class DabDetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the Conditional DETR decoder. This class adds one attribute to
    BaseModelOutputWithCrossAttentions, namely an optional stack of intermediate decoder activations, i.e. the output
    of each decoder layer, each of them gone through a layernorm. This is useful when training the model with auxiliary
    decoding losses.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
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
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
        reference_points (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, 2 (anchor points))`):
            Reference points (reference points of each layer of the decoder).
    """

    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    reference_points: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrModelOutput with ConditionalDetr->DabDetr,Conditional DETR->DAB-DETR,2 (anchor points)->4 (anchor points)
class DabDetrModelOutput(Seq2SeqModelOutput):
    """
    Base class for outputs of the Conditional DETR encoder-decoder model. This class adds one attribute to
    Seq2SeqModelOutput, namely an optional stack of intermediate decoder activations, i.e. the output of each decoder
    layer, each of them gone through a layernorm. This is useful when training the model with auxiliary decoding
    losses.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, sequence_length, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
        reference_points (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, 2 (anchor points))`):
            Reference points (reference points of each layer of the decoder).
    """

    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    reference_points: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.detr.modeling_detr.DetrObjectDetectionOutput with Detr->DabDetr
class DabDetrObjectDetectionOutput(ModelOutput):
    """
    Output type of [`DabDetrForObjectDetection`].

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
            possible padding). You can use [`~DabDetrImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the decoder at the output of each
            layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the encoder at the output of each
            layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the encoder, after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->DabDetr
class DabDetrFrozenBatchNorm2d(nn.Module):
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


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->DabDetr
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `DabDetrFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = DabDetrFrozenBatchNorm2d(module.num_features)

            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


# Modified from transformers.models.detr.modeling_detr.DetrConvEncoder with Detr->DabDetr
class DabDetrConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by DabDetrFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config: DabDetrConfig):
        super().__init__()

        self.config = config
        backbone = load_backbone(config)

        # replace batch norm by frozen batch norm
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.channels

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


# Copied from transformers.models.detr.modeling_detr.DetrConvModel with Detr->DabDetr
class DabDetrConvModel(nn.Module):
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


# Modified from transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrSinePositionEmbedding with ConditionalDetr->DabDetr
class DabDetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, config: DabDetrConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size / 2
        self.temperature_height = config.temperature_height
        self.temperature_width = config.temperature_width
        scale = config.sine_position_embedding_scale
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        # We use float32 to ensure reproducibility of the original implementation
        dim_tx = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        # Modifying dim_tx in place to avoid extra memory allocation -> dim_tx = self.temperature_width ** (2 * (dim_tx // 2) / self.embedding_dim)
        dim_tx //= 2
        dim_tx.mul_(2 / self.embedding_dim)
        dim_tx.copy_(self.temperature_width**dim_tx)
        pos_x = x_embed[:, :, :, None] / dim_tx

        # We use float32 to ensure reproducibility of the original implementation
        dim_ty = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        # Modifying dim_ty in place to avoid extra memory allocation -> dim_ty = self.temperature_height ** (2 * (dim_ty // 2) / self.embedding_dim)
        dim_ty //= 2
        dim_ty.mul_(2 / self.embedding_dim)
        dim_ty.copy_(self.temperature_height**dim_ty)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# function to generate sine positional embedding for 4d coordinates
def gen_sine_position_embeddings(pos_tensor, hidden_size=256):
    """
    This function computes position embeddings using sine and cosine functions from the input positional tensor,
    which has a shape of (batch_size, num_queries, 4).
    The last dimension of `pos_tensor` represents the following coordinates:
    - 0: x-coord
    - 1: y-coord
    - 2: width
    - 3: height

    The output shape is (batch_size, num_queries, 512), where final dim (hidden_size*2 = 512) is the total embedding dimension
    achieved by concatenating the sine and cosine values for each coordinate.
    """
    scale = 2 * math.pi
    dim = hidden_size // 2
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# Modified from transformers.models.detr.modeling_detr.DetrAttention
class DetrAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        config: DabDetrConfig,
        bias: bool = True,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.encoder_attention_heads
        self.attention_dropout = config.attention_dropout
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        batch_size, q_len, embed_dim = hidden_states.size()
        # add position embeddings to the hidden states before projecting to queries and keys
        if object_queries is not None:
            hidden_states_original = hidden_states
            hidden_states = hidden_states + object_queries

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states_original)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(batch_size, q_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


# Modified from transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrAttention with ConditionalDetr->DABDETR,Conditional DETR->DabDetr
class DabDetrAttention(nn.Module):
    """
    Cross-Attention used in DAB-DETR 'DAB-DETR for Fast Training Convergence' paper.

    The key q_proj, k_proj, v_proj are defined outside the attention. This attention allows the dim of q, k to be
    different to v.
    """

    def __init__(self, config: DabDetrConfig, bias: bool = True, is_cross: bool = False):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size * 2 if is_cross else config.hidden_size
        self.output_dim = config.hidden_size
        self.attention_heads = config.decoder_attention_heads
        self.attention_dropout = config.attention_dropout
        self.attention_head_dim = self.embed_dim // self.attention_heads
        if self.attention_head_dim * self.attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `attention_heads`:"
                f" {self.attention_heads})."
            )
        # head dimension of values
        self.values_head_dim = self.output_dim // self.attention_heads
        if self.values_head_dim * self.attention_heads != self.output_dim:
            raise ValueError(
                f"output_dim must be divisible by attention_heads (got `output_dim`: {self.output_dim} and `attention_heads`: {self.attention_heads})."
            )
        self.scaling = self.attention_head_dim**-0.5
        self.output_proj = nn.Linear(self.output_dim, self.output_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        # scaling query and refactor key-, value states
        query_states = hidden_states * self.scaling
        query_states = query_states.view(batch_size, -1, self.attention_heads, self.attention_head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.attention_heads, self.attention_head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.attention_heads, self.values_head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_probs = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)

        if attn_output.size() != (batch_size, self.attention_heads, q_len, self.values_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.attention_heads, q_len, self.values_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(batch_size, q_len, self.output_dim)
        attn_output = self.output_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class DabDetrDecoderLayerSelfAttention(nn.Module):
    def __init__(self, config: DabDetrConfig):
        super().__init__()
        self.dropout = config.dropout
        self.self_attn_query_content_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn_query_pos_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn_key_content_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn_key_pos_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn_value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_attn = DabDetrAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_position_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        residual = hidden_states
        query_content = self.self_attn_query_content_proj(hidden_states)
        query_pos = self.self_attn_query_pos_proj(query_position_embeddings)
        key_content = self.self_attn_key_content_proj(hidden_states)
        key_pos = self.self_attn_key_pos_proj(query_position_embeddings)
        value = self.self_attn_value_proj(hidden_states)

        query = query_content + query_pos
        key = key_content + key_pos

        hidden_states, attn_weights = self.self_attn(
            hidden_states=query,
            attention_mask=attention_mask,
            key_states=key,
            value_states=value,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        return hidden_states, attn_weights


class DabDetrDecoderLayerCrossAttention(nn.Module):
    def __init__(self, config: DabDetrConfig, is_first: bool = False):
        super().__init__()
        hidden_size = config.hidden_size
        self.cross_attn_query_content_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_query_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_key_content_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_key_pos_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_value_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_query_pos_sine_proj = nn.Linear(hidden_size, hidden_size)
        self.decoder_attention_heads = config.decoder_attention_heads
        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = DabDetrAttention(config, is_cross=True)

        self.keep_query_pos = config.keep_query_pos

        if not self.keep_query_pos and not is_first:
            self.cross_attn_query_pos_proj = None

        self.is_first = is_first
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        query_sine_embed: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        query_content = self.cross_attn_query_content_proj(hidden_states)
        key_content = self.cross_attn_key_content_proj(encoder_hidden_states)
        value = self.cross_attn_value_proj(encoder_hidden_states)

        batch_size, num_queries, n_model = query_content.shape
        _, height_width, _ = key_content.shape

        key_pos = self.cross_attn_key_pos_proj(object_queries)

        # For the first decoder layer, we add the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if self.is_first or self.keep_query_pos:
            query_pos = self.cross_attn_query_pos_proj(query_position_embeddings)
            query = query_content + query_pos
            key = key_content + key_pos
        else:
            query = query_content
            key = key_content

        query = query.view(
            batch_size, num_queries, self.decoder_attention_heads, n_model // self.decoder_attention_heads
        )
        query_sine_embed = self.cross_attn_query_pos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(
            batch_size, num_queries, self.decoder_attention_heads, n_model // self.decoder_attention_heads
        )
        query = torch.cat([query, query_sine_embed], dim=3).view(batch_size, num_queries, n_model * 2)
        key = key.view(batch_size, height_width, self.decoder_attention_heads, n_model // self.decoder_attention_heads)
        key_pos = key_pos.view(
            batch_size, height_width, self.decoder_attention_heads, n_model // self.decoder_attention_heads
        )
        key = torch.cat([key, key_pos], dim=3).view(batch_size, height_width, n_model * 2)

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights = self.cross_attn(
                hidden_states=query,
                attention_mask=encoder_attention_mask,
                key_states=key,
                value_states=value,
                output_attentions=output_attentions,
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.cross_attn_layer_norm(hidden_states)

        return hidden_states, cross_attn_weights


class DabDetrDecoderLayerFFN(nn.Module):
    def __init__(self, config: DabDetrConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, hidden_size)
        self.activation_fn = ACT2FN[config.activation_function]
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout
        self.keep_query_pos = config.keep_query_pos

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


# Modified from transformers.models.detr.modeling_detr.DetrEncoderLayer with DetrEncoderLayer->DabDetrEncoderLayer,DetrConfig->DabDetrConfig
class DabDetrEncoderLayer(nn.Module):
    def __init__(self, config: DabDetrConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DetrAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(self.hidden_size, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.hidden_size)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        object_queries: torch.Tensor,
        output_attentions: Optional[bool] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`torch.FloatTensor`, *optional*):
                Object queries (also called content embeddings), to be added to the hidden states.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Modified from transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrDecoderLayer with ConditionalDetr->DabDetr
class DabDetrDecoderLayer(nn.Module):
    def __init__(self, config: DabDetrConfig, is_first: bool = False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(DabDetrDecoderLayerSelfAttention(config))
        self.layer.append(DabDetrDecoderLayerCrossAttention(config, is_first))
        self.layer.append(DabDetrDecoderLayerFFN(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        query_sine_embed: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            object_queries (`torch.FloatTensor`, *optional*):
                object_queries that are added to the queries and keys
            in the cross-attention layer.
            query_position_embeddings (`torch.FloatTensor`, *optional*):
                object_queries that are added to the queries and keys
            in the self-attention layer.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.

        """
        hidden_states, self_attn_weights = self.layer[0](
            hidden_states=hidden_states,
            query_position_embeddings=query_position_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states, cross_attn_weights = self.layer[1](
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_position_embeddings=query_position_embeddings,
            object_queries=object_queries,
            encoder_attention_mask=encoder_attention_mask,
            query_sine_embed=query_sine_embed,
            output_attentions=output_attentions,
        )

        hidden_states = self.layer[2](hidden_states=hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


# Modified from transformers.models.detr.modeling_detr.DetrMLPPredictionHead with DetrMLPPredictionHead->DabDetrMLP
class DabDetrMLP(nn.Module):
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

    def forward(self, input_tensor):
        for i, layer in enumerate(self.layers):
            input_tensor = nn.functional.relu(layer(input_tensor)) if i < self.num_layers - 1 else layer(input_tensor)
        return input_tensor


# Modified from transformers.models.detr.modeling_detr.DetrPreTrainedModel with Detr->DabDetr
class DabDetrPreTrainedModel(PreTrainedModel):
    config_class = DabDetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"DabDetrConvEncoder", r"DabDetrEncoderLayer", r"DabDetrDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        xavier_std = self.config.init_xavier_std

        if isinstance(module, DabDetrMHAttentionMap):
            nn.init.zeros_(module.k_linear.bias)
            nn.init.zeros_(module.q_linear.bias)
            nn.init.xavier_uniform_(module.k_linear.weight, gain=xavier_std)
            nn.init.xavier_uniform_(module.q_linear.weight, gain=xavier_std)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, DabDetrForObjectDetection):
            nn.init.constant_(module.bbox_predictor.layers[-1].weight.data, 0)
            nn.init.constant_(module.bbox_predictor.layers[-1].bias.data, 0)

            # init prior_prob setting for focal loss
            prior_prob = self.config.initializer_bias_prior_prob or 1 / (self.config.num_labels + 1)
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            module.class_embed.bias.data = torch.ones(self.config.num_labels) * bias_value


DAB_DETR_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DabDetrConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DAB_DETR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`DetrImageProcessor.__call__`]
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
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Modified from transformers.models.detr.modeling_detr.DetrEncoder with Detr->DabDetr,DETR->ConditionalDETR
class DabDetrEncoder(DabDetrPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`DabDetrEncoderLayer`].

    The encoder updates the flattened feature map through multiple self-attention layers.

    Small tweak for DAB-DETR:

    - object_queries are added to the forward pass.

    Args:
        config: DabDetrConfig
    """

    def __init__(self, config: DabDetrConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.query_scale = DabDetrMLP(config.hidden_size, config.hidden_size, config.hidden_size, 2)
        self.layers = nn.ModuleList([DabDetrEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.norm = nn.LayerNorm(config.hidden_size) if config.normalize_before else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        object_queries,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(sequence_length, batch_size, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.

            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:

                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).

                [What are attention masks?](../glossary#attention-mask)

            object_queries (`torch.FloatTensor` of shape `(sequence_length, batch_size, hidden_size)`):
                Object queries that are added to the queries in each self-attention layer.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds

        # expand attention_mask
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # pos scaler
            pos_scales = self.query_scale(hidden_states)
            scaled_object_queries = object_queries * pos_scales
            # we add object_queries * pos_scaler as extra input to the encoder_layer
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                object_queries=scaled_object_queries,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self.norm:
            hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# Modified from transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrDecoder with ConditionalDetr->DabDetr,Conditional DETR->DAB-DETR
class DabDetrDecoder(DabDetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DabDetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for DAB-DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: DabDetrConfig
    """

    def __init__(self, config: DabDetrConfig):
        super().__init__(config)
        self.config = config
        self.dropout = config.dropout
        self.num_layers = config.decoder_layers
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [DabDetrDecoderLayer(config, is_first=(layer_id == 0)) for layer_id in range(config.decoder_layers)]
        )
        # in DAB-DETR, the decoder uses layernorm after the last decoder layer output
        self.hidden_size = config.hidden_size
        self.layernorm = nn.LayerNorm(self.hidden_size)

        # Default cond-elewise
        self.query_scale = DabDetrMLP(self.hidden_size, self.hidden_size, self.hidden_size, 2)

        self.ref_point_head = DabDetrMLP(
            config.query_dim // 2 * self.hidden_size, self.hidden_size, self.hidden_size, 2
        )

        self.bbox_embed = None

        # Default decoder_modulate_hw_attn is True
        self.ref_anchor_head = DabDetrMLP(self.hidden_size, self.hidden_size, 2, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds,
        encoder_hidden_states,
        memory_key_padding_mask,
        object_queries,
        query_position_embeddings,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(sequence_length, batch_size, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(encoder_sequence_length, batch_size, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            memory_key_padding_mask (`torch.Tensor.bool` of shape `(batch_size, sequence_length)`):
                The memory_key_padding_mask indicates which positions in the memory (encoder outputs) should be ignored during the attention computation,
                ensuring padding tokens do not influence the attention mechanism.
            object_queries (`torch.FloatTensor` of shape `(sequence_length, batch_size, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each cross-attention layer.
            query_position_embeddings (`torch.FloatTensor` of shape `(num_queries, batch_size, number_of_anchor_points)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            input_shape = inputs_embeds.size()[:-1]

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        intermediate = []
        reference_points = query_position_embeddings.sigmoid()
        ref_points = [reference_points]

        # expand encoder attention mask
        if encoder_hidden_states is not None and memory_key_padding_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, target_seq_len, source_seq_len]
            memory_key_padding_mask = _prepare_4d_attention_mask(
                memory_key_padding_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        for layer_id, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            obj_center = reference_points[..., : self.config.query_dim]
            query_sine_embed = gen_sine_position_embeddings(obj_center, self.hidden_size)
            query_pos = self.ref_point_head(query_sine_embed)

            # For the first decoder layer, we do not apply transformation over p_s
            pos_transformation = 1 if layer_id == 0 else self.query_scale(hidden_states)

            # apply transformation
            query_sine_embed = query_sine_embed[..., : self.hidden_size] * pos_transformation

            # modulated HW attentions
            refHW_cond = self.ref_anchor_head(hidden_states).sigmoid()  # nq, bs, 2
            query_sine_embed[..., self.hidden_size // 2 :] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
            query_sine_embed[..., : self.hidden_size // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    None,
                    object_queries,
                    query_pos,
                    query_sine_embed,
                    encoder_hidden_states,
                    memory_key_padding_mask,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    object_queries=object_queries,
                    query_position_embeddings=query_pos,
                    query_sine_embed=query_sine_embed,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=memory_key_padding_mask,
                    output_attentions=output_attentions,
                )

            # iter update
            hidden_states = layer_outputs[0]

            if self.bbox_embed is not None:
                new_reference_points = self.bbox_embed(hidden_states)

                new_reference_points[..., : self.config.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points[..., : self.config.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            intermediate.append(self.layernorm(hidden_states))

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        if self.layernorm is not None:
            hidden_states = self.layernorm(hidden_states)
            intermediate.pop()
            intermediate.append(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output_intermediate_hidden_states = torch.stack(intermediate)
        output_reference_points = torch.stack(ref_points)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                    output_intermediate_hidden_states,
                    output_reference_points,
                ]
                if v is not None
            )
        return DabDetrDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            intermediate_hidden_states=output_intermediate_hidden_states,
            reference_points=output_reference_points,
        )


@add_start_docstrings(
    """
    The bare DAB-DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states, intermediate hidden states, reference points, output coordinates without any specific head on top.
    """,
    DAB_DETR_START_DOCSTRING,
)
class DabDetrModel(DabDetrPreTrainedModel):
    def __init__(self, config: DabDetrConfig):
        super().__init__(config)

        self.auxiliary_loss = config.auxiliary_loss

        # Create backbone + positional encoding
        self.backbone = DabDetrConvEncoder(config)
        object_queries = DabDetrSinePositionEmbedding(config)

        self.query_refpoint_embeddings = nn.Embedding(config.num_queries, config.query_dim)
        self.random_refpoints_xy = config.random_refpoints_xy
        if self.random_refpoints_xy:
            self.query_refpoint_embeddings.weight.data[:, :2].uniform_(0, 1)
            self.query_refpoint_embeddings.weight.data[:, :2] = inverse_sigmoid(
                self.query_refpoint_embeddings.weight.data[:, :2]
            )
            self.query_refpoint_embeddings.weight.data[:, :2].requires_grad = False

        # Create projection layer
        self.input_projection = nn.Conv2d(
            self.backbone.intermediate_channel_sizes[-1], config.hidden_size, kernel_size=1
        )
        self.backbone = DabDetrConvModel(self.backbone, object_queries)

        self.encoder = DabDetrEncoder(config)
        self.decoder = DabDetrDecoder(config)

        # decoder related variables
        self.hidden_size = config.hidden_size
        self.num_queries = config.num_queries

        self.num_patterns = config.num_patterns
        if not isinstance(self.num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(self.num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.hidden_size)

        self.aux_loss = config.auxiliary_loss

        # Initialize weights and apply final processing
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

    @add_start_docstrings_to_model_forward(DAB_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DabDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DabDetrModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("IDEA-Research/dab_detr-base")
        >>> model = AutoModel.from_pretrained("IDEA-Research/dab_detr-base")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # the last hidden states are the final query embeddings of the Transformer decoder
        >>> # these are of shape (batch_size, num_queries, hidden_size)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 300, 256]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)

        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # pixel_values should be of shape (batch_size, num_channels, height, width)
        # pixel_mask should be of shape (batch_size, height, width)
        features, object_queries_list = self.backbone(pixel_values, pixel_mask)

        # get final feature map and downsampled mask
        feature_map, mask = features[-1]

        if mask is None:
            raise ValueError("Backbone does not return downsampled pixel mask")

        flattened_mask = mask.flatten(1)

        # Second, apply 1x1 convolution to reduce the channel dimension to hidden_size (256 by default)
        projected_feature_map = self.input_projection(feature_map)

        # Third, flatten the feature map + object_queries of shape NxCxHxW to HWxNxC, and permute it to NxHWxC
        # In other words, turn their shape into ( sequence_length, batch_size, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)
        reference_position_embeddings = self.query_refpoint_embeddings.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # Fourth, sent flattened_features + flattened_mask + object_queries through encoder
        # flattened_features is a Tensor of shape (heigth*width, batch_size, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                object_queries=object_queries,
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

        # Fifth, sent query embeddings + object_queries through the decoder (which is conditioned on the encoder output)
        num_queries = reference_position_embeddings.shape[1]
        if self.num_patterns == 0:
            queries = torch.zeros(batch_size, num_queries, self.hidden_size, device=device)
        else:
            queries = (
                self.patterns.weight[:, None, None, :]
                .repeat(1, self.num_queries, batch_size, 1)
                .flatten(0, 1)
                .permute(1, 0, 2)
            )  # bs, n_q*n_pat, hidden_size
            reference_position_embeddings = reference_position_embeddings.repeat(
                1, self.num_patterns, 1
            )  # bs, n_q*n_pat,  hidden_size

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            query_position_embeddings=reference_position_embeddings,
            object_queries=object_queries,
            encoder_hidden_states=encoder_outputs[0],
            memory_key_padding_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            # last_hidden_state
            output = (decoder_outputs[0],)
            reference_points = decoder_outputs[-1]
            intermediate_hidden_states = decoder_outputs[-2]

            # it has to follow the order of DABDETRModelOutput that is based on ModelOutput
            # If we only use one of the variables then the indexing will change.
            # E.g: if we return everything then 'decoder_attentions' is decoder_outputs[2], if we only use output_attentions then its decoder_outputs[1]
            if output_hidden_states and output_attentions:
                output += (
                    decoder_outputs[1],
                    decoder_outputs[2],
                    decoder_outputs[3],
                    encoder_outputs[0],
                    encoder_outputs[1],
                    encoder_outputs[2],
                )
            elif output_hidden_states:
                # decoder_hidden_states, encoder_last_hidden_state, encoder_hidden_states
                output += (
                    decoder_outputs[1],
                    encoder_outputs[0],
                    encoder_outputs[1],
                )
            elif output_attentions:
                # decoder_self_attention, decoder_cross_attention, encoder_attentions
                output += (
                    decoder_outputs[1],
                    decoder_outputs[2],
                    encoder_outputs[1],
                )

            output += (intermediate_hidden_states, reference_points)

            return output

        reference_points = decoder_outputs.reference_points
        intermediate_hidden_states = decoder_outputs.intermediate_hidden_states

        return DabDetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states if output_hidden_states else None,
            decoder_attentions=decoder_outputs.attentions if output_attentions else None,
            cross_attentions=decoder_outputs.cross_attentions if output_attentions else None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state if output_hidden_states else None,
            encoder_hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            encoder_attentions=encoder_outputs.attentions if output_attentions else None,
            intermediate_hidden_states=intermediate_hidden_states,
            reference_points=reference_points,
        )


# Copied from transformers.models.detr.modeling_detr.DetrMHAttentionMap with Detr->DabDetr
class DabDetrMHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, std=None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = nn.functional.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        queries_per_head = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        keys_per_head = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", queries_per_head * self.normalize_fact, keys_per_head)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), torch.finfo(weights.dtype).min)
        weights = nn.functional.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


@add_start_docstrings(
    """
    DAB_DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    DAB_DETR_START_DOCSTRING,
)
class DabDetrForObjectDetection(DabDetrPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    _tied_weights_keys = [
        r"bbox_predictor\.layers\.\d+\.(weight|bias)",
        r"model\.decoder\.bbox_embed\.layers\.\d+\.(weight|bias)",
    ]

    def __init__(self, config: DabDetrConfig):
        super().__init__(config)

        self.config = config
        self.auxiliary_loss = config.auxiliary_loss
        self.query_dim = config.query_dim
        # DAB-DETR encoder-decoder model
        self.model = DabDetrModel(config)

        _bbox_embed = DabDetrMLP(config.hidden_size, config.hidden_size, 4, 3)
        # Object detection heads
        self.class_embed = nn.Linear(config.hidden_size, config.num_labels)

        # Default bbox_embed_diff_each_layer is False
        self.bbox_predictor = _bbox_embed

        # Default iter_update is True
        self.model.decoder.bbox_embed = self.bbox_predictor

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/Atten4Vis/conditionalDETR/blob/master/models/dab_detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(DAB_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DabDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DabDetrObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("IDEA-Research/dab_detr-base")
        >>> model = AutoModelForObjectDetection.from_pretrained("IDEA-Research/dab_detr-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        >>>     outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([(image.height, image.width)])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected remote with confidence 0.833 at location [38.31, 72.1, 177.63, 118.45]
        Detected cat with confidence 0.831 at location [9.2, 51.38, 321.13, 469.0]
        Detected cat with confidence 0.804 at location [340.3, 16.85, 642.93, 370.95]
        Detected remote with confidence 0.683 at location [334.48, 73.49, 366.37, 190.01]
        Detected couch with confidence 0.535 at location [0.52, 1.19, 640.35, 475.1]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # First, sent images through DAB_DETR base model to obtain encoder + decoder outputs
        model_outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        reference_points = model_outputs.reference_points if return_dict else model_outputs[-1]
        intermediate_hidden_states = model_outputs.intermediate_hidden_states if return_dict else model_outputs[-2]

        # class logits + predicted bounding boxes
        logits = self.class_embed(intermediate_hidden_states[-1])

        reference_before_sigmoid = inverse_sigmoid(reference_points)
        tmp = self.bbox_predictor(intermediate_hidden_states)
        tmp[..., : self.query_dim] += reference_before_sigmoid
        outputs_coord = tmp.sigmoid()

        pred_boxes = outputs_coord[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class = None
            if self.config.auxiliary_loss:
                outputs_class = self.class_embed(intermediate_hidden_states)
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits, labels, self.device, pred_boxes, self.config, outputs_class, outputs_coord
            )

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + model_outputs
            else:
                output = (logits, pred_boxes) + model_outputs
            # Since DabDetrObjectDetectionOutput doesn't have reference points + intermedieate_hidden_states we cut down.
            return ((loss, loss_dict) + output) if loss is not None else output[:-2]

        return DabDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=model_outputs.last_hidden_state,
            decoder_hidden_states=model_outputs.decoder_hidden_states if output_hidden_states else None,
            decoder_attentions=model_outputs.decoder_attentions if output_attentions else None,
            cross_attentions=model_outputs.cross_attentions if output_attentions else None,
            encoder_last_hidden_state=model_outputs.encoder_last_hidden_state if output_hidden_states else None,
            encoder_hidden_states=model_outputs.encoder_hidden_states if output_hidden_states else None,
            encoder_attentions=model_outputs.encoder_attentions if output_attentions else None,
        )


__all__ = [
    "DabDetrForObjectDetection",
    "DabDetrModel",
    "DabDetrPreTrainedModel",
]

# coding=utf-8
# Copyright 2024 Microsoft Research and HuggingFace Inc. team.
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
"""PyTorch UDOP model."""

import collections
import logging
import math
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from transformers import UdopConfig
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    replace_return_docstrings,
)


logger = logging.getLogger(__name__)


_CONFIG_FOR_DOC = "UdopConfig"


UDOP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Args:
        config ([`UdopConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


UDOP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. UDOP is a model with relative position embeddings so
            you should be able to pad the inputs on both the right and the left. Indices can be obtained using
            [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for detail.
            [What are input IDs?](../glossary#input-ids)

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)

        bbox (`torch.LongTensor` of shape `({0}, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Batch of document images. Each image is divided into patches of shape `(num_channels, config.patch_size,
            config.patch_size)` and the total number of patches (=`patch_sequence_length`) equals to `((height /
            config.patch_size) * (width / config.patch_size))`.

        visual_bbox (`torch.LongTensor` of shape `(batch_size, patch_sequence_length, 4)`, *optional*):
            Bounding boxes of each patch in the image. If not provided, bounding boxes are created in the model.

        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using
            [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.
            [What are decoder input IDs?](../glossary#decoder-input-ids) T5 uses the `pad_token_id` as the starting
            token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last
            `decoder_input_ids` have to be input (see `past_key_values`). To know more on how to prepare
            `decoder_input_ids` for pretraining take a look at [T5 Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix. If
            `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of
            `inputs_embeds`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. It is used to update the
            cache in the correct position and to infer the complete sequence length.
"""


UDOP_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        bbox (`torch.LongTensor` of shape `({0}, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.

            Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
            token. See `pixel_values` for `patch_sequence_length`.

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Batch of document images. Each image is divided into patches of shape `(num_channels, config.patch_size,
            config.patch_size)` and the total number of patches (=`patch_sequence_length`) equals to `((height /
            config.patch_size) * (width / config.patch_size))`.

        visual_bbox (`torch.LongTensor` of shape `(batch_size, patch_sequence_length, 4)`, *optional*):
            Bounding boxes of each patch in the image. If not provided, bounding boxes are created in the model.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@dataclass
class BaseModelOutputWithAttentionMask(ModelOutput):
    """
    Class for the model's outputs that may also contain a past key/values (to speed up sequential decoding). Includes
    an additional attention mask.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model. If `past_key_values` is used only
            the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or
        when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`. Contains pre-computed hidden-states (key and values in the
            self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks)
            that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and
        `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def get_visual_bbox(image_size=224, patch_size=16):
    image_feature_pool_shape = [image_size // patch_size, image_size // patch_size]
    visual_bbox_x = torch.arange(0, 1.0 * (image_feature_pool_shape[1] + 1), 1.0)
    visual_bbox_x /= image_feature_pool_shape[1]

    visual_bbox_y = torch.arange(0, 1.0 * (image_feature_pool_shape[0] + 1), 1.0)
    visual_bbox_y /= image_feature_pool_shape[0]

    visual_bbox_input = torch.stack(
        [
            visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1),
            visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
            visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1),
            visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
        ],
        dim=-1,
    )

    visual_bbox_input = visual_bbox_input.view(-1, 4)

    return visual_bbox_input


def pad_sequence(seq, target_len, pad_value=0):
    if isinstance(seq, torch.Tensor):
        n = seq.shape[0]
    else:
        n = len(seq)
        seq = torch.tensor(seq)
    m = target_len - n
    if m > 0:
        ret = torch.stack([pad_value] * m).to(seq)
        seq = torch.cat([seq, ret], dim=0)
    return seq[:target_len]


def combine_image_text_embeddings(
    image_embeddings,
    inputs_embeds,
    bbox,
    visual_bbox,
    attention_mask=None,
    num_patches=14,
    max_len=0,
    image_size=224,
    patch_size=16,
):
    """
    Combine the image and text embeddings for the input to the encoder/decoder of UDOP.

    First, the image embeddings are created by checking for each visual patch if it is inside the bounding box of a
    token. If it is, the visual patch is combined with the token embedding. Then, the visual bounding boxes are combined
    with the text bounding boxes. Finally, the visual bounding boxes are combined with the text attention mask.
    """

    sequence_length = num_patches
    ocr_points_x = torch.clip(
        torch.floor((bbox[:, :, 0] + bbox[:, :, 2]) / 2.0 * sequence_length).long(), 0, sequence_length - 1
    )
    ocr_points_y = (
        torch.clip(torch.floor((bbox[:, :, 1] + bbox[:, :, 3]) / 2.0 * sequence_length).long(), 0, sequence_length - 1)
        * sequence_length
    )
    ocr_points = ocr_points_x + ocr_points_y
    # make sure bounding boxes are of type float to calculate means
    bbox = bbox.to(torch.float64)
    target_seg = (bbox.mean(-1) == 0.0) | (bbox.mean(-1) == 1.0)
    repeated_vision_embeds = torch.gather(
        image_embeddings, 1, ocr_points.unsqueeze(-1).repeat(1, 1, image_embeddings.size(-1))
    )
    repeated_vision_embeds[target_seg] = 0.0
    inputs_embeds += repeated_vision_embeds

    patch_inds = torch.full_like(image_embeddings[:, :, 0], True).bool()
    ind = torch.cat(
        [
            torch.arange(len(ocr_points))[:, None].repeat(1, ocr_points.size(-1))[:, :, None].to(ocr_points),
            ocr_points[:, :, None],
        ],
        dim=-1,
    )
    ind = ind.flatten(0, 1)
    rows, cols = zip(*ind)
    patch_inds[rows, cols] = False

    input_vision_patches = [image_embeddings[i][patch_inds[i]] for i in range(len(patch_inds))]

    if visual_bbox is None:
        visual_bbox = get_visual_bbox(image_size=image_size, patch_size=patch_size)
        visual_bbox = visual_bbox.unsqueeze(0).repeat(image_embeddings.size(0), 1, 1)
        visual_bbox = visual_bbox.to(image_embeddings.device)

    visual_bbox = [visual_bbox[i][patch_inds[i]] for i in range(len(patch_inds))]
    if attention_mask is not None:
        visual_attention_mask = [torch.tensor([1] * len(item)).to(attention_mask) for item in visual_bbox]

    if max_len == 0:
        max_len = image_embeddings.size(1)
    else:
        max_len = max_len - inputs_embeds.size(1)
    inputs_vision_patches = torch.stack(
        [pad_sequence(item, max_len, torch.zeros_like(image_embeddings[0, 0])) for item in input_vision_patches]
    )
    visual_bbox = torch.stack([pad_sequence(item, max_len, torch.zeros_like(bbox[0, 0])) for item in visual_bbox])
    if attention_mask is not None:
        visual_attention_mask = torch.stack(
            [pad_sequence(item, max_len, torch.zeros_like(attention_mask[0, 0])) for item in visual_attention_mask]
        )

    inputs_embeds = torch.cat([inputs_embeds, inputs_vision_patches], 1)
    bbox = torch.cat([bbox, visual_bbox], 1)
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, visual_attention_mask], 1)
    return inputs_embeds, bbox, attention_mask


class UdopPatchEmbeddings(nn.Module):
    """2D Image to Patch Embeddings"""

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.proj = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model"
                f" ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.proj(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings


class UdopPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models. Based on `T5PreTrainedModel`.
    """

    config_class = UdopConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _supports_cache_class = True
    _supports_static_cache = False
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, UdopLayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Conv2d):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(module.weight.data.to(torch.float32), mean=0.0, std=factor).to(
                module.weight.dtype
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RelativePositionBiasBase):
            factor = self.config.initializer_factor
            d_model = self.config.d_model
            module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
        elif isinstance(module, UdopModel):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, UdopForConditionalGeneration):
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, UdopDenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, UdopDenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, UdopAttention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    # Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetPreTrainedModel._shift_right with ProphetNet->Udop
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In Udop it is usually set to the"
            " pad_token_id. See Udop docs for more information"
        )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->Udop
class UdopLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the Udop style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # Udop uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->Udop
class UdopDenseActDense(nn.Module):
    def __init__(self, config: UdopConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5->Udop
class UdopDenseGatedActDense(nn.Module):
    def __init__(self, config: UdopConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5LayerFF with T5->Udop
class UdopLayerFF(nn.Module):
    def __init__(self, config: UdopConfig):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = UdopDenseGatedActDense(config)
        else:
            self.DenseReluDense = UdopDenseActDense(config)

        self.layer_norm = UdopLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5Attention with T5->Udop
class UdopAttention(nn.Module):
    def __init__(
        self,
        config: UdopConfig,
        has_relative_attention_bias=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, past_key_value, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5->Udop
class UdopLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None):
        super().__init__()
        self.SelfAttention = UdopAttention(
            config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx
        )
        self.layer_norm = UdopLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5->Udop
class UdopLayerCrossAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.EncDecAttention = UdopAttention(config, has_relative_attention_bias=False, layer_idx=layer_idx)
        self.layer_norm = UdopLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        cache_position=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5Block with T5->Udop
class UdopBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            UdopLayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx
            )
        )
        if self.is_decoder:
            self.layer.append(UdopLayerCrossAttention(config, layer_idx=layer_idx))

        self.layer.append(UdopLayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        cache_position=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states, past_key_value = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                query_length=cache_position[-1] + 1,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states, past_key_value = cross_attention_outputs[:2]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (past_key_value,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, past_key_value, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class UdopCellEmbeddings(nn.Module):
    def __init__(self, max_2d_position_embeddings=501, hidden_size=1024):
        super(UdopCellEmbeddings, self).__init__()
        self.max_2d_position_embeddings = max_2d_position_embeddings

        self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)

    def forward(self, bbox):
        bbox = torch.clip(bbox, 0.0, 1.0)
        bbox = (bbox * (self.max_2d_position_embeddings - 1)).long()
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

        embeddings = (
            left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
        )

        return embeddings


# get function for bucket computation
# protected member access seems to be lesser evil than copy paste whole function
get_relative_position_bucket = UdopAttention._relative_position_bucket
AUGMENTATION_RANGE = (0.80, 1.25)


class RelativePositionBiasBase(nn.Module, ABC):
    """
    Base class of relative biases.

    Args:
        num_heads (`int`):
            Number of attention heads in the model, it will create embeddings of size `num_heads`, which will be added to the scores of each token pair.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            Pair token metric (distance in the sequence, distance in pixels etc.) will be bucketed, parameter is defining number of such
            buckets.
        bidirectional (`bool`, *optional*, defaults to `True`):
            Whether the distance should be bidirectional for a pair of tokens. If `False`, then distance(tok1, tok2) == distance(tok2, tok1).
        scaling_factor (`int`, *optional*, defaults to 1):
            Defining factor which will be used to scale relative distance.
        max_distance (`int`, *optional*, defaults to 128):
            All distances above this value will end up in the one/same bucket.
        augmentation (`bool`, *optional*, defaults to `False`):
            Whether to multiply relative distances by a random scalar.
        expand (`bool`, *optional*, defaults to `False`):
            Whether to expand an existing pretrained model with subsequent additions of prefix_bucket.
    """

    def __init__(
        self,
        num_heads=None,
        relative_attention_num_buckets=32,
        bidirectional=True,
        scaling_factor=1,
        max_distance=128,
        level="tokens",
        augmentation=False,
        prefix_bucket=False,
        expand=False,
    ):
        super(RelativePositionBiasBase, self).__init__()
        self.prefix_bucket = prefix_bucket
        self.augmentation = augmentation
        self.level = level
        self.max_distance = max_distance
        self.scaling_factor = scaling_factor
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        self.expand = expand
        self.relative_attention_num_buckets = relative_attention_num_buckets
        extra_head = 2 if prefix_bucket and not self.expand else 0
        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets + extra_head, self.num_heads)

    @abstractmethod
    def prepare_input(
        self,
        attention_mask: Optional[Tensor] = None,
        bbox: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        pass

    def get_bucket(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        relative_position = self.prepare_input(attention_mask, bbox)
        rp_bucket: Tensor = get_relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.max_distance,
        )
        return rp_bucket

    def get_relative_position(self, positions):
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]
        relative_position = memory_position - context_position
        if self.augmentation and self.training:
            relative_position *= random.uniform(*AUGMENTATION_RANGE)
        relative_position *= self.scaling_factor

        return relative_position.to(torch.long)

    def forward(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        # re-using pretrained model with subsequent addition of prefix_bucket
        if self.expand and self.prefix_bucket:
            new_bias = nn.Embedding(self.relative_attention_num_buckets + 2, self.num_heads)
            new_bias.weight.data[: self.relative_attention_num_buckets] = self.relative_attention_bias.weight.data
            new_bias.weight.data[self.relative_attention_num_buckets :] = 0.1
            self.relative_attention_bias = new_bias
            self.expand = False

        rp_bucket = self.get_bucket(attention_mask, bbox)

        if self.prefix_bucket:
            if rp_bucket.size(0) == 1 and attention_mask.size(0) > 1:
                rp_bucket = rp_bucket.repeat(attention_mask.size(0), 1, 1)
            # based on assumption that prefix bboxes are negative
            is_prefix = bbox[:, :, 1] < 0
            num_prefix = is_prefix.sum(-1)
            for idx, num_prefix_row in enumerate(num_prefix.cpu().numpy()):
                rp_bucket[idx, :num_prefix_row, num_prefix_row:] = self.relative_attention_num_buckets
                rp_bucket[idx, num_prefix_row:, :num_prefix_row] = self.relative_attention_num_buckets + 1

        values: Tensor = self.relative_attention_bias(rp_bucket)
        if values.dim() != 4:
            raise ValueError("Wrong dimension of values tensor")
        values = values.permute([0, 3, 1, 2])

        return values


class RelativePositionBias1D(RelativePositionBiasBase):
    def __init__(self, scaling_factor=1, max_distance=128, **kwargs):
        """
        Reimplementation of T5 relative position bias. Distance between given tokens is their distance in the sequence.
        Parameters are the same as in base class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        if self.scaling_factor != 1:
            raise ValueError("No need to scale 1d features")
        relative_position = self.get_relative_position(
            torch.arange(attention_mask.size(1), dtype=torch.long, device=attention_mask.device)[None, :]
        )

        return relative_position


class RelativePositionBiasHorizontal(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings horizontal distance between two tokens. Parameters are the same as in base
        class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        if not self.scaling_factor > 1.0:
            raise ValueError("Need to scale the values of bboxes, as there are in small (0,1) range")
        if bbox is None:
            raise ValueError("Bbox is required for horizontal relative position bias")
        # get x positions of left point of bbox
        horizontal_position: Tensor = bbox[:, :, [0, 2]].mean(dim=-1)

        return self.get_relative_position(horizontal_position)


class RelativePositionBiasVertical(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings vertical distance between two tokens. Parameters are the same as in base
        class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        if not self.scaling_factor > 1.0:
            raise ValueError("Need to scale the values of bboxes, as there are in small (0,1) range")
        if bbox is None:
            raise ValueError("Bbox is required for vertical relative position bias")
        # get y positions of middle of bbox
        vertical_position: Tensor = bbox[:, :, [1, 3]].mean(dim=-1)

        return self.get_relative_position(vertical_position)


class RelativePositionBiasAggregated(nn.Module):
    def __init__(self, modules: Sequence[RelativePositionBiasBase]):
        """
        Class which sums up various computed biases.

        Args:
            modules (Sequence[RelativePositionBiasBase]):
                List of relative bias modules.
        """
        super().__init__()
        self.biases = nn.ModuleList(modules)

    def forward(
        self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None
    ) -> Union[float, Tensor]:
        output = 0.0
        for bias in self.biases:  # type: ignore
            output = bias(attention_mask, bbox) + output

        return output


BIAS_CLASSES = {
    "1d": RelativePositionBias1D,
    "horizontal": RelativePositionBiasHorizontal,
    "vertical": RelativePositionBiasVertical,
}


def create_relative_bias(config: UdopConfig) -> Sequence[RelativePositionBiasBase]:
    """
    Creates empty list or one/multiple relative biases.

    :param config: Model's configuration :return: Sequence with created bias modules.
    """
    bias_list = []
    if hasattr(config, "relative_bias_args"):
        for bias_kwargs_org in config.relative_bias_args:
            bias_kwargs = deepcopy(bias_kwargs_org)
            bias_type = bias_kwargs.pop("type")
            model_num_heads = config.num_heads if hasattr(config, "num_heads") else config.num_attention_heads
            if "num_heads" in bias_kwargs:
                if bias_kwargs["num_heads"] != model_num_heads:
                    raise ValueError("Number of heads must match num of heads in the model")
            else:
                bias_kwargs["num_heads"] = model_num_heads
            bias_list.append(BIAS_CLASSES[bias_type](**bias_kwargs))  # type: ignore

    return bias_list


class UdopStack(UdopPreTrainedModel):
    """
    This class is based on `T5Stack`, but modified to take into account the image modality as well as 2D position
    embeddings.
    """

    def __init__(self, config, embed_tokens=None, embed_patches=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.embed_patches = embed_patches
        self.is_decoder = config.is_decoder
        self._max_length = config.max_length
        self.num_layers = config.num_layers

        self.block = nn.ModuleList(
            [UdopBlock(config, has_relative_attention_bias=bool(i == 0), layer_idx=i) for i in range(self.num_layers)]
        )
        self.final_layer_norm = UdopLayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.dropout = nn.Dropout(config.dropout_rate)

        if not self.is_decoder:
            self.cell_2d_embedding = UdopCellEmbeddings(config.max_2d_position_embeddings, config.hidden_size)

        # get weights from encoder position bias
        self.relative_bias = self._get_relative_bias(config)

    def _tie_weights(self):
        for bias in self.relative_bias.biases:
            if isinstance(bias, RelativePositionBias1D):
                self._tie_or_clone_weights(
                    bias.relative_attention_bias, self.block[0].layer[0].SelfAttention.relative_attention_bias
                )

    @staticmethod
    def _get_relative_bias(config: UdopConfig) -> RelativePositionBiasAggregated:
        relative_bias_list = create_relative_bias(config)
        return RelativePositionBiasAggregated(relative_bias_list)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        bbox=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        visual_bbox=None,
        image_embeddings=None,
        position_bias=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # input embeddings processing

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None and torch.numel(input_ids) > 0:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is None and input_ids is not None and torch.numel(input_ids) == 0:
            input_ids = torch.full((4, 1024), self.config.pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
            attention_mask = torch.zeros((4, 1024), device=input_ids.device, dtype=input_ids.dtype)
            bbox = torch.zeros((4, 1024, 4), device=input_ids.device, dtype=input_ids.dtype)
            input_shape = input_ids.size()
            position_bias = torch.zeros_like(self.get_extended_attention_mask(attention_mask, input_shape))
            # encoder_attention_mask = attention_mask
            logger.warning("Empty batch")
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to intialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embeddings = self.embed_patches(pixel_values)

        if image_embeddings is not None:
            # combine visual and OCR text embeddings
            num_patches = self.config.image_size // self.config.patch_size
            inputs_embeds, bbox, attention_mask = combine_image_text_embeddings(
                image_embeddings,
                inputs_embeds,
                bbox,
                visual_bbox,
                attention_mask,
                num_patches,
                0,
                self.config.image_size,
                self.config.patch_size,
            )
            input_shape = inputs_embeds.size()[:-1]

        if not self.is_decoder and bbox is not None:
            inputs_embeds += self.cell_2d_embedding(bbox)

        batch_size, seq_length = input_shape

        if use_cache is True:
            assert self.is_decoder, "`use_cache` can only be set to `True` if {} is used as a decoder".format(self)

        # initialize past_key_values
        return_legacy_cache = False
        return_self_attention_cache = False
        if self.is_decoder and (use_cache or past_key_values is not None):
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. "
                    "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                    "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
            elif past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache if past_key_values is not None else None,
                output_attentions,
            )
        else:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        if self.is_decoder:  # modified lines
            position_bias = None
        else:
            position_bias = self.relative_bias(attention_mask=attention_mask, bbox=bbox)
            position_bias = position_bias + causal_mask
        encoder_decoder_position_bias = None

        hidden_states = inputs_embeds

        hidden_states = self.dropout(hidden_states)

        for i, layer_module in enumerate(self.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=causal_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[i],
                past_key_value=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            if use_cache is False:  # MP fixes
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, next_decoder_cache = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    attention_mask,
                    next_cache,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithAttentionMask(
            last_hidden_state=hidden_states,
            attention_mask=attention_mask,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


@add_start_docstrings(
    "The bare UDOP encoder-decoder Transformer outputting raw hidden-states without any specific head on top.",
    UDOP_START_DOCSTRING,
)
class UdopModel(UdopPreTrainedModel):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "encoder.embed_patches.proj.weight",
        "encoder.embed_patches.proj.bias",
        "encoder.relative_bias.biases.0.relative_attention_bias.weight",
        "decoder.relative_bias.biases.0.relative_attention_bias.weight",
    ]

    def __init__(self, config):
        super(UdopModel, self).__init__(config)

        # text and image embeddings
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_embed = UdopPatchEmbeddings(config)

        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UdopStack(encoder_config, self.shared, self.patch_embed)

        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UdopStack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(UDOP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        bbox: Dict[str, Any] = None,
        pixel_values: Optional[Tensor] = None,
        visual_bbox: Dict[str, Any] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_outputs: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        use_cache=True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[Tensor, ...]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from datasets import load_dataset
        >>> import torch

        >>> # load model and processor
        >>> # in this case, we already have performed OCR ourselves
        >>> # so we initialize the processor with `apply_ocr=False`
        >>> processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
        >>> model = AutoModel.from_pretrained("microsoft/udop-large")

        >>> # load an example image, along with the words and coordinates
        >>> # which were extracted using an OCR engine
        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train", trust_remote_code=True)
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> inputs = processor(image, words, boxes=boxes, return_tensors="pt")

        >>> decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

        >>> # forward pass
        >>> outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 1, 1024]
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                visual_bbox=visual_bbox,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]
        encoder_attention_mask = encoder_outputs.attention_mask if return_dict else encoder_outputs[1]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            # we filter out the attention mask
            decoder_outputs = tuple(value for idx, value in enumerate(decoder_outputs) if idx != 1)
            encoder_outputs = tuple(value for idx, value in enumerate(encoder_outputs) if idx != 1)
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """The UDOP encoder-decoder Transformer with a language modeling head on top, enabling to generate text given document
    images and an optional prompt.

    This class is based on [`T5ForConditionalGeneration`], extended to deal with images and layout (2D) data.""",
    UDOP_START_DOCSTRING,
)
class UdopForConditionalGeneration(UdopPreTrainedModel, GenerationMixin):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "encoder.embed_patches.proj.weight",
        "encoder.embed_patches.proj.bias",
        "encoder.relative_bias.biases.0.relative_attention_bias.weight",
        "decoder.relative_bias.biases.0.relative_attention_bias.weight",
        "lm_head.weight",
    ]

    def __init__(self, config):
        super(UdopForConditionalGeneration, self).__init__(config)

        # text and image embeddings
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_embed = UdopPatchEmbeddings(config)

        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UdopStack(encoder_config, self.shared, self.patch_embed)

        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UdopStack(decoder_config, self.shared)

        # The weights of the language modeling head are shared with those of the encoder and decoder
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(UDOP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        bbox: Dict[str, Any] = None,
        pixel_values: Optional[Tensor] = None,
        visual_bbox: Dict[str, Any] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_outputs: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        use_cache=True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[Tensor, ...]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, UdopForConditionalGeneration
        >>> from datasets import load_dataset

        >>> # load model and processor
        >>> # in this case, we already have performed OCR ourselves
        >>> # so we initialize the processor with `apply_ocr=False`
        >>> processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
        >>> model = UdopForConditionalGeneration.from_pretrained("microsoft/udop-large")

        >>> # load an example image, along with the words and coordinates
        >>> # which were extracted using an OCR engine
        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train", trust_remote_code=True)
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> # one can use the various task prefixes (prompts) used during pre-training
        >>> # e.g. the task prefix for DocVQA is "Question answering. "
        >>> question = "Question answering. What is the date on the form?"
        >>> encoding = processor(image, question, text_pair=words, boxes=boxes, return_tensors="pt")

        >>> # autoregressive generation
        >>> predicted_ids = model.generate(**encoding)
        >>> print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
        9/30/92
        ```"""

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                bbox=bbox,
                visual_bbox=visual_bbox,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]
        encoder_attention_mask = encoder_outputs.attention_mask if return_dict else encoder_outputs[1]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.config.d_model**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[2:] + (encoder_outputs[0],) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration._reorder_cache
    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
    "The bare UDOP Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    UDOP_START_DOCSTRING,
)
class UdopEncoderModel(UdopPreTrainedModel):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "encoder.embed_patches.proj.weight",
        "encoder.embed_patches.proj.bias",
        "encoder.relative_bias.biases.0.relative_attention_bias.weight",
    ]

    def __init__(self, config: UdopConfig):
        super().__init__(config)

        # text and image embeddings
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_embed = UdopPatchEmbeddings(config)

        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UdopStack(encoder_config, self.shared, self.patch_embed)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(UDOP_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithAttentionMask, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Tensor = None,
        bbox: Dict[str, Any] = None,
        attention_mask: Tensor = None,
        pixel_values: Optional[Tensor] = None,
        visual_bbox: Dict[str, Any] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutputWithAttentionMask]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, UdopEncoderModel
        >>> from huggingface_hub import hf_hub_download
        >>> from datasets import load_dataset

        >>> # load model and processor
        >>> # in this case, we already have performed OCR ourselves
        >>> # so we initialize the processor with `apply_ocr=False`
        >>> processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
        >>> model = UdopEncoderModel.from_pretrained("microsoft/udop-large")

        >>> # load an example image, along with the words and coordinates
        >>> # which were extracted using an OCR engine
        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train", trust_remote_code=True)
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> encoding = processor(image, words, boxes=boxes, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            bbox=bbox,
            visual_bbox=visual_bbox,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


__all__ = ["UdopForConditionalGeneration", "UdopPreTrainedModel", "UdopModel", "UdopEncoderModel"]

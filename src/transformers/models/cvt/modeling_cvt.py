# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch CvT model."""

import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_cvt import CvtConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "CvtConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/cvt-13"
_EXPECTED_OUTPUT_SHAPE = [1, 384, 14, 14]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/cvt-13"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


@dataclass
class BaseModelOutputWithCLSToken(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cls_token_value (`torch.FloatTensor` of shape `(batch_size, 1, hidden_size)`):
            Classification token at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
    """

    last_hidden_state: torch.FloatTensor = None
    cls_token_value: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath
class CvtDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class CvtEmbeddings(nn.Module):
    """
    Construct the CvT embeddings.
    """

    def __init__(self, patch_size, num_channels, embed_dim, stride, padding, dropout_rate):
        super().__init__()
        self.convolution_embeddings = CvtConvEmbeddings(
            patch_size=patch_size, num_channels=num_channels, embed_dim=embed_dim, stride=stride, padding=padding
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, pixel_values):
        hidden_state = self.convolution_embeddings(pixel_values)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class CvtConvEmbeddings(nn.Module):
    """
    Image to Conv Embedding.
    """

    def __init__(self, patch_size, num_channels, embed_dim, stride, padding):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.patch_size = patch_size
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.normalization = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values):
        pixel_values = self.projection(pixel_values)
        batch_size, num_channels, height, width = pixel_values.shape
        hidden_size = height * width
        # rearrange "b c h w -> b (h w) c"
        pixel_values = pixel_values.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        if self.normalization:
            pixel_values = self.normalization(pixel_values)
        # rearrange "b (h w) c" -> b c h w"
        pixel_values = pixel_values.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        return pixel_values


class CvtSelfAttentionConvProjection(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, stride):
        super().__init__()
        self.convolution = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
            groups=embed_dim,
        )
        self.normalization = nn.BatchNorm2d(embed_dim)

    def forward(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class CvtSelfAttentionLinearProjection(nn.Module):
    def forward(self, hidden_state):
        batch_size, num_channels, height, width = hidden_state.shape
        hidden_size = height * width
        # rearrange " b c h w -> b (h w) c"
        hidden_state = hidden_state.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        return hidden_state


class CvtSelfAttentionProjection(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, stride, projection_method="dw_bn"):
        super().__init__()
        if projection_method == "dw_bn":
            self.convolution_projection = CvtSelfAttentionConvProjection(embed_dim, kernel_size, padding, stride)
        self.linear_projection = CvtSelfAttentionLinearProjection()

    def forward(self, hidden_state):
        hidden_state = self.convolution_projection(hidden_state)
        hidden_state = self.linear_projection(hidden_state)
        return hidden_state


class CvtSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        with_cls_token=True,
        **kwargs,
    ):
        super().__init__()
        self.scale = embed_dim**-0.5
        self.with_cls_token = with_cls_token
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.convolution_projection_query = CvtSelfAttentionProjection(
            embed_dim,
            kernel_size,
            padding_q,
            stride_q,
            projection_method="linear" if qkv_projection_method == "avg" else qkv_projection_method,
        )
        self.convolution_projection_key = CvtSelfAttentionProjection(
            embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method
        )
        self.convolution_projection_value = CvtSelfAttentionProjection(
            embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method
        )

        self.projection_query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.projection_key = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.projection_value = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.dropout = nn.Dropout(attention_drop_rate)

    def rearrange_for_multi_head_attention(self, hidden_state):
        batch_size, hidden_size, _ = hidden_state.shape
        head_dim = self.embed_dim // self.num_heads
        # rearrange 'b t (h d) -> b h t d'
        return hidden_state.view(batch_size, hidden_size, self.num_heads, head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_state, height, width):
        if self.with_cls_token:
            cls_token, hidden_state = torch.split(hidden_state, [1, height * width], 1)
        batch_size, hidden_size, num_channels = hidden_state.shape
        # rearrange "b (h w) c -> b c h w"
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)

        key = self.convolution_projection_key(hidden_state)
        query = self.convolution_projection_query(hidden_state)
        value = self.convolution_projection_value(hidden_state)

        if self.with_cls_token:
            query = torch.cat((cls_token, query), dim=1)
            key = torch.cat((cls_token, key), dim=1)
            value = torch.cat((cls_token, value), dim=1)

        head_dim = self.embed_dim // self.num_heads

        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))

        attention_score = torch.einsum("bhlk,bhtk->bhlt", [query, key]) * self.scale
        attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.einsum("bhlt,bhtv->bhlv", [attention_probs, value])
        # rearrange"b h t d -> b t (h d)"
        _, _, hidden_size, _ = context.shape
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, hidden_size, self.num_heads * head_dim)
        return context


class CvtSelfOutput(nn.Module):
    """
    The residual connection is defined in CvtLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, embed_dim, drop_rate):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, hidden_state, input_tensor):
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class CvtAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        with_cls_token=True,
    ):
        super().__init__()
        self.attention = CvtSelfAttention(
            num_heads,
            embed_dim,
            kernel_size,
            padding_q,
            padding_kv,
            stride_q,
            stride_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            with_cls_token,
        )
        self.output = CvtSelfOutput(embed_dim, drop_rate)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_state, height, width):
        self_output = self.attention(hidden_state, height, width)
        attention_output = self.output(self_output, hidden_state)
        return attention_output


class CvtIntermediate(nn.Module):
    def __init__(self, embed_dim, mlp_ratio):
        super().__init__()
        self.dense = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.activation = nn.GELU()

    def forward(self, hidden_state):
        hidden_state = self.dense(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class CvtOutput(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, drop_rate):
        super().__init__()
        self.dense = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, hidden_state, input_tensor):
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = hidden_state + input_tensor
        return hidden_state


class CvtLayer(nn.Module):
    """
    CvtLayer composed by attention layers, normalization and multi-layer perceptrons (mlps).
    """

    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        mlp_ratio,
        drop_path_rate,
        with_cls_token=True,
    ):
        super().__init__()
        self.attention = CvtAttention(
            num_heads,
            embed_dim,
            kernel_size,
            padding_q,
            padding_kv,
            stride_q,
            stride_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            drop_rate,
            with_cls_token,
        )

        self.intermediate = CvtIntermediate(embed_dim, mlp_ratio)
        self.output = CvtOutput(embed_dim, mlp_ratio, drop_rate)
        self.drop_path = CvtDropPath(drop_prob=drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_before = nn.LayerNorm(embed_dim)
        self.layernorm_after = nn.LayerNorm(embed_dim)

    def forward(self, hidden_state, height, width):
        self_attention_output = self.attention(
            self.layernorm_before(hidden_state),  # in Cvt, layernorm is applied before self-attention
            height,
            width,
        )
        attention_output = self_attention_output
        attention_output = self.drop_path(attention_output)

        # first residual connection
        hidden_state = attention_output + hidden_state

        # in Cvt, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_state)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_state)
        layer_output = self.drop_path(layer_output)
        return layer_output


class CvtStage(nn.Module):
    def __init__(self, config, stage):
        super().__init__()
        self.config = config
        self.stage = stage
        if self.config.cls_token[self.stage]:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.embed_dim[-1]))

        self.embedding = CvtEmbeddings(
            patch_size=config.patch_sizes[self.stage],
            stride=config.patch_stride[self.stage],
            num_channels=config.num_channels if self.stage == 0 else config.embed_dim[self.stage - 1],
            embed_dim=config.embed_dim[self.stage],
            padding=config.patch_padding[self.stage],
            dropout_rate=config.drop_rate[self.stage],
        )

        drop_path_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate[self.stage], config.depth[stage])]

        self.layers = nn.Sequential(
            *[
                CvtLayer(
                    num_heads=config.num_heads[self.stage],
                    embed_dim=config.embed_dim[self.stage],
                    kernel_size=config.kernel_qkv[self.stage],
                    padding_q=config.padding_q[self.stage],
                    padding_kv=config.padding_kv[self.stage],
                    stride_kv=config.stride_kv[self.stage],
                    stride_q=config.stride_q[self.stage],
                    qkv_projection_method=config.qkv_projection_method[self.stage],
                    qkv_bias=config.qkv_bias[self.stage],
                    attention_drop_rate=config.attention_drop_rate[self.stage],
                    drop_rate=config.drop_rate[self.stage],
                    drop_path_rate=drop_path_rates[self.stage],
                    mlp_ratio=config.mlp_ratio[self.stage],
                    with_cls_token=config.cls_token[self.stage],
                )
                for _ in range(config.depth[self.stage])
            ]
        )

    def forward(self, hidden_state):
        cls_token = None
        hidden_state = self.embedding(hidden_state)
        batch_size, num_channels, height, width = hidden_state.shape
        # rearrange b c h w -> b (h w) c"
        hidden_state = hidden_state.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        if self.config.cls_token[self.stage]:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            hidden_state = torch.cat((cls_token, hidden_state), dim=1)

        for layer in self.layers:
            layer_outputs = layer(hidden_state, height, width)
            hidden_state = layer_outputs

        if self.config.cls_token[self.stage]:
            cls_token, hidden_state = torch.split(hidden_state, [1, height * width], 1)
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        return hidden_state, cls_token


class CvtEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList([])
        for stage_idx in range(len(config.depth)):
            self.stages.append(CvtStage(config, stage_idx))

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        hidden_state = pixel_values

        cls_token = None
        for _, (stage_module) in enumerate(self.stages):
            hidden_state, cls_token = stage_module(hidden_state)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, cls_token, all_hidden_states] if v is not None)

        return BaseModelOutputWithCLSToken(
            last_hidden_state=hidden_state,
            cls_token_value=cls_token,
            hidden_states=all_hidden_states,
        )


class CvtPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CvtConfig
    base_model_prefix = "cvt"
    main_input_name = "pixel_values"
    _no_split_modules = ["CvtLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, CvtStage):
            if self.config.cls_token[module.stage]:
                module.cls_token.data = nn.init.trunc_normal_(
                    torch.zeros(1, 1, self.config.embed_dim[-1]), mean=0.0, std=self.config.initializer_range
                )


CVT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CvtConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CVT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`CvtImageProcessor.__call__`]
            for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Cvt Model transformer outputting raw hidden-states without any specific head on top.",
    CVT_START_DOCSTRING,
)
class CvtModel(CvtPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.encoder = CvtEncoder(config)
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CVT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithCLSToken,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithCLSToken]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithCLSToken(
            last_hidden_state=sequence_output,
            cls_token_value=encoder_outputs.cls_token_value,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    CVT_START_DOCSTRING,
)
class CvtForImageClassification(CvtPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.cvt = CvtModel(config, add_pooling_layer=False)
        self.layernorm = nn.LayerNorm(config.embed_dim[-1])
        # Classifier head
        self.classifier = (
            nn.Linear(config.embed_dim[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CVT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.cvt(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        cls_token = outputs[1]
        if self.config.cls_token[-1]:
            sequence_output = self.layernorm(cls_token)
        else:
            batch_size, num_channels, height, width = sequence_output.shape
            # rearrange "b c h w -> b (h w) c"
            sequence_output = sequence_output.view(batch_size, num_channels, height * width).permute(0, 2, 1)
            sequence_output = self.layernorm(sequence_output)

        sequence_output_mean = sequence_output.mean(dim=1)
        logits = self.classifier(sequence_output_mean)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)


__all__ = ["CvtForImageClassification", "CvtModel", "CvtPreTrainedModel"]

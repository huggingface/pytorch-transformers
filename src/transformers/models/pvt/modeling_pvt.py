# coding=utf-8
# Copyright 2023 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao and The HuggingFace Inc. team.
# All rights reserved.
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
"""PyTorch PVT model."""

import collections
import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_pvt import PvtConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PvtConfig"

_CHECKPOINT_FOR_DOC = "Zetatech/pvt-tiny-224"
_EXPECTED_OUTPUT_SHAPE = [1, 50, 512]

_IMAGE_CLASS_CHECKPOINT = "Zetatech/pvt-tiny-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


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


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Pvt
class PvtDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class PvtPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(
        self,
        config: PvtConfig,
        image_size: Union[int, Iterable[int]],
        patch_size: Union[int, Iterable[int]],
        stride: int,
        num_channels: int,
        hidden_size: int,
        cls_token: bool = False,
    ):
        super().__init__()
        self.config = config
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1 if cls_token else num_patches, hidden_size)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) if cls_token else None
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=stride, stride=patch_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = height * width

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == self.config.image_size * self.config.image_size:
            return self.position_embeddings
        embeddings = embeddings.reshape(1, height, width, -1).permute(0, 3, 1, 2)
        interpolated_embeddings = F.interpolate(embeddings, size=(height, width), mode="bilinear")
        interpolated_embeddings = interpolated_embeddings.reshape(1, -1, height * width).permute(0, 2, 1)
        return interpolated_embeddings

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        patch_embed = self.projection(pixel_values)
        *_, height, width = patch_embed.shape
        patch_embed = patch_embed.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(patch_embed)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_token, embeddings), dim=1)
            position_embeddings = self.interpolate_pos_encoding(self.position_embeddings[:, 1:], height, width)
            position_embeddings = torch.cat((self.position_embeddings[:, :1], position_embeddings), dim=1)
        else:
            position_embeddings = self.interpolate_pos_encoding(self.position_embeddings, height, width)
        embeddings = self.dropout(embeddings + position_embeddings)

        return embeddings, height, width


class PvtSelfOutput(nn.Module):
    def __init__(self, config: PvtConfig, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class PvtEfficientSelfAttention(nn.Module):
    """Efficient self-attention mechanism with reduction of the sequence [PvT paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(
        self, config: PvtConfig, hidden_size: int, num_attention_heads: int, sequences_reduction_ratio: float
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sequences_reduction_ratio = sequences_reduction_ratio
        if sequences_reduction_ratio > 1:
            self.sequence_reduction = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sequences_reduction_ratio, stride=sequences_reduction_ratio
            )
            self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, hidden_states: int) -> torch.Tensor:
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.sequences_reduction_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            # Reshape to (batch_size, num_channels, height, width)
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            # Apply sequence reduction
            hidden_states = self.sequence_reduction(hidden_states)
            # Reshape back to (batch_size, seq_len, num_channels)
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class PvtAttention(nn.Module):
    def __init__(
        self, config: PvtConfig, hidden_size: int, num_attention_heads: int, sequences_reduction_ratio: float
    ):
        super().__init__()
        self.self = PvtEfficientSelfAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequences_reduction_ratio=sequences_reduction_ratio,
        )
        self.output = PvtSelfOutput(config, hidden_size=hidden_size)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = False
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PvtFFN(nn.Module):
    def __init__(
        self,
        config: PvtConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class PvtLayer(nn.Module):
    def __init__(
        self,
        config: PvtConfig,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequences_reduction_ratio: float,
        mlp_ratio: float,
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = PvtAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequences_reduction_ratio=sequences_reduction_ratio,
        )
        self.drop_path = PvtDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = PvtFFN(config=config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = False):
        self_attention_outputs = self.attention(
            hidden_states=self.layer_norm_1(hidden_states),
            height=height,
            width=width,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(self.layer_norm_2(hidden_states))

        mlp_output = self.drop_path(mlp_output)
        layer_output = hidden_states + mlp_output

        outputs = (layer_output,) + outputs

        return outputs


class PvtEncoder(nn.Module):
    def __init__(self, config: PvtConfig):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        drop_path_decays = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()

        # patch embeddings
        embeddings = []

        for i in range(config.num_encoder_blocks):
            embeddings.append(
                PvtPatchEmbeddings(
                    config=config,
                    image_size=config.image_size if i == 0 else self.config.image_size // (2 ** (i + 1)),
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                    cls_token=i == config.num_encoder_blocks - 1,
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    PvtLayer(
                        config=config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequences_reduction_ratio=config.sequence_reduction_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]
        num_blocks = len(self.block)
        hidden_states = pixel_values
        for idx, (embedding_layer, block_layer) in enumerate(zip(self.patch_embeddings, self.block)):
            # first, obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)
            # second, send embeddings through blocks
            for block in block_layer:
                layer_outputs = block(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
            if idx != num_blocks - 1:
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class PvtPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PvtConfig
    base_model_prefix = "pvt"
    main_input_name = "pixel_values"
    _no_split_modules = []

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PvtPatchEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.cls_token is not None:
                module.cls_token.data = nn.init.trunc_normal_(
                    module.cls_token.data,
                    mean=0.0,
                    std=self.config.initializer_range,
                )


PVT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~PvtConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PVT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`PvtImageProcessor.__call__`]
            for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Pvt encoder outputting raw hidden-states without any specific head on top.",
    PVT_START_DOCSTRING,
)
class PvtModel(PvtPreTrainedModel):
    def __init__(self, config: PvtConfig):
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = PvtEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PVT_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """
    Pvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    PVT_START_DOCSTRING,
)
class PvtForImageClassification(PvtPreTrainedModel):
    def __init__(self, config: PvtConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.pvt = PvtModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(PVT_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.pvt(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["PvtForImageClassification", "PvtModel", "PvtPreTrainedModel"]

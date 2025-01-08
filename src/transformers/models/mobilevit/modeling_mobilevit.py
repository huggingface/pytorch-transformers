# coding=utf-8
# Copyright 2022 Apple Inc. and The HuggingFace Inc. team. All rights reserved.
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
#
# Original license: https://github.com/apple/ml-cvnets/blob/main/LICENSE
"""PyTorch MobileViT model."""

import math
from typing import Dict, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)
from .configuration_mobilevit import MobileViTConfig


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "MobileViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "apple/mobilevit-small"
_EXPECTED_OUTPUT_SHAPE = [1, 640, 8, 8]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "apple/mobilevit-small"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class MobileViTConvLayer(nn.Module):
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_normalization: bool = True,
        use_activation: Union[bool, str] = True,
    ) -> None:
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation

        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
        )

        if use_normalization:
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
        else:
            self.normalization = None

        if use_activation:
            if isinstance(use_activation, str):
                self.activation = ACT2FN[use_activation]
            elif isinstance(config.hidden_act, str):
                self.activation = ACT2FN[config.hidden_act]
            else:
                self.activation = config.hidden_act
        else:
            self.activation = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.convolution(features)
        if self.normalization is not None:
            features = self.normalization(features)
        if self.activation is not None:
            features = self.activation(features)
        return features


class MobileViTInvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int, dilation: int = 1
    ) -> None:
        super().__init__()
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.expand_1x1 = MobileViTConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1
        )

        self.conv_3x3 = MobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
        )

        self.reduce_1x1 = MobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features

        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)

        return residual + features if self.use_residual else features


class MobileViTMobileNetLayer(nn.Module):
    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int = 1, num_stages: int = 1
    ) -> None:
        super().__init__()

        self.layer = nn.ModuleList()
        for i in range(num_stages):
            layer = MobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
            )
            self.layer.append(layer)
            in_channels = out_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            features = layer_module(features)
        return features


class MobileViTSelfAttention(nn.Module):
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        super().__init__()

        if hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

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
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class MobileViTSelfOutput(nn.Module):
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MobileViTAttention(nn.Module):
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        super().__init__()
        self.attention = MobileViTSelfAttention(config, hidden_size)
        self.output = MobileViTSelfOutput(config, hidden_size)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self_outputs = self.attention(hidden_states)
        attention_output = self.output(self_outputs)
        return attention_output


class MobileViTIntermediate(nn.Module):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MobileViTOutput(nn.Module):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class MobileViTTransformerLayer(nn.Module):
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.attention = MobileViTAttention(config, hidden_size)
        self.intermediate = MobileViTIntermediate(config, hidden_size, intermediate_size)
        self.output = MobileViTOutput(config, hidden_size, intermediate_size)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return layer_output


class MobileViTTransformer(nn.Module):
    def __init__(self, config: MobileViTConfig, hidden_size: int, num_stages: int) -> None:
        super().__init__()

        self.layer = nn.ModuleList()
        for _ in range(num_stages):
            transformer_layer = MobileViTTransformerLayer(
                config,
                hidden_size=hidden_size,
                intermediate_size=int(hidden_size * config.mlp_ratio),
            )
            self.layer.append(transformer_layer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class MobileViTLayer(nn.Module):
    """
    MobileViT block: https://arxiv.org/abs/2110.02178
    """

    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        hidden_size: int,
        num_stages: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size

        if stride == 2:
            self.downsampling_layer = MobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
            )
            in_channels = out_channels
        else:
            self.downsampling_layer = None

        self.conv_kxk = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
        )

        self.conv_1x1 = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        self.transformer = MobileViTTransformer(
            config,
            hidden_size=hidden_size,
            num_stages=num_stages,
        )

        self.layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        self.conv_projection = MobileViTConvLayer(
            config, in_channels=hidden_size, out_channels=in_channels, kernel_size=1
        )

        self.fusion = MobileViTConvLayer(
            config, in_channels=2 * in_channels, out_channels=in_channels, kernel_size=config.conv_kernel_size
        )

    def unfolding(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        patch_width, patch_height = self.patch_width, self.patch_height
        patch_area = int(patch_width * patch_height)

        batch_size, channels, orig_height, orig_width = features.shape

        new_height = (
            torch_int(torch.ceil(orig_height / patch_height) * patch_height)
            if torch.jit.is_tracing()
            else int(math.ceil(orig_height / patch_height) * patch_height)
        )
        new_width = (
            torch_int(torch.ceil(orig_width / patch_width) * patch_width)
            if torch.jit.is_tracing()
            else int(math.ceil(orig_width / patch_width) * patch_width)
        )

        interpolate = False
        if new_width != orig_width or new_height != orig_height:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            features = nn.functional.interpolate(
                features, size=(new_height, new_width), mode="bilinear", align_corners=False
            )
            interpolate = True

        # number of patches along width and height
        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width

        # convert from shape (batch_size, channels, orig_height, orig_width)
        # to the shape (batch_size * patch_area, num_patches, channels)
        patches = features.reshape(
            batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width
        )
        patches = patches.transpose(1, 2)
        patches = patches.reshape(batch_size, channels, num_patches, patch_area)
        patches = patches.transpose(1, 3)
        patches = patches.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_height, orig_width),
            "batch_size": batch_size,
            "channels": channels,
            "interpolate": interpolate,
            "num_patches": num_patches,
            "num_patches_width": num_patch_width,
            "num_patches_height": num_patch_height,
        }
        return patches, info_dict

    def folding(self, patches: torch.Tensor, info_dict: Dict) -> torch.Tensor:
        patch_width, patch_height = self.patch_width, self.patch_height
        patch_area = int(patch_width * patch_height)

        batch_size = info_dict["batch_size"]
        channels = info_dict["channels"]
        num_patches = info_dict["num_patches"]
        num_patch_height = info_dict["num_patches_height"]
        num_patch_width = info_dict["num_patches_width"]

        # convert from shape (batch_size * patch_area, num_patches, channels)
        # back to shape (batch_size, channels, orig_height, orig_width)
        features = patches.contiguous().view(batch_size, patch_area, num_patches, -1)
        features = features.transpose(1, 3)
        features = features.reshape(
            batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width
        )
        features = features.transpose(1, 2)
        features = features.reshape(
            batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width
        )

        if info_dict["interpolate"]:
            features = nn.functional.interpolate(
                features, size=info_dict["orig_size"], mode="bilinear", align_corners=False
            )

        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # reduce spatial dimensions if needed
        if self.downsampling_layer:
            features = self.downsampling_layer(features)

        residual = features

        # local representation
        features = self.conv_kxk(features)
        features = self.conv_1x1(features)

        # convert feature map to patches
        patches, info_dict = self.unfolding(features)

        # learn global representations
        patches = self.transformer(patches)
        patches = self.layernorm(patches)

        # convert patches back to feature maps
        features = self.folding(patches, info_dict)

        features = self.conv_projection(features)
        features = self.fusion(torch.cat((residual, features), dim=1))
        return features


class MobileViTEncoder(nn.Module):
    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__()
        self.config = config

        self.layer = nn.ModuleList()
        self.gradient_checkpointing = False

        # segmentation architectures like DeepLab and PSPNet modify the strides
        # of the classification backbones
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        elif config.output_stride == 16:
            dilate_layer_5 = True

        dilation = 1

        layer_1 = MobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[0],
            out_channels=config.neck_hidden_sizes[1],
            stride=1,
            num_stages=1,
        )
        self.layer.append(layer_1)

        layer_2 = MobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[1],
            out_channels=config.neck_hidden_sizes[2],
            stride=2,
            num_stages=3,
        )
        self.layer.append(layer_2)

        layer_3 = MobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[2],
            out_channels=config.neck_hidden_sizes[3],
            stride=2,
            hidden_size=config.hidden_sizes[0],
            num_stages=2,
        )
        self.layer.append(layer_3)

        if dilate_layer_4:
            dilation *= 2

        layer_4 = MobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[3],
            out_channels=config.neck_hidden_sizes[4],
            stride=2,
            hidden_size=config.hidden_sizes[1],
            num_stages=4,
            dilation=dilation,
        )
        self.layer.append(layer_4)

        if dilate_layer_5:
            dilation *= 2

        layer_5 = MobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[4],
            out_channels=config.neck_hidden_sizes[5],
            stride=2,
            hidden_size=config.hidden_sizes[2],
            num_stages=3,
            dilation=dilation,
        )
        self.layer.append(layer_5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                hidden_states = layer_module(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class MobileViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileViTConfig
    base_model_prefix = "mobilevit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MobileViTLayer"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


MOBILEVIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MOBILEVIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileViTImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MobileViT model outputting raw hidden-states without any specific head on top.",
    MOBILEVIT_START_DOCSTRING,
)
class MobileViTModel(MobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig, expand_output: bool = True):
        super().__init__(config)
        self.config = config
        self.expand_output = expand_output

        self.conv_stem = MobileViTConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=config.neck_hidden_sizes[0],
            kernel_size=3,
            stride=2,
        )

        self.encoder = MobileViTEncoder(config)

        if self.expand_output:
            self.conv_1x1_exp = MobileViTConvLayer(
                config,
                in_channels=config.neck_hidden_sizes[5],
                out_channels=config.neck_hidden_sizes[6],
                kernel_size=1,
            )

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        for layer_index, heads in heads_to_prune.items():
            mobilevit_layer = self.encoder.layer[layer_index]
            if isinstance(mobilevit_layer, MobileViTLayer):
                for transformer_layer in mobilevit_layer.transformer.layer:
                    transformer_layer.attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.conv_stem(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.expand_output:
            last_hidden_state = self.conv_1x1_exp(encoder_outputs[0])

            # global average pooling: (batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = torch.mean(last_hidden_state, dim=[-2, -1], keepdim=False)
        else:
            last_hidden_state = encoder_outputs[0]
            pooled_output = None

        if not return_dict:
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)
            return output + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    MobileViT model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILEVIT_START_DOCSTRING,
)
class MobileViTForImageClassification(MobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mobilevit = MobileViTModel(config)

        # Classifier head
        self.dropout = nn.Dropout(config.classifier_dropout_prob, inplace=True)
        self.classifier = (
            nn.Linear(config.neck_hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilevit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(self.dropout(pooled_output))

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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class MobileViTASPPPooling(nn.Module):
    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.conv_1x1 = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_size = features.shape[-2:]
        features = self.global_pool(features)
        features = self.conv_1x1(features)
        features = nn.functional.interpolate(features, size=spatial_size, mode="bilinear", align_corners=False)
        return features


class MobileViTASPP(nn.Module):
    """
    ASPP module defined in DeepLab papers: https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__()

        in_channels = config.neck_hidden_sizes[-2]
        out_channels = config.aspp_out_channels

        if len(config.atrous_rates) != 3:
            raise ValueError("Expected 3 values for atrous_rates")

        self.convs = nn.ModuleList()

        in_projection = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
        )
        self.convs.append(in_projection)

        self.convs.extend(
            [
                MobileViTConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=rate,
                    use_activation="relu",
                )
                for rate in config.atrous_rates
            ]
        )

        pool_layer = MobileViTASPPPooling(config, in_channels, out_channels)
        self.convs.append(pool_layer)

        self.project = MobileViTConvLayer(
            config, in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, use_activation="relu"
        )

        self.dropout = nn.Dropout(p=config.aspp_dropout_prob)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pyramid = []
        for conv in self.convs:
            pyramid.append(conv(features))
        pyramid = torch.cat(pyramid, dim=1)

        pooled_features = self.project(pyramid)
        pooled_features = self.dropout(pooled_features)
        return pooled_features


class MobileViTDeepLabV3(nn.Module):
    """
    DeepLabv3 architecture: https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__()
        self.aspp = MobileViTASPP(config)

        self.dropout = nn.Dropout2d(config.classifier_dropout_prob)

        self.classifier = MobileViTConvLayer(
            config,
            in_channels=config.aspp_out_channels,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        features = self.aspp(hidden_states[-1])
        features = self.dropout(features)
        features = self.classifier(features)
        return features


@add_start_docstrings(
    """
    MobileViT model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """,
    MOBILEVIT_START_DOCSTRING,
)
class MobileViTForSemanticSegmentation(MobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mobilevit = MobileViTModel(config, expand_output=False)
        self.segmentation_head = MobileViTDeepLabV3(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from transformers import AutoImageProcessor, MobileViTForSemanticSegmentation

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
        >>> model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        outputs = self.mobilevit(
            pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.segmentation_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
            loss = loss_fct(upsampled_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )


__all__ = [
    "MobileViTForImageClassification",
    "MobileViTForSemanticSegmentation",
    "MobileViTModel",
    "MobileViTPreTrainedModel",
]

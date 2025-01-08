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
"""PyTorch MobileNetV2 model."""

from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mobilenet_v2 import MobileNetV2Config


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "MobileNetV2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/mobilenet_v2_1.0_224"
_EXPECTED_OUTPUT_SHAPE = [1, 1280, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/mobilenet_v2_1.0_224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


def _build_tf_to_pytorch_map(model, config, tf_weights=None):
    """
    A map of modules from TF to PyTorch.
    """

    tf_to_pt_map = {}

    if isinstance(model, (MobileNetV2ForImageClassification, MobileNetV2ForSemanticSegmentation)):
        backbone = model.mobilenet_v2
    else:
        backbone = model

    # Use the EMA weights if available
    def ema(x):
        return x + "/ExponentialMovingAverage" if x + "/ExponentialMovingAverage" in tf_weights else x

    prefix = "MobilenetV2/Conv/"
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_stem.first_conv.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.first_conv.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.first_conv.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.first_conv.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.first_conv.normalization.running_var

    prefix = "MobilenetV2/expanded_conv/depthwise/"
    tf_to_pt_map[ema(prefix + "depthwise_weights")] = backbone.conv_stem.conv_3x3.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.conv_3x3.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.conv_3x3.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.conv_3x3.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.conv_3x3.normalization.running_var

    prefix = "MobilenetV2/expanded_conv/project/"
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_stem.reduce_1x1.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.reduce_1x1.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.reduce_1x1.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.reduce_1x1.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.reduce_1x1.normalization.running_var

    for i in range(16):
        tf_index = i + 1
        pt_index = i
        pointer = backbone.layer[pt_index]

        prefix = f"MobilenetV2/expanded_conv_{tf_index}/expand/"
        tf_to_pt_map[ema(prefix + "weights")] = pointer.expand_1x1.convolution.weight
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.expand_1x1.normalization.bias
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.expand_1x1.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.expand_1x1.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.expand_1x1.normalization.running_var

        prefix = f"MobilenetV2/expanded_conv_{tf_index}/depthwise/"
        tf_to_pt_map[ema(prefix + "depthwise_weights")] = pointer.conv_3x3.convolution.weight
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.conv_3x3.normalization.bias
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.conv_3x3.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.conv_3x3.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.conv_3x3.normalization.running_var

        prefix = f"MobilenetV2/expanded_conv_{tf_index}/project/"
        tf_to_pt_map[ema(prefix + "weights")] = pointer.reduce_1x1.convolution.weight
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.reduce_1x1.normalization.bias
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.reduce_1x1.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.reduce_1x1.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.reduce_1x1.normalization.running_var

    prefix = "MobilenetV2/Conv_1/"
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_1x1.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_1x1.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_1x1.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_1x1.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_1x1.normalization.running_var

    if isinstance(model, MobileNetV2ForImageClassification):
        prefix = "MobilenetV2/Logits/Conv2d_1c_1x1/"
        tf_to_pt_map[ema(prefix + "weights")] = model.classifier.weight
        tf_to_pt_map[ema(prefix + "biases")] = model.classifier.bias

    if isinstance(model, MobileNetV2ForSemanticSegmentation):
        prefix = "image_pooling/"
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_pool.convolution.weight
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_pool.normalization.bias
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_pool.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = model.segmentation_head.conv_pool.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = (
            model.segmentation_head.conv_pool.normalization.running_var
        )

        prefix = "aspp0/"
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_aspp.convolution.weight
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_aspp.normalization.bias
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_aspp.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = model.segmentation_head.conv_aspp.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = (
            model.segmentation_head.conv_aspp.normalization.running_var
        )

        prefix = "concat_projection/"
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_projection.convolution.weight
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_projection.normalization.bias
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_projection.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = (
            model.segmentation_head.conv_projection.normalization.running_mean
        )
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = (
            model.segmentation_head.conv_projection.normalization.running_var
        )

        prefix = "logits/semantic/"
        tf_to_pt_map[ema(prefix + "weights")] = model.segmentation_head.classifier.convolution.weight
        tf_to_pt_map[ema(prefix + "biases")] = model.segmentation_head.classifier.convolution.bias

    return tf_to_pt_map


def load_tf_weights_in_mobilenet_v2(model, config, tf_checkpoint_path):
    """Load TensorFlow checkpoints in a PyTorch model."""
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_checkpoint_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_checkpoint_path, name)
        tf_weights[name] = array

    # Build TF to PyTorch weights loading map
    tf_to_pt_map = _build_tf_to_pytorch_map(model, config, tf_weights)

    for name, pointer in tf_to_pt_map.items():
        logger.info(f"Importing {name}")
        if name not in tf_weights:
            logger.info(f"{name} not in tf pre-trained weights, skipping")
            continue

        array = tf_weights[name]

        if "depthwise_weights" in name:
            logger.info("Transposing depthwise")
            array = np.transpose(array, (2, 3, 0, 1))
        elif "weights" in name:
            logger.info("Transposing")
            if len(pointer.shape) == 2:  # copying into linear layer
                array = array.squeeze().transpose()
            else:
                array = np.transpose(array, (3, 2, 0, 1))

        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")

        logger.info(f"Initialize PyTorch weight {name} {array.shape}")
        pointer.data = torch.from_numpy(array)

        tf_weights.pop(name, None)
        tf_weights.pop(name + "/RMSProp", None)
        tf_weights.pop(name + "/RMSProp_1", None)
        tf_weights.pop(name + "/ExponentialMovingAverage", None)
        tf_weights.pop(name + "/Momentum", None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    return model


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


def apply_depth_multiplier(config: MobileNetV2Config, channels: int) -> int:
    return make_divisible(int(round(channels * config.depth_multiplier)), config.depth_divisible_by, config.min_depth)


def apply_tf_padding(features: torch.Tensor, conv_layer: nn.Conv2d) -> torch.Tensor:
    """
    Apply TensorFlow-style "SAME" padding to a convolution layer. See the notes at:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    """
    in_height = int(features.shape[-2])
    in_width = int(features.shape[-1])
    stride_height, stride_width = conv_layer.stride
    kernel_height, kernel_width = conv_layer.kernel_size
    dilation_height, dilation_width = conv_layer.dilation

    if in_height % stride_height == 0:
        pad_along_height = max(kernel_height - stride_height, 0)
    else:
        pad_along_height = max(kernel_height - (in_height % stride_height), 0)

    if in_width % stride_width == 0:
        pad_along_width = max(kernel_width - stride_width, 0)
    else:
        pad_along_width = max(kernel_width - (in_width % stride_width), 0)

    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top

    padding = (
        pad_left * dilation_width,
        pad_right * dilation_width,
        pad_top * dilation_height,
        pad_bottom * dilation_height,
    )
    return nn.functional.pad(features, padding, "constant", 0.0)


class MobileNetV2ConvLayer(nn.Module):
    def __init__(
        self,
        config: MobileNetV2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_normalization: bool = True,
        use_activation: Union[bool, str] = True,
        layer_norm_eps: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.config = config

        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        padding = 0 if config.tf_padding else int((kernel_size - 1) / 2) * dilation

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
                eps=config.layer_norm_eps if layer_norm_eps is None else layer_norm_eps,
                momentum=0.997,
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
        if self.config.tf_padding:
            features = apply_tf_padding(features, self.convolution)
        features = self.convolution(features)
        if self.normalization is not None:
            features = self.normalization(features)
        if self.activation is not None:
            features = self.activation(features)
        return features


class MobileNetV2InvertedResidual(nn.Module):
    def __init__(
        self, config: MobileNetV2Config, in_channels: int, out_channels: int, stride: int, dilation: int = 1
    ) -> None:
        super().__init__()

        expanded_channels = make_divisible(
            int(round(in_channels * config.expand_ratio)), config.depth_divisible_by, config.min_depth
        )

        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.expand_1x1 = MobileNetV2ConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1
        )

        self.conv_3x3 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
        )

        self.reduce_1x1 = MobileNetV2ConvLayer(
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


class MobileNetV2Stem(nn.Module):
    def __init__(self, config: MobileNetV2Config, in_channels: int, expanded_channels: int, out_channels: int) -> None:
        super().__init__()

        # The very first layer is a regular 3x3 convolution with stride 2 that expands to 32 channels.
        # All other expansion layers use the expansion factor to compute the number of output channels.
        self.first_conv = MobileNetV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=2,
        )

        if config.first_layer_is_expansion:
            self.expand_1x1 = None
        else:
            self.expand_1x1 = MobileNetV2ConvLayer(
                config, in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=1
            )

        self.conv_3x3 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=1,
            groups=expanded_channels,
        )

        self.reduce_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.first_conv(features)
        if self.expand_1x1 is not None:
            features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)
        return features


class MobileNetV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileNetV2Config
    load_tf_weights = load_tf_weights_in_mobilenet_v2
    base_model_prefix = "mobilenet_v2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


MOBILENET_V2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileNetV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MOBILENET_V2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileNetV2ImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MobileNetV2 model outputting raw hidden-states without any specific head on top.",
    MOBILENET_V2_START_DOCSTRING,
)
class MobileNetV2Model(MobileNetV2PreTrainedModel):
    def __init__(self, config: MobileNetV2Config, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        # Output channels for the projection layers
        channels = [16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320]
        channels = [apply_depth_multiplier(config, x) for x in channels]

        # Strides for the depthwise layers
        strides = [2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]

        self.conv_stem = MobileNetV2Stem(
            config,
            in_channels=config.num_channels,
            expanded_channels=apply_depth_multiplier(config, 32),
            out_channels=channels[0],
        )

        current_stride = 2  # first conv layer has stride 2
        dilation = 1

        self.layer = nn.ModuleList()
        for i in range(16):
            # Keep making the feature maps smaller or use dilated convolution?
            if current_stride == config.output_stride:
                layer_stride = 1
                layer_dilation = dilation
                dilation *= strides[i]  # larger dilation starts in next block
            else:
                layer_stride = strides[i]
                layer_dilation = 1
                current_stride *= layer_stride

            self.layer.append(
                MobileNetV2InvertedResidual(
                    config,
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=layer_stride,
                    dilation=layer_dilation,
                )
            )

        if config.finegrained_output and config.depth_multiplier < 1.0:
            output_channels = 1280
        else:
            output_channels = apply_depth_multiplier(config, 1280)

        self.conv_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=channels[-1],
            out_channels=output_channels,
            kernel_size=1,
        )

        self.pooler = nn.AdaptiveAvgPool2d((1, 1)) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    @add_start_docstrings_to_model_forward(MOBILENET_V2_INPUTS_DOCSTRING)
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

        hidden_states = self.conv_stem(pixel_values)

        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        last_hidden_state = self.conv_1x1(hidden_states)

        if self.pooler is not None:
            pooled_output = torch.flatten(self.pooler(last_hidden_state), start_dim=1)
        else:
            pooled_output = None

        if not return_dict:
            return tuple(v for v in [last_hidden_state, pooled_output, all_hidden_states] if v is not None)

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )


@add_start_docstrings(
    """
    MobileNetV2 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILENET_V2_START_DOCSTRING,
)
class MobileNetV2ForImageClassification(MobileNetV2PreTrainedModel):
    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mobilenet_v2 = MobileNetV2Model(config)

        last_hidden_size = self.mobilenet_v2.conv_1x1.convolution.out_channels

        # Classifier head
        self.dropout = nn.Dropout(config.classifier_dropout_prob, inplace=True)
        self.classifier = nn.Linear(last_hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILENET_V2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
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

        outputs = self.mobilenet_v2(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

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


class MobileNetV2DeepLabV3Plus(nn.Module):
    """
    The neural network from the paper "Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation" https://arxiv.org/abs/1802.02611
    """

    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.conv_pool = MobileNetV2ConvLayer(
            config,
            in_channels=apply_depth_multiplier(config, 320),
            out_channels=256,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
            layer_norm_eps=1e-5,
        )

        self.conv_aspp = MobileNetV2ConvLayer(
            config,
            in_channels=apply_depth_multiplier(config, 320),
            out_channels=256,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
            layer_norm_eps=1e-5,
        )

        self.conv_projection = MobileNetV2ConvLayer(
            config,
            in_channels=512,
            out_channels=256,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
            layer_norm_eps=1e-5,
        )

        self.dropout = nn.Dropout2d(config.classifier_dropout_prob)

        self.classifier = MobileNetV2ConvLayer(
            config,
            in_channels=256,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_size = features.shape[-2:]

        features_pool = self.avg_pool(features)
        features_pool = self.conv_pool(features_pool)
        features_pool = nn.functional.interpolate(
            features_pool, size=spatial_size, mode="bilinear", align_corners=True
        )

        features_aspp = self.conv_aspp(features)

        features = torch.cat([features_pool, features_aspp], dim=1)

        features = self.conv_projection(features)
        features = self.dropout(features)
        features = self.classifier(features)
        return features


@add_start_docstrings(
    """
    MobileNetV2 model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """,
    MOBILENET_V2_START_DOCSTRING,
)
class MobileNetV2ForSemanticSegmentation(MobileNetV2PreTrainedModel):
    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mobilenet_v2 = MobileNetV2Model(config, add_pooling_layer=False)
        self.segmentation_head = MobileNetV2DeepLabV3Plus(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILENET_V2_INPUTS_DOCSTRING)
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
        >>> from transformers import AutoImageProcessor, MobileNetV2ForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")
        >>> model = MobileNetV2ForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")

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

        outputs = self.mobilenet_v2(
            pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.segmentation_head(encoder_hidden_states[-1])

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
    "MobileNetV2ForImageClassification",
    "MobileNetV2ForSemanticSegmentation",
    "MobileNetV2Model",
    "MobileNetV2PreTrainedModel",
    "load_tf_weights_in_mobilenet_v2",
]

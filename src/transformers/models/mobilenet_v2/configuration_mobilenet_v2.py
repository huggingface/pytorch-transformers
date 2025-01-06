# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""MobileNetV2 model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MobileNetV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileNetV2Model`]. It is used to instantiate a
    MobileNetV2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileNetV2
    [google/mobilenet_v2_1.0_224](https://huggingface.co/google/mobilenet_v2_1.0_224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        depth_multiplier (`float`, *optional*, defaults to 1.0):
            Shrinks or expands the number of channels in each layer. Default is 1.0, which starts the network with 32
            channels. This is sometimes also called "alpha" or "width multiplier".
        depth_divisible_by (`int`, *optional*, defaults to 8):
            The number of channels in each layer will always be a multiple of this number.
        min_depth (`int`, *optional*, defaults to 8):
            All layers will have at least this many channels.
        expand_ratio (`float`, *optional*, defaults to 6.0):
            The number of output channels of the first layer in each block is input channels times expansion ratio.
        output_stride (`int`, *optional*, defaults to 32):
            The ratio between the spatial resolution of the input and output feature maps. By default the model reduces
            the input dimensions by a factor of 32. If `output_stride` is 8 or 16, the model uses dilated convolutions
            on the depthwise layers instead of regular convolutions, so that the feature maps never become more than 8x
            or 16x smaller than the input image.
        first_layer_is_expansion (`bool`, *optional*, defaults to `True`):
            True if the very first convolution layer is also the expansion layer for the first expansion block.
        finegrained_output (`bool`, *optional*, defaults to `True`):
            If true, the number of output channels in the final convolution layer will stay large (1280) even if
            `depth_multiplier` is less than 1.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu6"`):
            The non-linear activation function (function or string) in the Transformer encoder and convolution layers.
        tf_padding (`bool`, *optional*, defaults to `True`):
            Whether to use TensorFlow padding rules on the convolution layers.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.8):
            The dropout ratio for attached classifiers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 0.001):
            The epsilon used by the layer normalization layers.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.

    Example:

    ```python
    >>> from transformers import MobileNetV2Config, MobileNetV2Model

    >>> # Initializing a "mobilenet_v2_1.0_224" style configuration
    >>> configuration = MobileNetV2Config()

    >>> # Initializing a model from the "mobilenet_v2_1.0_224" style configuration
    >>> model = MobileNetV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mobilenet_v2"

    def __init__(
        self,
        num_channels=3,
        image_size=224,
        depth_multiplier=1.0,
        depth_divisible_by=8,
        min_depth=8,
        expand_ratio=6.0,
        output_stride=32,
        first_layer_is_expansion=True,
        finegrained_output=True,
        hidden_act="relu6",
        tf_padding=True,
        classifier_dropout_prob=0.8,
        initializer_range=0.02,
        layer_norm_eps=0.001,
        semantic_loss_ignore_index=255,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be greater than zero.")

        self.num_channels = num_channels
        self.image_size = image_size
        self.depth_multiplier = depth_multiplier
        self.depth_divisible_by = depth_divisible_by
        self.min_depth = min_depth
        self.expand_ratio = expand_ratio
        self.output_stride = output_stride
        self.first_layer_is_expansion = first_layer_is_expansion
        self.finegrained_output = finegrained_output
        self.hidden_act = hidden_act
        self.tf_padding = tf_padding
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.semantic_loss_ignore_index = semantic_loss_ignore_index


class MobileNetV2OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([("pixel_values", {0: "batch"})])

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "image-classification":
            return OrderedDict([("logits", {0: "batch"})])
        else:
            return OrderedDict([("last_hidden_state", {0: "batch"}), ("pooler_output", {0: "batch"})])

    @property
    def atol_for_validation(self) -> float:
        return 1e-4


__all__ = ["MobileNetV2Config", "MobileNetV2OnnxConfig"]

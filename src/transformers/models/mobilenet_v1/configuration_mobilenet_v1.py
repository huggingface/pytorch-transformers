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
"""MobileNetV1 model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MobileNetV1Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileNetV1Model`]. It is used to instantiate a
    MobileNetV1 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileNetV1
    [google/mobilenet_v1_1.0_224](https://huggingface.co/google/mobilenet_v1_1.0_224) architecture.

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
        min_depth (`int`, *optional*, defaults to 8):
            All layers will have at least this many channels.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu6"`):
            The non-linear activation function (function or string) in the Transformer encoder and convolution layers.
        tf_padding (`bool`, *optional*, defaults to `True`):
            Whether to use TensorFlow padding rules on the convolution layers.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.999):
            The dropout ratio for attached classifiers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 0.001):
            The epsilon used by the layer normalization layers.

    Example:

    ```python
    >>> from transformers import MobileNetV1Config, MobileNetV1Model

    >>> # Initializing a "mobilenet_v1_1.0_224" style configuration
    >>> configuration = MobileNetV1Config()

    >>> # Initializing a model from the "mobilenet_v1_1.0_224" style configuration
    >>> model = MobileNetV1Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mobilenet_v1"

    def __init__(
        self,
        num_channels=3,
        image_size=224,
        depth_multiplier=1.0,
        min_depth=8,
        hidden_act="relu6",
        tf_padding=True,
        classifier_dropout_prob=0.999,
        initializer_range=0.02,
        layer_norm_eps=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be greater than zero.")

        self.num_channels = num_channels
        self.image_size = image_size
        self.depth_multiplier = depth_multiplier
        self.min_depth = min_depth
        self.hidden_act = hidden_act
        self.tf_padding = tf_padding
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class MobileNetV1OnnxConfig(OnnxConfig):
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


__all__ = ["MobileNetV1Config", "MobileNetV1OnnxConfig"]

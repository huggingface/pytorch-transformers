# coding=utf-8
# Copyright 2023 Google Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""EfficientNet model configuration"""

from collections import OrderedDict
from typing import List, Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class EfficientNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EfficientNetModel`]. It is used to instantiate an
    EfficientNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the EfficientNet
    [google/efficientnet-b7](https://huggingface.co/google/efficientnet-b7) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 600):
            The input image size.
        width_coefficient (`float`, *optional*, defaults to 2.0):
            Scaling coefficient for network width at each stage.
        depth_coefficient (`float`, *optional*, defaults to 3.1):
            Scaling coefficient for network depth at each stage.
        depth_divisor `int`, *optional*, defaults to 8):
            A unit of network width.
        kernel_sizes (`List[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`):
            List of kernel sizes to be used in each block.
        in_channels (`List[int]`, *optional*, defaults to `[32, 16, 24, 40, 80, 112, 192]`):
            List of input channel sizes to be used in each block for convolutional layers.
        out_channels (`List[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`):
            List of output channel sizes to be used in each block for convolutional layers.
        depthwise_padding (`List[int]`, *optional*, defaults to `[]`):
            List of block indices with square padding.
        strides (`List[int]`, *optional*, defaults to `[1, 2, 2, 2, 1, 2, 1]`):
            List of stride sizes to be used in each block for convolutional layers.
        num_block_repeats (`List[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`):
            List of the number of times each block is to repeated.
        expand_ratios (`List[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`):
            List of scaling coefficient of each block.
        squeeze_expansion_ratio (`float`, *optional*, defaults to 0.25):
            Squeeze expansion ratio.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
            `"selu", `"gelu_new"`, `"silu"` and `"mish"` are supported.
        hiddem_dim (`int`, *optional*, defaults to 1280):
            The hidden dimension of the layer before the classification head.
        pooling_type (`str` or `function`, *optional*, defaults to `"mean"`):
            Type of final pooling to be applied before the dense classification head. Available options are [`"mean"`,
            `"max"`]
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        batch_norm_eps (`float`, *optional*, defaults to 1e-3):
            The epsilon used by the batch normalization layers.
        batch_norm_momentum (`float`, *optional*, defaults to 0.99):
            The momentum used by the batch normalization layers.
        dropout_rate (`float`, *optional*, defaults to 0.5):
            The dropout rate to be applied before final classifier layer.
        drop_connect_rate (`float`, *optional*, defaults to 0.2):
            The drop rate for skip connections.

    Example:
    ```python
    >>> from transformers import EfficientNetConfig, EfficientNetModel

    >>> # Initializing a EfficientNet efficientnet-b7 style configuration
    >>> configuration = EfficientNetConfig()

    >>> # Initializing a model (with random weights) from the efficientnet-b7 style configuration
    >>> model = EfficientNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "efficientnet"

    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 600,
        width_coefficient: float = 2.0,
        depth_coefficient: float = 3.1,
        depth_divisor: int = 8,
        kernel_sizes: List[int] = [3, 3, 5, 3, 5, 5, 3],
        in_channels: List[int] = [32, 16, 24, 40, 80, 112, 192],
        out_channels: List[int] = [16, 24, 40, 80, 112, 192, 320],
        depthwise_padding: List[int] = [],
        strides: List[int] = [1, 2, 2, 2, 1, 2, 1],
        num_block_repeats: List[int] = [1, 2, 2, 3, 3, 4, 1],
        expand_ratios: List[int] = [1, 6, 6, 6, 6, 6, 6],
        squeeze_expansion_ratio: float = 0.25,
        hidden_act: str = "swish",
        hidden_dim: int = 2560,
        pooling_type: str = "mean",
        initializer_range: float = 0.02,
        batch_norm_eps: float = 0.001,
        batch_norm_momentum: float = 0.99,
        dropout_rate: float = 0.5,
        drop_connect_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.image_size = image_size
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.depth_divisor = depth_divisor
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_padding = depthwise_padding
        self.strides = strides
        self.num_block_repeats = num_block_repeats
        self.expand_ratios = expand_ratios
        self.squeeze_expansion_ratio = squeeze_expansion_ratio
        self.hidden_act = hidden_act
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.initializer_range = initializer_range
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.num_hidden_layers = sum(num_block_repeats) * 4


class EfficientNetOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-5


__all__ = ["EfficientNetConfig", "EfficientNetOnnxConfig"]

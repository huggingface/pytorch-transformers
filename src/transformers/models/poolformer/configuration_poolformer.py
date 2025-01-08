# coding=utf-8
# Copyright 2022 Sea AI Labs and The HuggingFace Inc. team. All rights reserved.
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
"""PoolFormer model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class PoolFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of [`PoolFormerModel`]. It is used to instantiate a
    PoolFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PoolFormer
    [sail/poolformer_s12](https://huggingface.co/sail/poolformer_s12) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels in the input image.
        patch_size (`int`, *optional*, defaults to 16):
            The size of the input patch.
        stride (`int`, *optional*, defaults to 16):
            The stride of the input patch.
        pool_size (`int`, *optional*, defaults to 3):
            The size of the pooling window.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the number of channels in the output of the MLP to the number of channels in the input.
        depths (`list`, *optional*, defaults to `[2, 2, 6, 2]`):
            The depth of each encoder block.
        hidden_sizes (`list`, *optional*, defaults to `[64, 128, 320, 512]`):
            The hidden sizes of each encoder block.
        patch_sizes (`list`, *optional*, defaults to `[7, 3, 3, 3]`):
            The size of the input patch for each encoder block.
        strides (`list`, *optional*, defaults to `[4, 2, 2, 2]`):
            The stride of the input patch for each encoder block.
        padding (`list`, *optional*, defaults to `[2, 1, 1, 1]`):
            The padding of the input patch for each encoder block.
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            The number of encoder blocks.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The dropout rate for the dropout layers.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function for the hidden layers.
        use_layer_scale (`bool`, *optional*, defaults to `True`):
            Whether to use layer scale.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-05):
            The initial value for the layer scale.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The initializer range for the weights.

    Example:

    ```python
    >>> from transformers import PoolFormerConfig, PoolFormerModel

    >>> # Initializing a PoolFormer sail/poolformer_s12 style configuration
    >>> configuration = PoolFormerConfig()

    >>> # Initializing a model (with random weights) from the sail/poolformer_s12 style configuration
    >>> model = PoolFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "poolformer"

    def __init__(
        self,
        num_channels=3,
        patch_size=16,
        stride=16,
        pool_size=3,
        mlp_ratio=4.0,
        depths=[2, 2, 6, 2],
        hidden_sizes=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        padding=[2, 1, 1, 1],
        num_encoder_blocks=4,
        drop_path_rate=0.0,
        hidden_act="gelu",
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        initializer_range=0.02,
        **kwargs,
    ):
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.hidden_sizes = hidden_sizes
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.num_encoder_blocks = num_encoder_blocks
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class PoolFormerOnnxConfig(OnnxConfig):
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
        return 2e-3


__all__ = ["PoolFormerConfig", "PoolFormerOnnxConfig"]

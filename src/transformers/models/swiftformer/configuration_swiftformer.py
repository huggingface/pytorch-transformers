# coding=utf-8
# Copyright 2023 MBZUAI and The HuggingFace Inc. team. All rights reserved.
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
"""SwiftFormer model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class SwiftFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SwiftFormerModel`]. It is used to instantiate an
    SwiftFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SwiftFormer
    [MBZUAI/swiftformer-xs](https://huggingface.co/MBZUAI/swiftformer-xs) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels
        depths (`List[int]`, *optional*, defaults to `[3, 3, 6, 4]`):
            Depth of each stage
        embed_dims (`List[int]`, *optional*, defaults to `[48, 56, 112, 220]`):
            The embedding dimension at each stage
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of size of the hidden dimensionality of an MLP to the dimensionality of its input.
        downsamples (`List[bool]`, *optional*, defaults to `[True, True, True, True]`):
            Whether or not to downsample inputs between two stages.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (string). `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        down_patch_size (`int`, *optional*, defaults to 3):
            The size of patches in downsampling layers.
        down_stride (`int`, *optional*, defaults to 2):
            The stride of convolution kernels in downsampling layers.
        down_pad (`int`, *optional*, defaults to 1):
            Padding in downsampling layers.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Rate at which to increase dropout probability in DropPath.
        drop_mlp_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate for the MLP component of SwiftFormer.
        drop_conv_encoder_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate for the ConvEncoder component of SwiftFormer.
        use_layer_scale (`bool`, *optional*, defaults to `True`):
            Whether to scale outputs from token mixers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-05):
            Factor by which outputs from token mixers are scaled.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.


    Example:

    ```python
    >>> from transformers import SwiftFormerConfig, SwiftFormerModel

    >>> # Initializing a SwiftFormer swiftformer-base-patch16-224 style configuration
    >>> configuration = SwiftFormerConfig()

    >>> # Initializing a model (with random weights) from the swiftformer-base-patch16-224 style configuration
    >>> model = SwiftFormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "swiftformer"

    def __init__(
        self,
        image_size=224,
        num_channels=3,
        depths=[3, 3, 6, 4],
        embed_dims=[48, 56, 112, 220],
        mlp_ratio=4,
        downsamples=[True, True, True, True],
        hidden_act="gelu",
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_path_rate=0.0,
        drop_mlp_rate=0.0,
        drop_conv_encoder_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        batch_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_channels = num_channels
        self.depths = depths
        self.embed_dims = embed_dims
        self.mlp_ratio = mlp_ratio
        self.downsamples = downsamples
        self.hidden_act = hidden_act
        self.down_patch_size = down_patch_size
        self.down_stride = down_stride
        self.down_pad = down_pad
        self.drop_path_rate = drop_path_rate
        self.drop_mlp_rate = drop_mlp_rate
        self.drop_conv_encoder_rate = drop_conv_encoder_rate
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.batch_norm_eps = batch_norm_eps


class SwiftFormerOnnxConfig(OnnxConfig):
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
        return 1e-4


__all__ = ["SwiftFormerConfig", "SwiftFormerOnnxConfig"]

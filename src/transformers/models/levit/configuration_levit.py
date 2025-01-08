# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""LeViT model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class LevitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LevitModel`]. It is used to instantiate a LeViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LeViT
    [facebook/levit-128S](https://huggingface.co/facebook/levit-128S) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size of the input image.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the initial convolution layers of patch embedding.
        stride (`int`, *optional*, defaults to 2):
            The stride size for the initial convolution layers of patch embedding.
        padding (`int`, *optional*, defaults to 1):
            The padding size for the initial convolution layers of patch embedding.
        patch_size (`int`, *optional*, defaults to 16):
            The patch size for embeddings.
        hidden_sizes (`List[int]`, *optional*, defaults to `[128, 256, 384]`):
            Dimension of each of the encoder blocks.
        num_attention_heads (`List[int]`, *optional*, defaults to `[4, 8, 12]`):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        depths (`List[int]`, *optional*, defaults to `[4, 4, 4]`):
            The number of layers in each encoder block.
        key_dim (`List[int]`, *optional*, defaults to `[16, 16, 16]`):
            The size of key in each of the encoder blocks.
        drop_path_rate (`int`, *optional*, defaults to 0):
            The dropout probability for stochastic depths, used in the blocks of the Transformer encoder.
        mlp_ratios (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        attention_ratios (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            Ratio of the size of the output dimension compared to input dimension of attention layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import LevitConfig, LevitModel

    >>> # Initializing a LeViT levit-128S style configuration
    >>> configuration = LevitConfig()

    >>> # Initializing a model (with random weights) from the levit-128S style configuration
    >>> model = LevitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "levit"

    def __init__(
        self,
        image_size=224,
        num_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
        patch_size=16,
        hidden_sizes=[128, 256, 384],
        num_attention_heads=[4, 8, 12],
        depths=[4, 4, 4],
        key_dim=[16, 16, 16],
        drop_path_rate=0,
        mlp_ratio=[2, 2, 2],
        attention_ratio=[2, 2, 2],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_sizes = hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.depths = depths
        self.key_dim = key_dim
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.attention_ratio = attention_ratio
        self.mlp_ratio = mlp_ratio
        self.initializer_range = initializer_range
        self.down_ops = [
            ["Subsample", key_dim[0], hidden_sizes[0] // key_dim[0], 4, 2, 2],
            ["Subsample", key_dim[0], hidden_sizes[1] // key_dim[0], 4, 2, 2],
        ]


# Copied from transformers.models.vit.configuration_vit.ViTOnnxConfig
class LevitOnnxConfig(OnnxConfig):
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


__all__ = ["LevitConfig", "LevitOnnxConfig"]

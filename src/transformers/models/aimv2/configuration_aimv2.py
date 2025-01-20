# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""AIMv2 model configuration"""

from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

__all__ = ["AIMv2Config"]


class AIMv2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of an [`AIMv2Model`].
    Instantiating a configuration with the defaults will yield a similar configuration
    to that of the [apple/aimv2-large-patch14-224](https://huggingface.co/apple/aimv2-large-patch14-224) architecture.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 2816):
            Dimension of the SwiGLU representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer
            in the Transformer.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            Image size.
        patch_size (`int`, *optional*, defaults to 14):
            Patch size.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon value used for the RMS normalization layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for attention probabilities.
        projection_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for the projection layer after the attention.
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries, keys and values.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias in the feed-forward and projection layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
        kwargs:
            Keyword arguments for the [`PretrainedConfig`].
    """

    model_type: str = "aimv2"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = False,
        use_bias: bool = False,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.projection_dropout = projection_dropout
        self.qkv_bias = qkv_bias
        self.use_bias = use_bias

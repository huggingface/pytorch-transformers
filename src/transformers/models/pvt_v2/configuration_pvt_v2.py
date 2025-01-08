# coding=utf-8
# Copyright 2024 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
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
"""Pvt V2 model configuration"""

from typing import Callable, List, Tuple, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices


logger = logging.get_logger(__name__)


class PvtV2Config(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PvtV2Model`]. It is used to instantiate a Pvt V2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Pvt V2 B0
    [OpenGVLab/pvt_v2_b0](https://huggingface.co/OpenGVLab/pvt_v2_b0) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`Union[int, Tuple[int, int]]`, *optional*, defaults to 224):
            The input image size. Pass int value for square image, or tuple of (height, width).
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_encoder_blocks (`[int]`, *optional*, defaults to 4):
            The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
        depths (`List[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
            The number of layers in each encoder block.
        sr_ratios (`List[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
            Spatial reduction ratios in each encoder block.
        hidden_sizes (`List[int]`, *optional*, defaults to `[32, 64, 160, 256]`):
            Dimension of each of the encoder blocks.
        patch_sizes (`List[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
            Patch size for overlapping patch embedding before each encoder block.
        strides (`List[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
            Stride for overlapping patch embedding before each encoder block.
        num_attention_heads (`List[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        mlp_ratios (`List[int]`, *optional*, defaults to `[8, 8, 4, 4]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not a learnable bias should be added to the queries, keys and values.
        linear_attention (`bool`, *optional*, defaults to `False`):
            Use linear attention complexity. If set to True, `sr_ratio` is ignored and average pooling is used for
            dimensionality reduction in the attention layers rather than strided convolution.
        out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage.
    Example:

    ```python
    >>> from transformers import PvtV2Model, PvtV2Config

    >>> # Initializing a pvt_v2_b0 style configuration
    >>> configuration = PvtV2Config()

    >>> # Initializing a model from the OpenGVLab/pvt_v2_b0 style configuration
    >>> model = PvtV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pvt_v2"

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        num_channels: int = 3,
        num_encoder_blocks: int = 4,
        depths: List[int] = [2, 2, 2, 2],
        sr_ratios: List[int] = [8, 4, 2, 1],
        hidden_sizes: List[int] = [32, 64, 160, 256],
        patch_sizes: List[int] = [7, 3, 3, 3],
        strides: List[int] = [4, 2, 2, 2],
        num_attention_heads: List[int] = [1, 2, 5, 8],
        mlp_ratios: List[int] = [8, 8, 4, 4],
        hidden_act: Union[str, Callable] = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        drop_path_rate: float = 0.0,
        layer_norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        linear_attention: bool = False,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size

        self.image_size = image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.hidden_sizes = hidden_sizes
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.mlp_ratios = mlp_ratios
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.linear_attention = linear_attention
        self.stage_names = [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )


__all__ = ["PvtV2Config"]

# coding=utf-8
# Copyright 2024 Meta Platforms, Inc. and affiliates, and the HuggingFace Inc. team. All rights reserved.
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
"""Mimi model configuration"""

import math

import numpy as np

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MimiConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`MimiModel`]. It is used to instantiate a
    Mimi model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [kyutai/mimi](https://huggingface.co/kyutai/mimi) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        frame_rate (`float`, *optional*, defaults to 12.5):
            Framerate of the model.
        audio_channels (`int`, *optional*, defaults to 1):
            Number of channels in the audio data. Either 1 for mono or 2 for stereo.
        hidden_size (`int`, *optional*, defaults to 512):
            Intermediate representation dimension.
        num_filters (`int`, *optional*, defaults to 64):
            Number of convolution kernels of first `MimiConv1d` down sampling layer.
        num_residual_layers (`int`,  *optional*, defaults to 1):
            Number of residual layers.
        upsampling_ratios (`Sequence[int]`, *optional*):
            Kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it
            will use the ratios in the reverse order to the ones specified here that must match the decoder order.
            If not specified, will defaults to `[8, 6, 5, 4]`
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for the initial convolution.
        last_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size for the last convolution layer.
        residual_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size for the residual layers.
        dilation_growth_rate (`int`, *optional*, defaults to 2):
            How much to increase the dilation with each layer.
        use_causal_conv (`bool`, *optional*, defaults to `True`):
            Whether to use fully causal convolution.
        pad_mode (`str`, *optional*, defaults to `"constant"`):
            Padding mode for the convolutions.
        compress (`int`, *optional*, defaults to 2):
            Reduced dimensionality in residual branches.
        trim_right_ratio (`float`, *optional*, defaults to 1.0):
            Ratio for trimming at the right of the transposed convolution under the `use_causal_conv = True` setup. If
            equal to 1.0, it means that all the trimming is done at the right.
        codebook_size (`int`, *optional*, defaults to 2048):
            Number of discret codes in each codebooks.
        codebook_dim (`int`, *optional*, defaults to 256):
            Dimension of the unquantized codebook vectors. If not defined, uses `hidden_size`.
        num_quantizers (`int`, *optional*, defaults to 32):
            Number of quantizer channels, or codebooks, in the quantizer.
        use_conv_shortcut (`bool`, *optional*, defaults to `False`):
            Whether to use a convolutional layer as the 'skip' connection in the `MimiResnetBlock` block. If False,
            an identity function will be used, giving a generic residual connection.
        vector_quantization_hidden_dimension (`int`, *optional*, defaults to 256):
            Intermediate representation dimension in the residual vector quantization space.
        num_semantic_quantizers (`int`, *optional*, defaults to 1):
            Number of semantic quantizer channels, or codebooks, in the semantic quantizer. Must be lower than `num_quantizers`.
        upsample_groups (`int`, *optional*, defaults to 512):
            If `frame_rate!=encodec_frame_rate`, indicates the number of groups used in the upsampling operation to go from one rate to another.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the Transformer models.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP representations.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        head_dim (`int`, *optional*, defaults to `hidden_size // num_attention_heads`):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8000):
            The maximum sequence length that this model might ever be used with. Mimi's sliding window attention
            allows sequence of up to 8000 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the LayerNorm normalization layers.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        sliding_window (`int`, *optional*, defaults to 250):
            Sliding window attention window size. If not specified, will default to `250`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layer_scale_initial_scale (`float`, *optional*, defaults to 0.01):
            Initiale scale of the residual rescaling operation done in the Transformer models.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
    Example:

    ```python
    >>> from transformers import MimiModel, MimiConfig

    >>> # Initializing a "kyutai/mimi" style configuration
    >>> configuration = MimiConfig()

    >>> # Initializing a model (with random weights) from the "kyutai/mimi" style configuration
    >>> model = MimiModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mimi"

    def __init__(
        self,
        sampling_rate=24_000,
        frame_rate=12.5,
        audio_channels=1,
        hidden_size=512,
        num_filters=64,
        num_residual_layers=1,
        upsampling_ratios=None,
        kernel_size=7,
        last_kernel_size=3,
        residual_kernel_size=3,
        dilation_growth_rate=2,
        use_causal_conv=True,
        pad_mode="constant",
        compress=2,
        trim_right_ratio=1.0,
        codebook_size=2048,
        codebook_dim=256,
        num_quantizers=32,
        use_conv_shortcut=False,
        vector_quantization_hidden_dimension=256,
        num_semantic_quantizers=1,
        upsample_groups=512,
        num_hidden_layers=8,
        intermediate_size=2048,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="gelu",
        max_position_embeddings=8000,
        initializer_range=0.02,
        norm_eps=1e-5,
        use_cache=False,
        rope_theta=10000.0,
        sliding_window=250,
        attention_dropout=0.0,
        layer_scale_initial_scale=0.01,
        attention_bias=False,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.frame_rate = frame_rate
        self.audio_channels = audio_channels
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_residual_layers = num_residual_layers
        self.upsampling_ratios = upsampling_ratios if upsampling_ratios else [8, 6, 5, 4]
        self.kernel_size = kernel_size
        self.last_kernel_size = last_kernel_size
        self.residual_kernel_size = residual_kernel_size
        self.dilation_growth_rate = dilation_growth_rate
        self.use_causal_conv = use_causal_conv
        self.pad_mode = pad_mode
        self.compress = compress
        self.trim_right_ratio = trim_right_ratio
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim if codebook_dim is not None else hidden_size
        self.num_quantizers = num_quantizers
        self.use_conv_shortcut = use_conv_shortcut
        self.vector_quantization_hidden_dimension = vector_quantization_hidden_dimension
        self.upsample_groups = upsample_groups
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.layer_scale_initial_scale = layer_scale_initial_scale
        self.attention_bias = attention_bias

        if num_semantic_quantizers >= self.num_quantizers:
            raise ValueError(
                f"The number of semantic quantizers should be lower than the total number of quantizers {self.num_quantizers}, but is currently {num_semantic_quantizers}."
            )
        self.num_semantic_quantizers = num_semantic_quantizers
        super().__init__(**kwargs)

    @property
    def encodec_frame_rate(self) -> int:
        hop_length = np.prod(self.upsampling_ratios)
        return math.ceil(self.sampling_rate / hop_length)

    @property
    def num_codebooks(self) -> int:
        # alias to num_quantizers
        return self.num_quantizers


__all__ = ["MimiConfig"]

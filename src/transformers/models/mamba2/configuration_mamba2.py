# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""MAMBA2 configuration"""

import math

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Mamba2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Mamba2Model`]. It is used to instantiate a MAMBA2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MAMBA2
    [state-spaces/mamba2-2.8b](https://huggingface.co/state-spaces/mamba2-2.8b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_heads (`int`, *optional*, defaults to 128):
            Number of heads for the evolution matrices of mamba 2.
        head_dim (`int`, *optional*, defaults to 64):
            Dimension of each head.
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the MAMBA2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Mamba2Model`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        state_size (`int`, *optional*, defaults to 128): shape of the state space latents.
        num_hidden_layers (`int`, *optional*, defaults to 64):
            Number of hidden layers in the model.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end of sentence token in the vocabulary.
        expand (`int`, *optional*, defaults to 2): Expanding factor used to determine the intermediate size.
        conv_kernel (`int`, *optional*, defaults to 4): Size of the convolution kernel.
        n_groups (`int`, *optional*, defaults to 8):
            Number of groups for the evolution matrices of mamba 2.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the convolution layer of the mixer block.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.1):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        residual_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
        time_step_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj.bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj.bias`.
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        time_step_limit (`tuple`, *optional*, defaults to `(0.0, inf)`):
            Accepted range of time step values.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
            Whether or not to rescale `out_proj` weights when initializing.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the cache should be used.
        rms_norm (`bool`, *optional*, defaults to `True`):
            Whether to use RMS norm or not.
        chunk_size (`int`, *optional*, defaults to 256):
            Size of the chunks that will comprise the sequence.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embeddings or not.


    Example:

    ```python
    >>> from transformers import Mamba2Config, Mamba2Model

    >>> # Initializing a Mamba2 configuration
    >>> configuration = Mamba2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Mamba2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mamba2"

    def __init__(
        self,
        num_heads=128,
        head_dim=64,
        vocab_size=32768,
        hidden_size=4096,
        state_size=128,
        num_hidden_layers=64,
        layer_norm_epsilon=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        expand=2,
        conv_kernel=4,
        n_groups=8,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        time_step_limit=(0.0, float("inf")),
        rescale_prenorm_residual=False,
        use_cache=True,
        rms_norm=True,
        chunk_size=256,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.n_groups = n_groups
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rms_norm = rms_norm
        self.state_size = state_size
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Mamba2Config"]

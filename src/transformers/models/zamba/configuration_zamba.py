# coding=utf-8
# Copyright 2024 Zyphra Technologies and the HuggingFace Inc. team. All rights reserved.
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
"""Zamba model configuration"""

import math

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class ZambaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ZambaModel`]. It is used to instantiate a
    Zamba model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Zamba-v0.1 model.

    [Zyphra/Zamba-7B-v1](https://huggingface.co/Zyphra/Zamba-7B-v1)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Zamba model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ZambaModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        hidden_size (`int`, *optional*, defaults to 3712):
            Dimension of the hidden representations.
        attention_hidden_size (`int`, *optional*):
            Dimension of the hidden representations of the inputs to the Attention layer.
        intermediate_size (`int`, *optional*, defaults to 14848):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 76):
            Number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        attention_head_dim (`int`, *optional*):
            Dimension of the attention head in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=None`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf).
        n_mamba_heads (`int`, *optional*, defaults to 2):
            Number of mamba heads for each mamba layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder.
        hidden_mamba_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the mamba layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
            integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
            logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
            sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
            significantly.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            This value doesn't have any real effect. The maximum sequence length that this model is intended to be
            used with. It can be used with longer sequences, but performance may degrade.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attn_layer_period (`int`, *optional*, defaults to 6):
            Once in this many layers, we will have a shared attention layer
        attn_layer_offset (`int`, *optional*, defaults to 4):
            Offset of the shared attention layer
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
            `causal-conv1d` are installed, and the mamba modules are running on a CUDA device. Raises ValueError if
            `True` and kernels are not available
        mamba_d_state (`int`, *optional*, defaults to 16):
            The dimension the mamba state space latents
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor (relative to hidden_size) used to determine the mamba intermediate size
        mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the mamba discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj_bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj_bias`.
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block

    """

    model_type = "zamba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        tie_word_embeddings=True,
        hidden_size=3712,
        attention_hidden_size=None,
        intermediate_size=14848,
        num_hidden_layers=76,
        num_attention_heads=16,
        attention_head_dim=None,
        num_key_value_heads=16,
        n_mamba_heads=2,
        hidden_act="gelu",
        hidden_mamba_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=4096,
        attention_dropout=0.0,
        attn_layer_period=6,
        attn_layer_offset=4,
        use_mamba_kernels=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        if attention_hidden_size is None:
            self.attention_hidden_size = 2 * hidden_size
        else:
            self.attention_hidden_size = attention_hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if attention_head_dim is None:
            self.attention_head_dim = 2 * self.hidden_size // self.num_attention_heads
        else:
            self.attention_head_dim = attention_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout

        self.num_key_value_heads = num_key_value_heads
        self.n_mamba_heads = n_mamba_heads
        self.hidden_act = hidden_act
        self.hidden_mamba_act = hidden_mamba_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep

        self.attn_layer_period = attn_layer_period
        self.attn_layer_offset = attn_layer_offset

        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        self.layers_block_type = self._layers_block_type(num_hidden_layers, attn_layer_period, attn_layer_offset)

        assert (
            self.mamba_expand * self.hidden_size
        ) % self.n_mamba_heads == 0, "`intermediate_size` should be divisible by `n_mamba_heads`."

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _layers_block_type(self, num_hidden_layers, attn_layer_period, attn_layer_offset):
        layers = [
            "mamba",
            "mamba",
            "hybrid",
        ] + ["hybrid" if i % attn_layer_period == attn_layer_offset else "mamba" for i in range(num_hidden_layers - 3)]
        return layers

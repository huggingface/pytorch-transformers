# coding=utf-8
# Copyright 2024 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""Bamba model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class BambaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BambaModel`]. It is used to instantiate a
    BambaModel model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with defaults taken from [ibm-fms/Bamba-9.8b-2.2T-hf](https://huggingface.co/ibm-fms/Bamba-9.8b-2.2T-hf).

    The BambaModel is a hybrid [mamba2](https://github.com/state-spaces/mamba) architecture with SwiGLU.
    The checkpoints are  jointly trained by IBM, Princeton, and UIUC.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 128000):
            Vocabulary size of the Bamba model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BambaModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
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
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            Max cached sequence length for the model
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attn_layer_indices (`list`, *optional*):
            Specifies the layer indices that will have full attention. Must contain values at most num_hidden_layers.
        mamba_n_heads (`int`, *optional*, defaults to 128):
            The number of mamba heads used in the v2 implementation.
        mamba_d_head (`int`, *optional*, defaults to `"auto"`):
            Head embeddding dimension size
        mamba_n_groups (`int`, *optional*, defaults to 1):
            The number of the mamba groups used in the v2 implementation.
        mamba_d_state (`int`, *optional*, defaults to 256):
            The dimension the mamba state space latents
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor (relative to hidden_size) used to determine the mamba intermediate size
        mamba_chunk_size (`int`, *optional*, defaults to 256):
            The chunks in which to break the sequence when doing prefill/training
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block

    """

    model_type = "bamba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=128000,
        tie_word_embeddings=False,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=262144,
        attention_dropout=0.0,
        attn_layer_indices=None,
        mamba_n_heads=128,
        mamba_d_head="auto",
        mamba_n_groups=1,
        mamba_d_state=256,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=256,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.attention_bias = False
        self.mlp_bias = False

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep

        self.attn_layer_indices = attn_layer_indices
        self.rope_theta = 10000.0
        self.rope_scaling = None
        self.partial_rotary_factor = 0.5

        mamba_intermediate = mamba_expand * hidden_size

        if mamba_intermediate % mamba_n_heads != 0:
            raise ValueError("mamba_n_heads must divide mamba_expand * hidden_size")

        # for the mamba_v2, must satisfy the following
        if mamba_d_head == "auto":
            mamba_d_head = mamba_intermediate // mamba_n_heads

        if mamba_d_head * mamba_n_heads != mamba_intermediate:
            raise ValueError("The dimensions for the Mamba head state do not match the model intermediate_size")

        self.mamba_n_heads = mamba_n_heads
        self.mamba_d_head = mamba_d_head
        self.mamba_n_groups = mamba_n_groups
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return [
            "attention" if (self.attn_layer_indices and i in self.attn_layer_indices) else "mamba"
            for i in range(self.num_hidden_layers)
        ]


__all__ = ["BambaConfig"]

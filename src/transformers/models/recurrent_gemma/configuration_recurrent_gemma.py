# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
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
"""RecurrentGemma model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class RecurrentGemmaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RecurrentGemmaModel`]. It is used to instantiate a RecurrentGemma
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RecurrentGemma-7B.

    e.g. [google/recurrentgemma-2b](https://huggingface.co/google/recurrentgemma-2b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_hidden_layers (`int`, *optional*, defaults to 26):
            The number of hidden layers in the model.
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the RecurrentGemma model. Defines the number of
            different tokens that can be represented by the
            `inputs_ids` passed when calling [`RecurrentGemmaModel`]
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 7680):
            Dimension of the MLP representations.
        num_attention_heads (`int`, *optional*, defaults to 10):
            The number of heads for the attention block and the number of
            heads/blocks for the block-diagonal layers used in the RG-LRU gates.
            This number must divide `hidden_size` and `lru_width`.
        lru_width (`int` or `None`, *optional*):
            Dimension of the hidden representations of the RG-LRU. If `None`
            this will be set to `hidden_size`.
            Whether to scale the output of the embeddings by `sqrt(hidden_size)`.
        attention_window_size (`int`, *optional*, defaults to 2048):
            The size of the attention window used in the attention block.
        conv1d_width (`int`, *optional*, defaults to 4):
            The kernel size of conv1d layers used in the recurrent blocks.
        logits_soft_cap (`float`, *optional*, defaults to 30.0):
            The value at which the logits should be soft-capped to after the transformer and LM-head computation in the Causal LM architecture.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values
            attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        hidden_activation (``str` or `function``, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The hidden activation used in the recurrent block as well as the MLP layer of the decoder layers.
        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
            The partial rotary factor used in the initialization of the rotary embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        block_types (`List[str]`, *optional*, defaults to `('recurrent', 'recurrent', 'attention')`):
            List of aleternating blocks that will be repeated to initialize the `temporal_block` layer.
        attention_dropout (`float`, *optional*, defaults to 0.0): dropout value to use after the attention softmax.
        num_key_value_heads (`16`, *optional*, defaults to 16): Number of key value heads to use GQA.
        attention_bias (`bool`, *optional*, defaults to `False`): whether or not the linear q,k,v of the Attention layer should have bias
        w_init_variance_scale (`float`, *optional*, defaults to 0.01): weight initialization variance.
    ```python
    >>> from transformers import RecurrentGemmaModel, RecurrentGemmaConfig

    >>> # Initializing a RecurrentGemma recurrentgemma-2b style configuration
    >>> configuration = RecurrentGemmaConfig()

    >>> # Initializing a model from the recurrentgemma-2b style configuration
    >>> model = RecurrentGemmaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "recurrent_gemma"

    def __init__(
        self,
        num_hidden_layers=26,
        vocab_size=256000,
        hidden_size=2560,
        intermediate_size=3 * 2560,
        num_attention_heads=10,
        lru_width=None,
        attention_window_size=2048,
        conv1d_width=4,
        logits_soft_cap=30.0,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        hidden_activation="gelu_pytorch_tanh",
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        block_types=("recurrent", "recurrent", "attention"),
        attention_dropout=0.0,
        num_key_value_heads=None,
        attention_bias=False,
        w_init_variance_scale=0.01,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.lru_width = lru_width if lru_width is not None else hidden_size
        self.attention_window_size = attention_window_size
        self.conv1d_width = conv1d_width
        self.logits_soft_cap = logits_soft_cap
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.block_types = list(block_types)
        self.hidden_activation = hidden_activation
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("The number of `num_key_value_heads` must be smaller than `num_attention_heads`")
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.w_init_variance_scale = w_init_variance_scale
        self.final_w_init_variance_scale = 2.0 / self.num_hidden_layers
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return (self.block_types * 100)[: self.num_hidden_layers]


__all__ = ["RecurrentGemmaConfig"]

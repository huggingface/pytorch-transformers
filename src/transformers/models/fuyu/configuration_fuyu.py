# coding=utf-8
# Copyright 2023 Adept AI and the HuggingFace Inc. team. All rights reserved.
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
"""Fuyu model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class FuyuConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FuyuForCausalLM`]. It is used to instantiate an
    Fuyu model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [adept/fuyu-8b](https://huggingface.co/adept/fuyu-8b).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 262144):
            Vocabulary size of the Fuyu model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FuyuForCausalLM`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 16384):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 36):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu2"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 16384):
            The maximum sequence length that this model might ever be used with.
        image_size (`int`, *optional*, defaults to 300):
            The input image size.
        patch_size (`int`, *optional*, defaults to 30):
            The input vision transformer encoding patch size.
        num_channels (`int`, *optional*, defaults to 3):
            The input image number of channels.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`. Whether to tie weight embeddings
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 25000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalFuyu/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        qk_layernorm (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the Queries and Keys after projecting the hidden states
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio after applying the MLP to the hidden states.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio after computing the attention scores.
        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
            Percentage of the query and keys which will have rotary embedding.

        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`Union[int, List[int]]`, *optional*, defaults to 2):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize the `language``[`Aut`].

    ```python
    >>> from transformers import FuyuConfig

    >>> # Initializing a Fuyu fuyu-7b style configuration
    >>> configuration = FuyuConfig()
    ```"""

    model_type = "fuyu"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=36,
        num_attention_heads=64,
        hidden_act="relu2",
        max_position_embeddings=16384,
        image_size=300,
        patch_size=30,
        num_channels=3,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=25000.0,
        rope_scaling=None,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        partial_rotary_factor=0.5,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        text_config=None,
        **kwargs,
    ):
        if text_config is None:
            text_config = {
                "vocab_size": vocab_size,
                "max_position_embeddings": max_position_embeddings,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads,
                "hidden_act": hidden_act,
                "initializer_range": initializer_range,
                "layer_norm_eps": layer_norm_eps,
                "use_cache": use_cache,
                "rope_theta": rope_theta,
                "rope_scaling": rope_scaling,
                "qk_layernorm": qk_layernorm,
                "hidden_dropout": hidden_dropout,
                "attention_dropout": attention_dropout,
                "partial_rotary_factor": partial_rotary_factor,
                "pad_token_id": pad_token_id,
                "bos_token_id": bos_token_id,
                "eos_token_id": eos_token_id,
                "tie_word_embeddings": tie_word_embeddings,
            }
            logger.info("text_config is None. initializing the text model with default values.")
        text_model_type = text_config["model_type"] if "model_type" in text_config else "persimmon"
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        self._vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.qk_layernorm = qk_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.partial_rotary_factor = partial_rotary_factor
        self._rope_scaling_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with two fields, `type` and `factor`, " f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")


__all__ = ["FuyuConfig"]

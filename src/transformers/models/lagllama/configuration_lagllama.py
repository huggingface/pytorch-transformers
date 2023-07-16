# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" LLaMA model configuration"""
from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LagLlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LagLlamaModel`]. It is used to instantiate an LagLLaMA
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        input_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import LagLlamaModel, LagLlamaConfig

    >>> # Initializing a LagLLaMA configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LagLlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "lagllama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        distribution_output: str = "student_t",
        loss: str = "nll",
        input_size: int = 1,
        lags_sequence: List[int] = [1, 2, 3, 4, 5, 6, 7],
        scaling: str = "mean",
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_context_length=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_scaling=None,
        **kwargs,
    ):
        self.distribution_output = distribution_output
        self.loss = loss
        self.input_size = input_size
        self.scaling = scaling
        self.lags_sequence = lags_sequence
        self.max_position_embeddings = max_context_length
        self.hidden_size = hidden_size
        self.feature_size = input_size * len(lags_sequence) + self._number_of_features
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        super().__init__(**kwargs)

    @property
    def _number_of_features(self) -> int:
        # the log1p(abs(loc)) and log(scale) features of the context window
        return self.input_size * 2

    # Copied from transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")

# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
"""CPMAnt model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class CpmAntConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CpmAntModel`]. It is used to instantiate an
    CPMAnt model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CPMAnt
    [openbmb/cpm-ant-10b](https://huggingface.co/openbmb/cpm-ant-10b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30720):
            Vocabulary size of the CPMAnt model. Defines the number of different tokens that can be represented by the
            `input` passed when calling [`CpmAntModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the encoder layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads in the Transformer encoder.
        dim_head (`int`, *optional*, defaults to 128):
            Dimension of attention heads for each attention layer in the Transformer encoder.
        dim_ff (`int`, *optional*, defaults to 10240):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of layers of the Transformer encoder.
        dropout_p (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder.
        position_bias_num_buckets (`int`, *optional*, defaults to 512):
            The number of position_bias buckets.
        position_bias_max_distance (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        init_std (`float`, *optional*, defaults to 1.0):
            Initialize parameters with std = init_std.
        prompt_types (`int`, *optional*, defaults to 32):
            The type of prompt.
        prompt_length (`int`, *optional*, defaults to 32):
            The length of prompt.
        segment_types (`int`, *optional*, defaults to 32):
            The type of segment.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use cache.

    Example:

    ```python
    >>> from transformers import CpmAntModel, CpmAntConfig

    >>> # Initializing a CPMAnt cpm-ant-10b style configuration
    >>> configuration = CpmAntConfig()

    >>> # Initializing a model from the cpm-ant-10b style configuration
    >>> model = CpmAntModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "cpmant"

    def __init__(
        self,
        vocab_size: int = 30720,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        dim_head: int = 128,
        dim_ff: int = 10240,
        num_hidden_layers: int = 48,
        dropout_p: int = 0.0,
        position_bias_num_buckets: int = 512,
        position_bias_max_distance: int = 2048,
        eps: int = 1e-6,
        init_std: float = 1.0,
        prompt_types: int = 32,
        prompt_length: int = 32,
        segment_types: int = 32,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt_types = prompt_types
        self.prompt_length = prompt_length
        self.segment_types = segment_types
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_hidden_layers = num_hidden_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.eps = eps
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.init_std = init_std


__all__ = ["CpmAntConfig"]

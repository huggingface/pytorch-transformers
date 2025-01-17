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
"""minGRU configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MinGRUConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MinGRUModel`]. It is used to instantiate a MinGRU
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50280):
            Vocabulary size of the MinGRU model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MinGRUModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the model.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout of MinGRU hidden layers
        classifier_dropout (`float`, *optional*, defaults to 0.1):
            Dropout of classification head
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-12):
            The epsilon to use in the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the end of sentence token in the vocabulary.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in ["in_proj", "out_proj"] of the linear layers
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the cache should be used.
        pooling_type (`str`, *optional*, defaults to `last`):
            How to pool the hidden states of the last layer
        pooling_activation (`str`, *optional*, defaults to `silu`):
            Activation function applied to the projection of the pooled hidden states
        g_epsilon (`float`, *optional*, defaults to 0.5):
            Epsilon of the g activation function. See appendix B.2.1 for more details

    Example:

    ```python
    >>> from transformers import MinGRUConfig, MinGRUModel

    >>> # Initializing a MinGRU configuration
    >>> configuration = MinGRUConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MinGRUModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mingru"

    def __init__(
        self,
        vocab_size=50280,
        hidden_size=768,
        num_hidden_layers=12,
        dropout=0.1,
        classifier_dropout=0.1,
        layer_norm_epsilon=1e-12,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        use_bias=False,
        initializer_range=0.02,
        use_cache=True,
        pooling_type="last",
        pooling_activation="silu",
        g_epsilon=0.5,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pooling_type = pooling_type
        self.pooling_activation = pooling_activation
        self.g_epsilon = g_epsilon
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)

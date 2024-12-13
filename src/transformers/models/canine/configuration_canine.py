# coding=utf-8
# Copyright Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""CANINE model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class CanineConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CanineModel`]. It is used to instantiate an
    CANINE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CANINE
    [google/canine-s](https://huggingface.co/google/canine-s) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the deep Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoders.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoders.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoders, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 16384):
            The maximum sequence length that this model might ever be used with.
        type_vocab_size (`int`, *optional*, defaults to 16):
            The vocabulary size of the `token_type_ids` passed when calling [`CanineModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 57344):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 57345):
            End of stream token id.
        downsampling_rate (`int`, *optional*, defaults to 4):
            The rate at which to downsample the original character sequence length before applying the deep Transformer
            encoder.
        upsampling_kernel_size (`int`, *optional*, defaults to 4):
            The kernel size (i.e. the number of characters in each window) of the convolutional projection layer when
            projecting back from `hidden_size`*2 to `hidden_size`.
        num_hash_functions (`int`, *optional*, defaults to 8):
            The number of hash functions to use. Each hash function has its own embedding matrix.
        num_hash_buckets (`int`, *optional*, defaults to 16384):
            The number of hash buckets to use.
        local_transformer_stride (`int`, *optional*, defaults to 128):
            The stride of the local attention of the first shallow Transformer encoder. Defaults to 128 for good
            TPU/XLA memory alignment.

    Example:

    ```python
    >>> from transformers import CanineConfig, CanineModel

    >>> # Initializing a CANINE google/canine-s style configuration
    >>> configuration = CanineConfig()

    >>> # Initializing a model (with random weights) from the google/canine-s style configuration
    >>> model = CanineModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "canine"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=16384,
        type_vocab_size=16,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=0xE000,
        eos_token_id=0xE001,
        downsampling_rate=4,
        upsampling_kernel_size=4,
        num_hash_functions=8,
        num_hash_buckets=16384,
        local_transformer_stride=128,  # Good TPU/XLA memory alignment.
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps

        # Character config:
        self.downsampling_rate = downsampling_rate
        self.upsampling_kernel_size = upsampling_kernel_size
        self.num_hash_functions = num_hash_functions
        self.num_hash_buckets = num_hash_buckets
        self.local_transformer_stride = local_transformer_stride


__all__ = ["CanineConfig"]

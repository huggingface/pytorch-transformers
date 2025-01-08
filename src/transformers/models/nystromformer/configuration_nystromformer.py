# coding=utf-8
# Copyright 2022 UW-Madison and The HuggingFace Inc. team. All rights reserved.
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
"""Nystromformer model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class NystromformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NystromformerModel`]. It is used to instantiate
    an Nystromformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Nystromformer
    [uw-madison/nystromformer-512](https://huggingface.co/uw-madison/nystromformer-512) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30000):
            Vocabulary size of the Nystromformer model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`NystromformerModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`NystromformerModel`].
        segment_means_seq_len (`int`, *optional*, defaults to 64):
            Sequence length used in segment-means.
        num_landmarks (`int`, *optional*, defaults to 64):
            The number of landmark (or Nystrom) points to use in Nystrom approximation of the softmax self-attention
            matrix.
        conv_kernel_size (`int`, *optional*, defaults to 65):
            The kernel size of depthwise convolution used in Nystrom approximation.
        inv_coeff_init_option (`bool`, *optional*, defaults to `False`):
            Whether or not to use exact coefficient computation for the initial values for the iterative method of
            calculating the Moore-Penrose inverse of a matrix.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.

    Example:

    ```python
    >>> from transformers import NystromformerModel, NystromformerConfig

    >>> # Initializing a Nystromformer uw-madison/nystromformer-512 style configuration
    >>> configuration = NystromformerConfig()

    >>> # Initializing a model from the uw-madison/nystromformer-512 style configuration
    >>> model = NystromformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "nystromformer"

    def __init__(
        self,
        vocab_size=30000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=510,
        type_vocab_size=2,
        segment_means_seq_len=64,
        num_landmarks=64,
        conv_kernel_size=65,
        inv_coeff_init_option=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
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
        self.segment_means_seq_len = segment_means_seq_len
        self.num_landmarks = num_landmarks
        self.conv_kernel_size = conv_kernel_size
        self.inv_coeff_init_option = inv_coeff_init_option
        self.layer_norm_eps = layer_norm_eps
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


__all__ = ["NystromformerConfig"]

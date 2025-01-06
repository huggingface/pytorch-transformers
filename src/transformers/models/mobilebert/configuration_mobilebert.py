# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""MobileBERT model configuration"""

from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MobileBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MobileBertModel`] or a [`TFMobileBertModel`]. It
    is used to instantiate a MobileBERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the MobileBERT
    [google/mobilebert-uncased](https://huggingface.co/google/mobilebert-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the MobileBERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`MobileBertModel`] or [`TFMobileBertModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 512):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`MobileBertModel`] or
            [`TFMobileBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.

        pad_token_id (`int`, *optional*, defaults to 0):
            The ID of the token in the word embedding to use as padding.
        embedding_size (`int`, *optional*, defaults to 128):
            The dimension of the word embedding vectors.
        trigram_input (`bool`, *optional*, defaults to `True`):
            Use a convolution of trigram as input.
        use_bottleneck (`bool`, *optional*, defaults to `True`):
            Whether to use bottleneck in BERT.
        intra_bottleneck_size (`int`, *optional*, defaults to 128):
            Size of bottleneck layer output.
        use_bottleneck_attention (`bool`, *optional*, defaults to `False`):
            Whether to use attention inputs from the bottleneck transformation.
        key_query_shared_bottleneck (`bool`, *optional*, defaults to `True`):
            Whether to use the same linear transformation for query&key in the bottleneck.
        num_feedforward_networks (`int`, *optional*, defaults to 4):
            Number of FFNs in a block.
        normalization_type (`str`, *optional*, defaults to `"no_norm"`):
            The normalization type in MobileBERT.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Examples:

    ```python
    >>> from transformers import MobileBertConfig, MobileBertModel

    >>> # Initializing a MobileBERT configuration
    >>> configuration = MobileBertConfig()

    >>> # Initializing a model (with random weights) from the configuration above
    >>> model = MobileBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "mobilebert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=512,
        num_hidden_layers=24,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_act="relu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        embedding_size=128,
        trigram_input=True,
        use_bottleneck=True,
        intra_bottleneck_size=128,
        use_bottleneck_attention=False,
        key_query_shared_bottleneck=True,
        num_feedforward_networks=4,
        normalization_type="no_norm",
        classifier_activation=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_size = embedding_size
        self.trigram_input = trigram_input
        self.use_bottleneck = use_bottleneck
        self.intra_bottleneck_size = intra_bottleneck_size
        self.use_bottleneck_attention = use_bottleneck_attention
        self.key_query_shared_bottleneck = key_query_shared_bottleneck
        self.num_feedforward_networks = num_feedforward_networks
        self.normalization_type = normalization_type
        self.classifier_activation = classifier_activation

        if self.use_bottleneck:
            self.true_hidden_size = intra_bottleneck_size
        else:
            self.true_hidden_size = hidden_size

        self.classifier_dropout = classifier_dropout


# Copied from transformers.models.bert.configuration_bert.BertOnnxConfig with Bert->MobileBert
class MobileBertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )


__all__ = ["MobileBertConfig", "MobileBertOnnxConfig"]

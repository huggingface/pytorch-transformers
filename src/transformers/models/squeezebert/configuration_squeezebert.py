# coding=utf-8
# Copyright 2020 The SqueezeBert authors and The HuggingFace Inc. team.
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
"""SqueezeBERT model configuration"""

from collections import OrderedDict
from typing import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class SqueezeBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SqueezeBertModel`]. It is used to instantiate a
    SqueezeBERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SqueezeBERT
    [squeezebert/squeezebert-uncased](https://huggingface.co/squeezebert/squeezebert-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the SqueezeBERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`SqueezeBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):

        pad_token_id (`int`, *optional*, defaults to 0):
            The ID of the token in the word embedding to use as padding.
        embedding_size (`int`, *optional*, defaults to 768):
            The dimension of the word embedding vectors.

        q_groups (`int`, *optional*, defaults to 4):
            The number of groups in Q layer.
        k_groups (`int`, *optional*, defaults to 4):
            The number of groups in K layer.
        v_groups (`int`, *optional*, defaults to 4):
            The number of groups in V layer.
        post_attention_groups (`int`, *optional*, defaults to 1):
            The number of groups in the first feed forward network layer.
        intermediate_groups (`int`, *optional*, defaults to 4):
            The number of groups in the second feed forward network layer.
        output_groups (`int`, *optional*, defaults to 4):
            The number of groups in the third feed forward network layer.

    Examples:

    ```python
    >>> from transformers import SqueezeBertConfig, SqueezeBertModel

    >>> # Initializing a SqueezeBERT configuration
    >>> configuration = SqueezeBertConfig()

    >>> # Initializing a model (with random weights) from the configuration above
    >>> model = SqueezeBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "squeezebert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        embedding_size=768,
        q_groups=4,
        k_groups=4,
        v_groups=4,
        post_attention_groups=1,
        intermediate_groups=4,
        output_groups=4,
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
        self.q_groups = q_groups
        self.k_groups = k_groups
        self.v_groups = v_groups
        self.post_attention_groups = post_attention_groups
        self.intermediate_groups = intermediate_groups
        self.output_groups = output_groups


# # Copied from transformers.models.bert.configuration_bert.BertOnxxConfig with Bert->SqueezeBert
class SqueezeBertOnnxConfig(OnnxConfig):
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


__all__ = ["SqueezeBertConfig", "SqueezeBertOnnxConfig"]

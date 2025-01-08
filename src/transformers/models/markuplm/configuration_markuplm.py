# coding=utf-8
# Copyright 2021, The Microsoft Research Asia MarkupLM Team authors
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
"""MarkupLM model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MarkupLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MarkupLMModel`]. It is used to instantiate a
    MarkupLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MarkupLM
    [microsoft/markuplm-base](https://huggingface.co/microsoft/markuplm-base) architecture.

    Configuration objects inherit from [`BertConfig`] and can be used to control the model outputs. Read the
    documentation from [`BertConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the MarkupLM model. Defines the different tokens that can be represented by the
            *inputs_ids* passed to the forward method of [`MarkupLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
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
            The vocabulary size of the `token_type_ids` passed into [`MarkupLMModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        max_tree_id_unit_embeddings (`int`, *optional*, defaults to 1024):
            The maximum value that the tree id unit embedding might ever use. Typically set this to something large
            just in case (e.g., 1024).
        max_xpath_tag_unit_embeddings (`int`, *optional*, defaults to 256):
            The maximum value that the xpath tag unit embedding might ever use. Typically set this to something large
            just in case (e.g., 256).
        max_xpath_subs_unit_embeddings (`int`, *optional*, defaults to 1024):
            The maximum value that the xpath subscript unit embedding might ever use. Typically set this to something
            large just in case (e.g., 1024).
        tag_pad_id (`int`, *optional*, defaults to 216):
            The id of the padding token in the xpath tags.
        subs_pad_id (`int`, *optional*, defaults to 1001):
            The id of the padding token in the xpath subscripts.
        xpath_tag_unit_hidden_size (`int`, *optional*, defaults to 32):
            The hidden size of each tree id unit. One complete tree index will have
            (50*xpath_tag_unit_hidden_size)-dim.
        max_depth (`int`, *optional*, defaults to 50):
            The maximum depth in xpath.

    Examples:

    ```python
    >>> from transformers import MarkupLMModel, MarkupLMConfig

    >>> # Initializing a MarkupLM microsoft/markuplm-base style configuration
    >>> configuration = MarkupLMConfig()

    >>> # Initializing a model from the microsoft/markuplm-base style configuration
    >>> model = MarkupLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "markuplm"

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
        bos_token_id=0,
        eos_token_id=2,
        max_xpath_tag_unit_embeddings=256,
        max_xpath_subs_unit_embeddings=1024,
        tag_pad_id=216,
        subs_pad_id=1001,
        xpath_unit_hidden_size=32,
        max_depth=50,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
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
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        # additional properties
        self.max_depth = max_depth
        self.max_xpath_tag_unit_embeddings = max_xpath_tag_unit_embeddings
        self.max_xpath_subs_unit_embeddings = max_xpath_subs_unit_embeddings
        self.tag_pad_id = tag_pad_id
        self.subs_pad_id = subs_pad_id
        self.xpath_unit_hidden_size = xpath_unit_hidden_size


__all__ = ["MarkupLMConfig"]

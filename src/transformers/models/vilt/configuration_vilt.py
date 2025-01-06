# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""VilT model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class ViltConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViLTModel`]. It is used to instantiate an ViLT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViLT
    [dandelin/vilt-b32-mlm](https://huggingface.co/dandelin/vilt-b32-mlm) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the text part of the model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`ViltModel`].
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`ViltModel`]. This is used when encoding
            text.
        modality_type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the modalities passed when calling [`ViltModel`]. This is used after concatening the
            embeddings of the text and image modalities.
        max_position_embeddings (`int`, *optional*, defaults to 40):
            The maximum sequence length that this model might ever be used with.
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
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        max_image_length (`int`, *optional*, defaults to -1):
            The maximum number of patches to take as input for the Transformer encoder. If set to a positive integer,
            the encoder will sample `max_image_length` patches at maximum. If set to -1, will not be taken into
            account.
        num_images (`int`, *optional*, defaults to -1):
            The number of images to use for natural language visual reasoning. If set to a positive integer, will be
            used by [`ViltForImagesAndTextClassification`] for defining the classifier head.

    Example:

    ```python
    >>> from transformers import ViLTModel, ViLTConfig

    >>> # Initializing a ViLT dandelin/vilt-b32-mlm style configuration
    >>> configuration = ViLTConfig()

    >>> # Initializing a model from the dandelin/vilt-b32-mlm style configuration
    >>> model = ViLTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vilt"

    def __init__(
        self,
        vocab_size=30522,
        type_vocab_size=2,
        modality_type_vocab_size=2,
        max_position_embeddings=40,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=384,
        patch_size=32,
        num_channels=3,
        qkv_bias=True,
        max_image_length=-1,
        tie_word_embeddings=False,
        num_images=-1,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.modality_type_vocab_size = modality_type_vocab_size
        self.max_position_embeddings = max_position_embeddings

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.max_image_length = max_image_length
        self.num_images = num_images


__all__ = ["ViltConfig"]

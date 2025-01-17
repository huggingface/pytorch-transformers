# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Pix2Struct model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Pix2StructTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Pix2StructTextModel`]. It is used to instantiate
    a Pix2Struct text model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Pix2Struct text decoder used by
    the [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50244):
            Vocabulary size of the `Pix2Struct` text model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`Pix2StructTextModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Dimensionality of the key, query, value projections in each attention head.
        d_ff (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        dense_act_fn (`Union[Callable, str]`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string).
        decoder_start_token_id (`int`, *optional*, defaults to 0):
            The id of the `decoder_start_token_id` token.
        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the `padding` token.
        eos_token_id (`int`, *optional*, defaults to 1):
            The id of the `end-of-sequence` token.

    Example:

    ```python
    >>> from transformers import Pix2StructTextConfig, Pix2StructTextModel

    >>> # Initializing a Pix2StructTextConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructTextConfig()

    >>> # Initializing a Pix2StructTextModel (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pix2struct_text_model"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "decoder_attention_heads": "num_heads",
        "encoder_attention_heads": "num_heads",
        "encoder_layers": "num_layers",
        "decoder_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size=50244,
        hidden_size=768,
        d_kv=64,
        d_ff=2048,
        num_layers=12,
        num_heads=12,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        dense_act_fn="gelu_new",
        decoder_start_token_id=0,
        use_cache=False,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        is_decoder=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.use_cache = use_cache

        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id

        # for backwards compatibility
        self.dense_act_fn = dense_act_fn

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=is_decoder,
            **kwargs,
        )


class Pix2StructVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Pix2StructVisionModel`]. It is used to
    instantiate a Pix2Struct vision model according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of the Pix2Struct-base
    [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        patch_embed_hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the input patch_embedding layer in the Transformer encoder.
        d_ff (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        d_kv (`int`, *optional*, defaults to 64):
            Dimensionality of the key, query, value projections per attention head.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        dense_act_fn (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        seq_len (`int`, *optional*, defaults to 4096):
            Maximum sequence length (here number of patches) supported by the model.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance (in tokens) to use for each attention layer.

    Example:

    ```python
    >>> from transformers import Pix2StructVisionConfig, Pix2StructVisionModel

    >>> # Initializing a Pix2StructVisionConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructVisionConfig()

    >>> # Initializing a Pix2StructVisionModel (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pix2struct_vision_model"

    def __init__(
        self,
        hidden_size=768,
        patch_embed_hidden_size=768,
        d_ff=2048,
        d_kv=64,
        num_hidden_layers=12,
        num_attention_heads=12,
        dense_act_fn="gelu_new",
        layer_norm_eps=1e-6,
        dropout_rate=0.0,
        attention_dropout=0.0,
        initializer_range=1e-10,
        initializer_factor=1.0,
        seq_len=4096,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.patch_embed_hidden_size = patch_embed_hidden_size
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.dense_act_fn = dense_act_fn
        self.seq_len = seq_len
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.d_kv = d_kv


class Pix2StructConfig(PretrainedConfig):
    r"""
    [`Pix2StructConfig`] is the configuration class to store the configuration of a
    [`Pix2StructForConditionalGeneration`]. It is used to instantiate a Pix2Struct model according to the specified
    arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the Pix2Struct-base
    [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Pix2StructTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Pix2StructVisionConfig`].
        initializer_factor (`float`, *optional*, defaults to 1.0):
            Factor to multiply the initialization range with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        is_vqa (`bool`, *optional*, defaults to `False`):
            Whether the model has been fine-tuned for VQA or not.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration

    >>> # Initializing a Pix2StructConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructConfig()

    >>> # Initializing a Pix2StructForConditionalGeneration (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Pix2StructConfig from a Pix2StructTextConfig and a Pix2StructVisionConfig

    >>> # Initializing a Pix2Struct text and Pix2Struct vision configuration
    >>> config_text = Pix2StructTextConfig()
    >>> config_vision = Pix2StructVisionConfig()

    >>> config = Pix2StructConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "pix2struct"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        initializer_factor=1.0,
        initializer_range=0.02,
        is_vqa=False,
        tie_word_embeddings=False,
        is_encoder_decoder=True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, is_encoder_decoder=is_encoder_decoder, **kwargs)

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the Pix2StructTextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the Pix2StructVisionConfig with default values.")

        text_config["is_encoder_decoder"] = is_encoder_decoder
        text_config["tie_word_embeddings"] = tie_word_embeddings
        self.text_config = Pix2StructTextConfig(**text_config)
        self.vision_config = Pix2StructVisionConfig(**vision_config)

        self.decoder_start_token_id = self.text_config.decoder_start_token_id
        self.pad_token_id = self.text_config.pad_token_id
        self.eos_token_id = self.text_config.eos_token_id

        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range

        self.text_config.initializer_range = self.initializer_range
        self.vision_config.initializer_range = self.initializer_range

        self.is_vqa = is_vqa

    @classmethod
    def from_text_vision_configs(
        cls, text_config: Pix2StructTextConfig, vision_config: Pix2StructVisionConfig, **kwargs
    ):
        r"""
        Instantiate a [`Pix2StructConfig`] (or a derived class) from pix2struct text model configuration and pix2struct
        vision model configuration.

        Returns:
            [`Pix2StructConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


__all__ = ["Pix2StructConfig", "Pix2StructTextConfig", "Pix2StructVisionConfig"]

# coding=utf-8
# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""KOSMOS-2 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Kosmos2TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2TextModel`]. It is used to instantiate a
    KOSMOS-2 text decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text decoder of the KOSMOS-2
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 65037):
            Vocabulary size of the Kosmos2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Kosmos2Model`].
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        embed_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the layers and the pooler layer.
        layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        ffn_dim (`int`, *optional*, defaults to 8192):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Scale embeddings by diving by sqrt(embed_dim).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*, defaults to 1):
            Token id used for padding.
        bos_token_id (`int`, *optional*, defaults to 0):
            Token id used for beginning of string.
        eos_token_id (`int`, *optional*, defaults to 2):
            Token id used for end of string.
    ```"""

    model_type = "kosmos_2_text_model"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "embed_dim",
        "num_hidden_layers": "layers",
    }

    def __init__(
        self,
        vocab_size=65037,
        max_position_embeddings=2048,
        embed_dim=2048,
        layers=24,
        ffn_dim=8192,
        attention_heads=32,
        activation_function="gelu",
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        layerdrop=0.0,
        layer_norm_eps=1e-5,
        init_std=0.02,
        scale_embedding=True,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = embed_dim
        self.layers = layers
        self.ffn_dim = ffn_dim
        self.attention_heads = attention_heads
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.init_std = init_std
        self.scale_embedding = scale_embedding
        self.use_cache = use_cache


class Kosmos2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2VisionModel`]. It is used to instantiate a
    KOSMOS-2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the KOSMOS-2
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
    ```"""

    model_type = "kosmos_2_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


class Kosmos2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2Model`]. It is used to instantiate a
    KOSMOS-2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the KOSMOS-2
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2VisionConfig`].
        latent_query_num (`int`, *optional*, defaults to 64):
            The number of latent query tokens that represent the image features used in the text decoder component.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import Kosmos2Config, Kosmos2Model

    >>> # Initializing a Kosmos-2 kosmos-2-patch14-224 style configuration
    >>> configuration = Kosmos2Config()

    >>> # Initializing a model (with random weights) from the kosmos-2-patch14-224 style configuration
    >>> model = Kosmos2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "kosmos-2"
    sub_configs = {"text_config": Kosmos2TextConfig, "vision_config": Kosmos2VisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        latent_query_num=64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `Kosmos2TextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `Kosmos2VisionConfig` with default values.")

        self.text_config = Kosmos2TextConfig(**text_config)
        self.vision_config = Kosmos2VisionConfig(**vision_config)

        self.latent_query_num = latent_query_num


__all__ = ["Kosmos2Config"]

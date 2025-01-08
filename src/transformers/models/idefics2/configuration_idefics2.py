# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Idefics2 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class Idefics2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Idefics2VisionModel`]. It is used to instantiate a
    Idefics2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SigLIP checkpoint
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) used in the Idefics2 model
    [HuggingFaceM4/idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for initializing all weight matrices in the model.

    Example:

    ```python
    >>> from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionTransformer
    >>> from transformers.models.idefics2.configuration_idefics2 import Idefics2VisionConfig

    >>> # Initializing a Idefics2VisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = Idefics2VisionConfig()

    >>> # Initializing a Idefics2VisionTransformer (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = Idefics2VisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "idefics2_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=0.02,
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
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range


class Idefics2PerceiverConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the perceiver block.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        resampler_n_latents (`int`, *optional*, defaults to 64):
            Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
        resampler_depth (`int`, *optional*, defaults to 3):
            Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (<= 3).
        resampler_n_heads (`int`, *optional*, defaults to 16):
            Number of heads in each Transformer block (for multi-headed self-attention).
        resampler_head_dim (`int`, *optional*, defaults to 96):
            Dimensionality of each head projection in the Transformer block.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key-value heads in the perceiver attention block.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    """

    model_type = "idefics2_perceiver"

    def __init__(
        self,
        hidden_act="silu",
        hidden_size=4096,
        rms_norm_eps=1e-06,
        resampler_n_latents=64,
        resampler_depth=3,
        resampler_n_heads=16,
        resampler_head_dim=96,
        num_key_value_heads=4,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.resampler_n_latents = resampler_n_latents
        self.resampler_depth = resampler_depth
        self.resampler_n_heads = resampler_n_heads
        self.num_key_value_heads = num_key_value_heads
        self.resampler_head_dim = resampler_head_dim
        self.attention_dropout = attention_dropout
        if self.num_key_value_heads > self.resampler_n_heads:
            raise ValueError(
                f"num_key_value_heads={self.num_key_value_heads} must be less than or equal to"
                f" resampler_n_heads={self.resampler_n_heads}"
            )
        super().__init__(**kwargs)


class Idefics2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Idefics2Model`]. It is used to instantiate a
    Idefics2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the model of the Idefics2
    [HuggingFaceM4/idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should cache the key/value pairs of the attention mechanism.
        image_token_id (`int`, *optional*, defaults to 32001):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the token embeddings.
        vision_config (`IdeficsVisionConfig` or `dict`, *optional*):
            Custom vision config or dict
        perceiver_config (`IdeficsPerceiverConfig` or `dict`, *optional*):
            Custom perceiver config or dict
        text_config (`MistralConfig` or `dict`, *optional*):
            Custom text config or dict for the text model

    Example:
    ```python
    >>> from transformers import Idefics2Model, Idefics2Config
    >>> # Initializing configuration
    >>> configuration = Idefics2Config()
    >>> # Initializing a model from the configuration
    >>> model = Idefics2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "idefics2"
    sub_configs = {
        "text_config": AutoConfig,
        "perceiver_config": Idefics2PerceiverConfig,
        "vision_config": Idefics2VisionConfig,
    }

    def __init__(
        self,
        use_cache=True,
        image_token_id=32_001,
        tie_word_embeddings=False,
        vision_config=None,
        perceiver_config=None,
        text_config=None,
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

        if perceiver_config is None:
            self.perceiver_config = Idefics2PerceiverConfig()
            logger.info("perciver_config is None, using default perceiver config")
        elif isinstance(perceiver_config, dict):
            self.perceiver_config = Idefics2PerceiverConfig(**perceiver_config)
        elif isinstance(perceiver_config, Idefics2PerceiverConfig):
            self.perceiver_config = perceiver_config

        if vision_config is None:
            self.vision_config = Idefics2VisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = Idefics2VisionConfig(**vision_config)
        elif isinstance(vision_config, Idefics2VisionConfig):
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "mistral"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            logger.info("text_config is None, using default text config")
            text_config = CONFIG_MAPPING["mistral"](
                max_position_embeddings=4096 * 8,
                rms_norm_eps=1e-5,
                # None in the original configuration_mistral, we set it to the unk_token_id
                pad_token_id=0,
                tie_word_embeddings=False,
            )

        self.text_config = text_config
        if self.text_config.hidden_size != self.perceiver_config.hidden_size:
            self.perceiver_config.hidden_size = self.text_config.hidden_size
            self.perceiver_config.rms_norm_eps = self.text_config.rms_norm_eps
            logger.warning_once(
                "Perceiver config has a different `hidden_size` than text config, which means default values were used. "
                "In your model's config on the hub, add `hidden_size` and `rms_norm_eps` keys under the `perceiver_config` dict. "
            )

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


__all__ = ["Idefics2Config"]

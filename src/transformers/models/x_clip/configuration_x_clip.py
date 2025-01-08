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
"""X-CLIP model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class XCLIPTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XCLIPModel`]. It is used to instantiate an X-CLIP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the X-CLIP text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`XCLIPModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import XCLIPTextModel, XCLIPTextConfig

    >>> # Initializing a XCLIPTextModel with microsoft/xclip-base-patch32 style configuration
    >>> configuration = XCLIPTextConfig()

    >>> # Initializing a XCLIPTextConfig from the microsoft/xclip-base-patch32 style configuration
    >>> model = XCLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xclip_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout


class XCLIPVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XCLIPModel`]. It is used to instantiate an X-CLIP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

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
        mit_hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers of the Multiframe Integration Transformer (MIT).
        mit_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Multiframe Integration Transformer
            (MIT).
        mit_num_hidden_layers (`int`, *optional*, defaults to 1):
            Number of hidden layers in the Multiframe Integration Transformer (MIT).
        mit_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Multiframe Integration Transformer (MIT).
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"`, `"gelu_new"` and `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate.

    Example:

    ```python
    >>> from transformers import XCLIPVisionModel, XCLIPVisionConfig

    >>> # Initializing a XCLIPVisionModel with microsoft/xclip-base-patch32 style configuration
    >>> configuration = XCLIPVisionConfig()

    >>> # Initializing a XCLIPVisionModel model from the microsoft/xclip-base-patch32 style configuration
    >>> model = XCLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xclip_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        mit_hidden_size=512,
        mit_intermediate_size=2048,
        mit_num_hidden_layers=1,
        mit_num_attention_heads=8,
        num_channels=3,
        image_size=224,
        patch_size=32,
        num_frames=8,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mit_hidden_size = mit_hidden_size
        self.mit_intermediate_size = mit_intermediate_size
        self.mit_num_hidden_layers = mit_num_hidden_layers
        self.mit_num_attention_heads = mit_num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.drop_path_rate = drop_path_rate


class XCLIPConfig(PretrainedConfig):
    r"""
    [`XCLIPConfig`] is the configuration class to store the configuration of a [`XCLIPModel`]. It is used to
    instantiate X-CLIP model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the X-CLIP
    [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`XCLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`XCLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        prompt_layers (`int`, *optional*, defaults to 2):
            Number of layers in the video specific prompt generator.
        prompt_alpha (`float`, *optional*, defaults to 0.1):
            Alpha value to use in the video specific prompt generator.
        prompt_hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the video specific prompt generator. If string,
            `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        prompt_num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the cross-attention of the video specific prompt generator.
        prompt_attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers in the video specific prompt generator.
        prompt_projection_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the projection layers in the video specific prompt generator.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* parameter. Default is used as per the original XCLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

    model_type = "xclip"
    sub_configs = {"text_config": XCLIPTextConfig, "vision_config": XCLIPVisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        prompt_layers=2,
        prompt_alpha=0.1,
        prompt_hidden_act="quick_gelu",
        prompt_num_attention_heads=8,
        prompt_attention_dropout=0.0,
        prompt_projection_dropout=0.0,
        logit_scale_init_value=2.6592,
        **kwargs,
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = XCLIPTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key not in ["transformers_version"]:
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `XCLIPTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = XCLIPVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key not in ["transformers_version"]:
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `XCLIPVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `XCLIPTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `XCLIPVisionConfig` with default values.")

        self.text_config = XCLIPTextConfig(**text_config)
        self.vision_config = XCLIPVisionConfig(**vision_config)

        self.projection_dim = projection_dim
        self.prompt_layers = prompt_layers
        self.prompt_alpha = prompt_alpha
        self.prompt_hidden_act = prompt_hidden_act
        self.prompt_num_attention_heads = prompt_num_attention_heads
        self.prompt_attention_dropout = prompt_attention_dropout
        self.prompt_projection_dropout = prompt_projection_dropout
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: XCLIPTextConfig, vision_config: XCLIPVisionConfig, **kwargs):
        r"""
        Instantiate a [`XCLIPConfig`] (or a derived class) from xclip text model configuration and xclip vision model
        configuration.

        Returns:
            [`XCLIPConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


__all__ = ["XCLIPConfig", "XCLIPTextConfig", "XCLIPVisionConfig"]

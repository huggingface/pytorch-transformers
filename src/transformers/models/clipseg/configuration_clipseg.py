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
"""CLIPSeg model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class CLIPSegTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPSegModel`]. It is used to instantiate an
    CLIPSeg model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CLIPSeg
    [CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the CLIPSeg text model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`CLIPSegModel`].
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
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 49406):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 49407):
            End of stream token id.

    Example:

    ```python
    >>> from transformers import CLIPSegTextConfig, CLIPSegTextModel

    >>> # Initializing a CLIPSegTextConfig with CIDAS/clipseg-rd64 style configuration
    >>> configuration = CLIPSegTextConfig()

    >>> # Initializing a CLIPSegTextModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
    >>> model = CLIPSegTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clipseg_text_model"
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
        bos_token_id=49406,
        eos_token_id=49407,
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


class CLIPSegVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPSegModel`]. It is used to instantiate an
    CLIPSeg model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CLIPSeg
    [CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

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
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
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

    Example:

    ```python
    >>> from transformers import CLIPSegVisionConfig, CLIPSegVisionModel

    >>> # Initializing a CLIPSegVisionConfig with CIDAS/clipseg-rd64 style configuration
    >>> configuration = CLIPSegVisionConfig()

    >>> # Initializing a CLIPSegVisionModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
    >>> model = CLIPSegVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clipseg_vision_model"
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


class CLIPSegConfig(PretrainedConfig):
    r"""
    [`CLIPSegConfig`] is the configuration class to store the configuration of a [`CLIPSegModel`]. It is used to
    instantiate a CLIPSeg model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the CLIPSeg
    [CIDAS/clipseg-rd64](https://huggingface.co/CIDAS/clipseg-rd64) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPSegTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPSegVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original CLIPSeg implementation.
        extract_layers (`List[int]`, *optional*, defaults to `[3, 6, 9]`):
            Layers to extract when forwarding the query image through the frozen visual backbone of CLIP.
        reduce_dim (`int`, *optional*, defaults to 64):
            Dimensionality to reduce the CLIP vision embedding.
        decoder_num_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads in the decoder of CLIPSeg.
        decoder_attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        decoder_hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layers in the Transformer decoder.
        conditional_layer (`int`, *optional*, defaults to 0):
            The layer to use of the Transformer encoder whose activations will be combined with the condition
            embeddings using FiLM (Feature-wise Linear Modulation). If 0, the last layer is used.
        use_complex_transposed_convolution (`bool`, *optional*, defaults to `False`):
            Whether to use a more complex transposed convolution in the decoder, enabling more fine-grained
            segmentation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import CLIPSegConfig, CLIPSegModel

    >>> # Initializing a CLIPSegConfig with CIDAS/clipseg-rd64 style configuration
    >>> configuration = CLIPSegConfig()

    >>> # Initializing a CLIPSegModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
    >>> model = CLIPSegModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLIPSegConfig from a CLIPSegTextConfig and a CLIPSegVisionConfig

    >>> # Initializing a CLIPSegText and CLIPSegVision configuration
    >>> config_text = CLIPSegTextConfig()
    >>> config_vision = CLIPSegVisionConfig()

    >>> config = CLIPSegConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "clipseg"
    sub_configs = {"text_config": CLIPSegTextConfig, "vision_config": CLIPSegVisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        extract_layers=[3, 6, 9],
        reduce_dim=64,
        decoder_num_attention_heads=4,
        decoder_attention_dropout=0.0,
        decoder_hidden_act="quick_gelu",
        decoder_intermediate_size=2048,
        conditional_layer=0,
        use_complex_transposed_convolution=False,
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
            _text_config_dict = CLIPSegTextConfig(**text_config_dict).to_dict()

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
                            f"`text_config_dict` is provided which will be used to initialize `CLIPSegTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = CLIPSegVisionConfig(**vision_config_dict).to_dict()
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
                            f"`vision_config_dict` is provided which will be used to initialize `CLIPSegVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `CLIPSegTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `CLIPSegVisionConfig` with default values.")

        self.text_config = CLIPSegTextConfig(**text_config)
        self.vision_config = CLIPSegVisionConfig(**vision_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.extract_layers = extract_layers
        self.reduce_dim = reduce_dim
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_attention_dropout = decoder_attention_dropout
        self.decoder_hidden_act = decoder_hidden_act
        self.decoder_intermediate_size = decoder_intermediate_size
        self.conditional_layer = conditional_layer
        self.initializer_factor = 1.0
        self.use_complex_transposed_convolution = use_complex_transposed_convolution

    @classmethod
    def from_text_vision_configs(cls, text_config: CLIPSegTextConfig, vision_config: CLIPSegVisionConfig, **kwargs):
        r"""
        Instantiate a [`CLIPSegConfig`] (or a derived class) from clipseg text model configuration and clipseg vision
        model configuration.

        Returns:
            [`CLIPSegConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


__all__ = ["CLIPSegConfig", "CLIPSegTextConfig", "CLIPSegVisionConfig"]

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
"""CLVP model configuration"""

import os
from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    pass

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class ClvpEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClvpEncoder`]. It is used to instantiate a CLVP
    text or CLVP speech encoder according to the specified arguments. Instantiating a configuration with the defaults
    will yield a similar configuration to that of the encoder of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256):
            Vocabulary size of the CLVP Encoder model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the projection vector.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the feed-forward layers in [`ClvpEncoderMLP`].
        use_rotary_embedding (`bool`, *optional*, defaults to `True`):
            Whether to use rotary_embedding or not.
        use_attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in Query, Key and Value layers during self attention.
        summary_type (`str`, *optional*, defaults to `"mean"`):
            What strategy to use to get pooler_output from the last_hidden_state. `"last"`, `"first"`, `"mean"` and
            `"cls_index"` are supported.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        bos_token_id (`int`, *optional*, defaults to 255):
            Beginning of sequence token id.
        eos_token_id (`int`, *optional*, defaults to 0):
            End of sequence token id.

    Example:

    ```python
    >>> from transformers import ClvpEncoderConfig, ClvpEncoder

    >>> # Initializing a ClvpEncoderConfig with susnato/clvp_dev style configuration
    >>> encoder_configuration = ClvpEncoderConfig()

    >>> # Initializing a ClvpEncoder (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpEncoder(encoder_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clvp_encoder"
    base_config_key = ["text_config", "speech_config"]

    def __init__(
        self,
        vocab_size=256,
        hidden_size=768,
        intermediate_size=1536,
        projection_dim=768,
        num_hidden_layers=20,
        num_attention_heads=12,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.1,
        dropout=0.1,
        use_rotary_embedding=True,
        use_attention_bias=False,
        summary_type="mean",
        initializer_factor=1.0,
        bos_token_id=255,
        eos_token_id=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.use_rotary_embedding = use_rotary_embedding
        self.use_attention_bias = use_attention_bias
        self.summary_type = summary_type
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], config_type: str = "text_config", **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # make sure to have the config_type be either "text_config" or "speech_config"
        # this is to make sure that we can load only text or speech configs from the nested ClvpConfig.
        if config_type not in cls.base_config_key:
            raise ValueError(
                f"We can only load either 'text_config' or 'speech_config' but you are trying to load" f"{config_type}"
            )

        # get the text config dict if we are loading from ClvpConfig
        if config_dict.get("model_type") == "clvp":
            config_dict = config_dict[config_type]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ClvpDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClvpDecoder`]. It is used to instantiate a CLVP
    Decoder Model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Decoder part of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    The architecture is similar to GPT2.

    Args:
        vocab_size (`int`, *optional*, defaults to 8194):
            Vocabulary size of the model.
        max_position_embeddings (`int`, *optional*, defaults to 608):
            The maximum sequence length of mel tokens that this model might ever be used with. Similar to `n_positions`
            in `GPT2Config`.
        max_text_tokens (`int`, *optional*, defaults to 404):
            The maximum sequence length of text tokens that this model might ever be used with. Similar to
            `n_positions` in `GPT2Config`.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 30):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times `hidden_size`.
        num_mel_attn_blocks (`int`, *optional*, defaults to 6):
            Denotes the number of self attention layers in [`ClvpConditioningEncoder`].
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary.

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio to be used after the projection and activation.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        bos_token_id (`int`, *optional*, defaults to 8192):
            Beginning of sequence token id, used at the start of the generation.
        eos_token_id (`int`, *optional*, defaults to 8193):
            End of sequence token id, used in the method
            [`ClvpModelForConditionalGeneration.fix_speech_decoder_output()`] to correct decoder outputs.
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted mel features. This value is used in [`ClvpConditioningEncoder`].
        use_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in Query, Key and Value layers during self attention.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        decoder_fixing_codes (`list`, *optional*, defaults to `[83, 45, 45, 248]`):
            These values are used in the method `fix_speech_decoder_output` to fix decoder generated outputs.

    Example:

    ```python
    >>> from transformers import ClvpDecoderConfig, ClvpDecoder

    >>> # Initializing a ClvpDecoderConfig with susnato/clvp_dev style configuration
    >>> decoder_configuration = ClvpDecoderConfig()

    >>> # Initializing a ClvpDecoder (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpDecoder(decoder_configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clvp_decoder"
    base_config_key = "decoder_config"

    def __init__(
        self,
        vocab_size=8194,
        max_position_embeddings=608,
        max_text_tokens=404,
        hidden_size=1024,
        num_hidden_layers=30,
        num_attention_heads=16,
        n_inner=None,
        num_mel_attn_blocks=6,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attention_dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        use_cache=True,
        bos_token_id=8192,
        eos_token_id=8193,
        feature_size=80,
        use_attention_bias=True,
        initializer_factor=1.0,
        decoder_fixing_codes=[83, 45, 45, 248],
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.max_text_tokens = max_text_tokens
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_inner = n_inner
        self.num_mel_attn_blocks = num_mel_attn_blocks
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.use_cache = use_cache
        self.feature_size = feature_size
        self.use_attention_bias = use_attention_bias
        self.initializer_factor = initializer_factor
        self.decoder_fixing_codes = decoder_fixing_codes

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class ClvpConfig(PretrainedConfig):
    r"""
    [`ClvpConfig`] is the configuration class to store the configuration of a [`ClvpModelForConditionalGeneration`]. It
    is used to instantiate a CLVP model according to the specified arguments, defining the text model, speech model and
    decoder model configs. Instantiating a configuration with the defaults will yield a similar configuration to that
    of the CLVP [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize the CLVP text encoder.
        speech_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize CLVP speech encoder.
        decoder_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClvpDecoderConfig`].
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of text and speech projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original CLVP implementation.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
            testing).
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ClvpConfig, ClvpModelForConditionalGeneration

    >>> # Initializing a ClvpConfig with susnato/clvp_dev style configuration
    >>> configuration = ClvpConfig()

    >>> # Initializing a ClvpModelForConditionalGeneration (with random weights) from the susnato/clvp_dev style configuration
    >>> model = ClvpModelForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLVPConfig from a CLVPTextConfig, CLVPSpeechConfig and a CLVPAutoRegressiveConfig
    >>> from transformers import ClvpEncoderConfig, ClvpDecoderConfig

    >>> # Initializing a CLVP text, CLVP speech and CLVP decoder configuration
    >>> config_text = ClvpEncoderConfig()
    >>> config_speech = ClvpEncoderConfig()
    >>> decoder_config = ClvpDecoderConfig()

    >>> config = ClvpConfig.from_sub_model_configs(config_text, config_speech, decoder_config)
    ```"""

    model_type = "clvp"
    sub_configs = {
        "text_config": ClvpEncoderConfig,
        "speech_config": ClvpEncoderConfig,
        "decoder_config": ClvpDecoderConfig,
    }

    def __init__(
        self,
        text_config=None,
        speech_config=None,
        decoder_config=None,
        projection_dim=768,
        logit_scale_init_value=2.6592,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `ClvpEncoderConfig` with default values.")

        if speech_config is None:
            speech_config = {}
            logger.info("`speech_config` is `None`. initializing the `ClvpEncoderConfig` with default values.")

        if decoder_config is None:
            decoder_config = {}
            logger.info("`decoder_config` is `None`. initializing the `ClvpDecoderConfig` with default values.")

        self.text_config = ClvpEncoderConfig(**text_config)
        self.speech_config = ClvpEncoderConfig(**speech_config)
        self.decoder_config = ClvpDecoderConfig(**decoder_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = initializer_factor

    @classmethod
    def from_sub_model_configs(
        cls,
        text_config: ClvpEncoderConfig,
        speech_config: ClvpEncoderConfig,
        decoder_config: ClvpDecoderConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`ClvpConfig`] (or a derived class) from CLVP text model configuration, CLVP speech model
        configuration and CLVP decoder model configuration.

        Args:
            text_config (`ClvpEncoderConfig`):
                Text model configuration of type [`ClvpEncoderConfig`].
            speech_config (`ClvpEncoderConfig`):
                Speech model configuration of type [`ClvpEncoderConfig`].
            decoder_config (`ClvpDecoderConfig`):
                Decoder model configuration of type [`ClvpDecoderConfig`].

        Returns:
            [`ClvpConfig`]: An instance of a configuration object
        """

        return cls(
            text_config=text_config.to_dict(),
            speech_config=speech_config.to_dict(),
            decoder_config=decoder_config.to_dict(),
            **kwargs,
        )


__all__ = ["ClvpConfig", "ClvpDecoderConfig", "ClvpEncoderConfig"]

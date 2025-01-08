# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""MusicGen model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class MusicgenDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`MusicgenDecoder`]. It is used to instantiate a
    MusicGen decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MusicGen
    [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 2048):
            Vocabulary size of the MusicgenDecoder model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`MusicgenDecoder`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer block.
        ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer block.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, text_encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_factor (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(hidden_size).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models)
        num_codebooks (`int`, *optional*, defaults to 4):
            The number of parallel codebooks forwarded to the model.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether input and output word embeddings should be tied.
        audio_channels (`int`, *optional*, defaults to 1
            Number of channels in the audio data. Either 1 for mono or 2 for stereo. Stereo models generate a separate
            audio stream for the left/right output channels. Mono models generate a single audio stream output.
    """

    model_type = "musicgen_decoder"
    base_config_key = "decoder_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=2048,
        max_position_embeddings=2048,
        num_hidden_layers=24,
        ffn_dim=4096,
        num_attention_heads=16,
        layerdrop=0.0,
        use_cache=True,
        activation_function="gelu",
        hidden_size=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        initializer_factor=0.02,
        scale_embedding=False,
        num_codebooks=4,
        audio_channels=1,
        pad_token_id=2048,
        bos_token_id=2048,
        eos_token_id=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.initializer_factor = initializer_factor
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.num_codebooks = num_codebooks

        if audio_channels not in [1, 2]:
            raise ValueError(f"Expected 1 (mono) or 2 (stereo) audio channels, got {audio_channels} channels.")
        self.audio_channels = audio_channels

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MusicgenConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MusicgenModel`]. It is used to instantiate a
    MusicGen model according to the specified arguments, defining the text encoder, audio encoder and MusicGen decoder
    configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **text_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the text encoder config.
                - **audio_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the audio encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Example:

    ```python
    >>> from transformers import (
    ...     MusicgenConfig,
    ...     MusicgenDecoderConfig,
    ...     T5Config,
    ...     EncodecConfig,
    ...     MusicgenForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> text_encoder_config = T5Config()
    >>> audio_encoder_config = EncodecConfig()
    >>> decoder_config = MusicgenDecoderConfig()

    >>> configuration = MusicgenConfig.from_sub_models_config(
    ...     text_encoder_config, audio_encoder_config, decoder_config
    ... )

    >>> # Initializing a MusicgenForConditionalGeneration (with random weights) from the facebook/musicgen-small style configuration
    >>> model = MusicgenForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("musicgen-model")

    >>> # loading model and config from pretrained folder
    >>> musicgen_config = MusicgenConfig.from_pretrained("musicgen-model")
    >>> model = MusicgenForConditionalGeneration.from_pretrained("musicgen-model", config=musicgen_config)
    ```"""

    model_type = "musicgen"
    sub_configs = {
        "text_encoder": AutoConfig,
        "audio_encoder": AutoConfig,
        "decoder": MusicgenDecoderConfig,
    }
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError("Config has to be initialized with text_encoder, audio_encoder and decoder config")

        text_encoder_config = kwargs.pop("text_encoder")
        text_encoder_model_type = text_encoder_config.pop("model_type")

        audio_encoder_config = kwargs.pop("audio_encoder")
        audio_encoder_model_type = audio_encoder_config.pop("model_type")

        decoder_config = kwargs.pop("decoder")

        self.text_encoder = AutoConfig.for_model(text_encoder_model_type, **text_encoder_config)
        self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **audio_encoder_config)
        self.decoder = MusicgenDecoderConfig(**decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_sub_models_config(
        cls,
        text_encoder_config: PretrainedConfig,
        audio_encoder_config: PretrainedConfig,
        decoder_config: MusicgenDecoderConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MusicgenConfig`] (or a derived class) from text encoder, audio encoder and decoder
        configurations.

        Returns:
            [`MusicgenConfig`]: An instance of a configuration object
        """

        return cls(
            text_encoder=text_encoder_config.to_dict(),
            audio_encoder=audio_encoder_config.to_dict(),
            decoder=decoder_config.to_dict(),
            **kwargs,
        )

    @property
    # This is a property because you might want to change the codec model on the fly
    def sampling_rate(self):
        return self.audio_encoder.sampling_rate


__all__ = ["MusicgenConfig", "MusicgenDecoderConfig"]

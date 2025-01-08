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
"""FastSpeech2Conformer model configuration"""

from typing import Dict

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class FastSpeech2ConformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerModel`]. It is used to
    instantiate a FastSpeech2Conformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2Conformer [espnet/fastspeech2_conformer](https://huggingface.co/espnet/fastspeech2_conformer)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 384):
            The dimensionality of the hidden layers.
        vocab_size (`int`, *optional*, defaults to 78):
            The size of the vocabulary.
        num_mel_bins (`int`, *optional*, defaults to 80):
            The number of mel filters used in the filter bank.
        encoder_num_attention_heads (`int`, *optional*, defaults to 2):
            The number of attention heads in the encoder.
        encoder_layers (`int`, *optional*, defaults to 4):
            The number of layers in the encoder.
        encoder_linear_units (`int`, *optional*, defaults to 1536):
            The number of units in the linear layer of the encoder.
        decoder_layers (`int`, *optional*, defaults to 4):
            The number of layers in the decoder.
        decoder_num_attention_heads (`int`, *optional*, defaults to 2):
            The number of attention heads in the decoder.
        decoder_linear_units (`int`, *optional*, defaults to 1536):
            The number of units in the linear layer of the decoder.
        speech_decoder_postnet_layers (`int`, *optional*, defaults to 5):
            The number of layers in the post-net of the speech decoder.
        speech_decoder_postnet_units (`int`, *optional*, defaults to 256):
            The number of units in the post-net layers of the speech decoder.
        speech_decoder_postnet_kernel (`int`, *optional*, defaults to 5):
            The kernel size in the post-net of the speech decoder.
        positionwise_conv_kernel_size (`int`, *optional*, defaults to 3):
            The size of the convolution kernel used in the position-wise layer.
        encoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Specifies whether to normalize before encoder layers.
        decoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Specifies whether to normalize before decoder layers.
        encoder_concat_after (`bool`, *optional*, defaults to `False`):
            Specifies whether to concatenate after encoder layers.
        decoder_concat_after (`bool`, *optional*, defaults to `False`):
            Specifies whether to concatenate after decoder layers.
        reduction_factor (`int`, *optional*, defaults to 1):
            The factor by which the speech frame rate is reduced.
        speaking_speed (`float`, *optional*, defaults to 1.0):
            The speed of the speech produced.
        use_macaron_style_in_conformer (`bool`, *optional*, defaults to `True`):
            Specifies whether to use macaron style in the conformer.
        use_cnn_in_conformer (`bool`, *optional*, defaults to `True`):
            Specifies whether to use convolutional neural networks in the conformer.
        encoder_kernel_size (`int`, *optional*, defaults to 7):
            The kernel size used in the encoder.
        decoder_kernel_size (`int`, *optional*, defaults to 31):
            The kernel size used in the decoder.
        duration_predictor_layers (`int`, *optional*, defaults to 2):
            The number of layers in the duration predictor.
        duration_predictor_channels (`int`, *optional*, defaults to 256):
            The number of channels in the duration predictor.
        duration_predictor_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size used in the duration predictor.
        energy_predictor_layers (`int`, *optional*, defaults to 2):
            The number of layers in the energy predictor.
        energy_predictor_channels (`int`, *optional*, defaults to 256):
            The number of channels in the energy predictor.
        energy_predictor_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size used in the energy predictor.
        energy_predictor_dropout (`float`, *optional*, defaults to 0.5):
            The dropout rate in the energy predictor.
        energy_embed_kernel_size (`int`, *optional*, defaults to 1):
            The kernel size used in the energy embed layer.
        energy_embed_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate in the energy embed layer.
        stop_gradient_from_energy_predictor (`bool`, *optional*, defaults to `False`):
            Specifies whether to stop gradients from the energy predictor.
        pitch_predictor_layers (`int`, *optional*, defaults to 5):
            The number of layers in the pitch predictor.
        pitch_predictor_channels (`int`, *optional*, defaults to 256):
            The number of channels in the pitch predictor.
        pitch_predictor_kernel_size (`int`, *optional*, defaults to 5):
            The kernel size used in the pitch predictor.
        pitch_predictor_dropout (`float`, *optional*, defaults to 0.5):
            The dropout rate in the pitch predictor.
        pitch_embed_kernel_size (`int`, *optional*, defaults to 1):
            The kernel size used in the pitch embed layer.
        pitch_embed_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate in the pitch embed layer.
        stop_gradient_from_pitch_predictor (`bool`, *optional*, defaults to `True`):
            Specifies whether to stop gradients from the pitch predictor.
        encoder_dropout_rate (`float`, *optional*, defaults to 0.2):
            The dropout rate in the encoder.
        encoder_positional_dropout_rate (`float`, *optional*, defaults to 0.2):
            The positional dropout rate in the encoder.
        encoder_attention_dropout_rate (`float`, *optional*, defaults to 0.2):
            The attention dropout rate in the encoder.
        decoder_dropout_rate (`float`, *optional*, defaults to 0.2):
            The dropout rate in the decoder.
        decoder_positional_dropout_rate (`float`, *optional*, defaults to 0.2):
            The positional dropout rate in the decoder.
        decoder_attention_dropout_rate (`float`, *optional*, defaults to 0.2):
            The attention dropout rate in the decoder.
        duration_predictor_dropout_rate (`float`, *optional*, defaults to 0.2):
            The dropout rate in the duration predictor.
        speech_decoder_postnet_dropout (`float`, *optional*, defaults to 0.5):
            The dropout rate in the speech decoder postnet.
        max_source_positions (`int`, *optional*, defaults to 5000):
            if `"relative"` position embeddings are used, defines the maximum source input positions.
        use_masking (`bool`, *optional*, defaults to `True`):
            Specifies whether to use masking in the model.
        use_weighted_masking (`bool`, *optional*, defaults to `False`):
            Specifies whether to use weighted masking in the model.
        num_speakers (`int`, *optional*):
            Number of speakers. If set to > 1, assume that the speaker ids will be provided as the input and use
            speaker id embedding layer.
        num_languages (`int`, *optional*):
            Number of languages. If set to > 1, assume that the language ids will be provided as the input and use the
            languge id embedding layer.
        speaker_embed_dim (`int`, *optional*):
            Speaker embedding dimension. If set to > 0, assume that speaker_embedding will be provided as the input.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Specifies whether the model is an encoder-decoder.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerModel, FastSpeech2ConformerConfig

    >>> # Initializing a FastSpeech2Conformer style configuration
    >>> configuration = FastSpeech2ConformerConfig()

    >>> # Initializing a model from the FastSpeech2Conformer style configuration
    >>> model = FastSpeech2ConformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fastspeech2_conformer"
    base_config_key = "model_config"
    attribute_map = {"num_hidden_layers": "encoder_layers", "num_attention_heads": "encoder_num_attention_heads"}

    def __init__(
        self,
        hidden_size=384,
        vocab_size=78,
        num_mel_bins=80,
        encoder_num_attention_heads=2,
        encoder_layers=4,
        encoder_linear_units=1536,
        decoder_layers=4,
        decoder_num_attention_heads=2,
        decoder_linear_units=1536,
        speech_decoder_postnet_layers=5,
        speech_decoder_postnet_units=256,
        speech_decoder_postnet_kernel=5,
        positionwise_conv_kernel_size=3,
        encoder_normalize_before=False,
        decoder_normalize_before=False,
        encoder_concat_after=False,
        decoder_concat_after=False,
        reduction_factor=1,
        speaking_speed=1.0,
        use_macaron_style_in_conformer=True,
        use_cnn_in_conformer=True,
        encoder_kernel_size=7,
        decoder_kernel_size=31,
        duration_predictor_layers=2,
        duration_predictor_channels=256,
        duration_predictor_kernel_size=3,
        energy_predictor_layers=2,
        energy_predictor_channels=256,
        energy_predictor_kernel_size=3,
        energy_predictor_dropout=0.5,
        energy_embed_kernel_size=1,
        energy_embed_dropout=0.0,
        stop_gradient_from_energy_predictor=False,
        pitch_predictor_layers=5,
        pitch_predictor_channels=256,
        pitch_predictor_kernel_size=5,
        pitch_predictor_dropout=0.5,
        pitch_embed_kernel_size=1,
        pitch_embed_dropout=0.0,
        stop_gradient_from_pitch_predictor=True,
        encoder_dropout_rate=0.2,
        encoder_positional_dropout_rate=0.2,
        encoder_attention_dropout_rate=0.2,
        decoder_dropout_rate=0.2,
        decoder_positional_dropout_rate=0.2,
        decoder_attention_dropout_rate=0.2,
        duration_predictor_dropout_rate=0.2,
        speech_decoder_postnet_dropout=0.5,
        max_source_positions=5000,
        use_masking=True,
        use_weighted_masking=False,
        num_speakers=None,
        num_languages=None,
        speaker_embed_dim=None,
        is_encoder_decoder=True,
        **kwargs,
    ):
        if positionwise_conv_kernel_size % 2 == 0:
            raise ValueError(
                f"positionwise_conv_kernel_size must be odd, but got {positionwise_conv_kernel_size} instead."
            )
        if encoder_kernel_size % 2 == 0:
            raise ValueError(f"encoder_kernel_size must be odd, but got {encoder_kernel_size} instead.")
        if decoder_kernel_size % 2 == 0:
            raise ValueError(f"decoder_kernel_size must be odd, but got {decoder_kernel_size} instead.")
        if duration_predictor_kernel_size % 2 == 0:
            raise ValueError(
                f"duration_predictor_kernel_size must be odd, but got {duration_predictor_kernel_size} instead."
            )
        if energy_predictor_kernel_size % 2 == 0:
            raise ValueError(
                f"energy_predictor_kernel_size must be odd, but got {energy_predictor_kernel_size} instead."
            )
        if energy_embed_kernel_size % 2 == 0:
            raise ValueError(f"energy_embed_kernel_size must be odd, but got {energy_embed_kernel_size} instead.")
        if pitch_predictor_kernel_size % 2 == 0:
            raise ValueError(
                f"pitch_predictor_kernel_size must be odd, but got {pitch_predictor_kernel_size} instead."
            )
        if pitch_embed_kernel_size % 2 == 0:
            raise ValueError(f"pitch_embed_kernel_size must be odd, but got {pitch_embed_kernel_size} instead.")
        if hidden_size % encoder_num_attention_heads != 0:
            raise ValueError("The hidden_size must be evenly divisible by encoder_num_attention_heads.")
        if hidden_size % decoder_num_attention_heads != 0:
            raise ValueError("The hidden_size must be evenly divisible by decoder_num_attention_heads.")
        if use_masking and use_weighted_masking:
            raise ValueError("Either use_masking or use_weighted_masking can be True, but not both.")

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.encoder_config = {
            "num_attention_heads": encoder_num_attention_heads,
            "layers": encoder_layers,
            "kernel_size": encoder_kernel_size,
            "attention_dropout_rate": encoder_attention_dropout_rate,
            "dropout_rate": encoder_dropout_rate,
            "positional_dropout_rate": encoder_positional_dropout_rate,
            "linear_units": encoder_linear_units,
            "normalize_before": encoder_normalize_before,
            "concat_after": encoder_concat_after,
        }
        self.decoder_config = {
            "num_attention_heads": decoder_num_attention_heads,
            "layers": decoder_layers,
            "kernel_size": decoder_kernel_size,
            "attention_dropout_rate": decoder_attention_dropout_rate,
            "dropout_rate": decoder_dropout_rate,
            "positional_dropout_rate": decoder_positional_dropout_rate,
            "linear_units": decoder_linear_units,
            "normalize_before": decoder_normalize_before,
            "concat_after": decoder_concat_after,
        }
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_layers = encoder_layers
        self.duration_predictor_channels = duration_predictor_channels
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.duration_predictor_layers = duration_predictor_layers
        self.energy_embed_dropout = energy_embed_dropout
        self.energy_embed_kernel_size = energy_embed_kernel_size
        self.energy_predictor_channels = energy_predictor_channels
        self.energy_predictor_dropout = energy_predictor_dropout
        self.energy_predictor_kernel_size = energy_predictor_kernel_size
        self.energy_predictor_layers = energy_predictor_layers
        self.pitch_embed_dropout = pitch_embed_dropout
        self.pitch_embed_kernel_size = pitch_embed_kernel_size
        self.pitch_predictor_channels = pitch_predictor_channels
        self.pitch_predictor_dropout = pitch_predictor_dropout
        self.pitch_predictor_kernel_size = pitch_predictor_kernel_size
        self.pitch_predictor_layers = pitch_predictor_layers
        self.positionwise_conv_kernel_size = positionwise_conv_kernel_size
        self.speech_decoder_postnet_units = speech_decoder_postnet_units
        self.speech_decoder_postnet_dropout = speech_decoder_postnet_dropout
        self.speech_decoder_postnet_kernel = speech_decoder_postnet_kernel
        self.speech_decoder_postnet_layers = speech_decoder_postnet_layers
        self.reduction_factor = reduction_factor
        self.speaking_speed = speaking_speed
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.max_source_positions = max_source_positions
        self.use_cnn_in_conformer = use_cnn_in_conformer
        self.use_macaron_style_in_conformer = use_macaron_style_in_conformer
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.num_speakers = num_speakers
        self.num_languages = num_languages
        self.speaker_embed_dim = speaker_embed_dim
        self.duration_predictor_dropout_rate = duration_predictor_dropout_rate
        self.is_encoder_decoder = is_encoder_decoder

        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


class FastSpeech2ConformerHifiGanConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerHifiGanModel`]. It is used to
    instantiate a FastSpeech2Conformer HiFi-GAN vocoder model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2Conformer
    [espnet/fastspeech2_conformer_hifigan](https://huggingface.co/espnet/fastspeech2_conformer_hifigan) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_dim (`int`, *optional*, defaults to 80):
            The number of frequency bins in the input log-mel spectrogram.
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the upsampling network.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 2, 2]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
            length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[16, 16, 4, 4]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
            length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match the length of
            *upsample_rates*.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
            fusion (MRF) module.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            multi-receptive field fusion (MRF) module.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the spectrogram before vocoding using the vocoder's learned mean and variance.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig

    >>> # Initializing a FastSpeech2ConformerHifiGan configuration
    >>> configuration = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FastSpeech2ConformerHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "hifigan"
    base_config_key = "vocoder_config"

    def __init__(
        self,
        model_in_dim=80,
        upsample_initial_channel=512,
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        initializer_range=0.01,
        leaky_relu_slope=0.1,
        normalize_before=True,
        **kwargs,
    ):
        self.model_in_dim = model_in_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.initializer_range = initializer_range
        self.leaky_relu_slope = leaky_relu_slope
        self.normalize_before = normalize_before
        super().__init__(**kwargs)


class FastSpeech2ConformerWithHifiGanConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerWithHifiGan`]. It is used to
    instantiate a `FastSpeech2ConformerWithHifiGanModel` model according to the specified sub-models configurations,
    defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2ConformerModel [espnet/fastspeech2_conformer](https://huggingface.co/espnet/fastspeech2_conformer) and
    FastSpeech2ConformerHifiGan
    [espnet/fastspeech2_conformer_hifigan](https://huggingface.co/espnet/fastspeech2_conformer_hifigan) architectures.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_config (`typing.Dict`, *optional*):
            Configuration of the text-to-speech model.
        vocoder_config (`typing.Dict`, *optional*):
            Configuration of the vocoder model.
    model_config ([`FastSpeech2ConformerConfig`], *optional*):
        Configuration of the text-to-speech model.
    vocoder_config ([`FastSpeech2ConformerHiFiGanConfig`], *optional*):
        Configuration of the vocoder model.

    Example:

    ```python
    >>> from transformers import (
    ...     FastSpeech2ConformerConfig,
    ...     FastSpeech2ConformerHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGan,
    ... )

    >>> # Initializing FastSpeech2ConformerWithHifiGan sub-modules configurations.
    >>> model_config = FastSpeech2ConformerConfig()
    >>> vocoder_config = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a FastSpeech2ConformerWithHifiGan module style configuration
    >>> configuration = FastSpeech2ConformerWithHifiGanConfig(model_config.to_dict(), vocoder_config.to_dict())

    >>> # Initializing a model (with random weights)
    >>> model = FastSpeech2ConformerWithHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "fastspeech2_conformer_with_hifigan"
    sub_configs = {"model_config": FastSpeech2ConformerConfig, "vocoder_config": FastSpeech2ConformerHifiGanConfig}

    def __init__(
        self,
        model_config: Dict = None,
        vocoder_config: Dict = None,
        **kwargs,
    ):
        if model_config is None:
            model_config = {}
            logger.info("model_config is None. initializing the model with default values.")

        if vocoder_config is None:
            vocoder_config = {}
            logger.info("vocoder_config is None. initializing the coarse model with default values.")

        self.model_config = FastSpeech2ConformerConfig(**model_config)
        self.vocoder_config = FastSpeech2ConformerHifiGanConfig(**vocoder_config)

        super().__init__(**kwargs)


__all__ = ["FastSpeech2ConformerConfig", "FastSpeech2ConformerHifiGanConfig", "FastSpeech2ConformerWithHifiGanConfig"]

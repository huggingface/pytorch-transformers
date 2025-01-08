# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Speech2Text model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Speech2TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Speech2TextModel`]. It is used to instantiate a
    Speech2Text model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Speech2Text
    [facebook/s2t-small-librispeech-asr](https://huggingface.co/facebook/s2t-small-librispeech-asr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 10000):
            Vocabulary size of the Speech2Text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`Speech2TextModel`]
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](https://arxiv.org/abs/1909.11556) for
            more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](https://arxiv.org/abs/1909.11556) for
            more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models).
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is set up as an encoder-decoder architecture for sequence-to-sequence tasks.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        d_model (`int`, *optional*, defaults to 256):
            Dimensionality of the layers and the pooler layer.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        decoder_start_token_id (`int`, *optional*, defaults to 2):
            The initial token ID of the decoder when decoding sequences.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Whether the embeddings are scaled by the square root of `d_model`.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end-of-sequence token.
        max_source_positions (`int`, *optional*, defaults to 6000):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        max_target_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        num_conv_layers (`int`, *optional*, defaults to 2):
            Number of 1D convolutional layers in the conv module.
        conv_kernel_sizes (`Tuple[int]`, *optional*, defaults to `(5, 5)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the conv module. The length
            of `conv_kernel_sizes` has to match `num_conv_layers`.
        conv_channels (`int`, *optional*, defaults to 1024):
            An integer defining the number of output channels of each convolution layers except the final one in the
            conv module.
        input_feat_per_channel (`int`, *optional*, defaults to 80):
            An integer specifying the size of feature vector. This is also the dimensions of log-mel filter-bank
            features.
        input_channels (`int`, *optional*, defaults to 1):
            An integer specifying number of input channels of the input feature vector.

    Example:

    ```python
    >>> from transformers import Speech2TextConfig, Speech2TextModel

    >>> # Initializing a Speech2Text s2t_transformer_s style configuration
    >>> configuration = Speech2TextConfig()

    >>> # Initializing a model (with random weights) from the s2t_transformer_s style configuration
    >>> model = Speech2TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "speech_to_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=10000,
        encoder_layers=12,
        encoder_ffn_dim=2048,
        encoder_attention_heads=4,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=4,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="relu",
        d_model=256,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=2,
        scale_embedding=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        max_source_positions=6000,
        max_target_positions=1024,
        num_conv_layers=2,
        conv_kernel_sizes=(5, 5),
        conv_channels=1024,
        input_feat_per_channel=80,
        input_channels=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.num_conv_layers = num_conv_layers
        self.conv_kernel_sizes = list(conv_kernel_sizes)
        self.conv_channels = conv_channels
        self.input_feat_per_channel = input_feat_per_channel
        self.input_channels = input_channels

        if len(self.conv_kernel_sizes) != self.num_conv_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.conv_kernel_sizes)` == `config.num_conv_layers` "
                f"but is `len(config.conv_kernel_sizes) = {len(self.conv_kernel_sizes)}`, "
                f"`config.num_conv_layers = {self.num_conv_layers}`."
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )


__all__ = ["Speech2TextConfig"]

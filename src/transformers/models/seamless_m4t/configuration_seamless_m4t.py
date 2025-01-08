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
"""SeamlessM4T model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class SeamlessM4TConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~SeamlessM4TModel`]. It is used to instantiate an
    SeamlessM4T model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SeamlessM4T
    ["facebook/hf-seamless-m4t-medium"](https://huggingface.co/"facebook/hf-seamless-m4t-medium") architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 256102):
            Vocabulary size of the SeamlessM4T model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`~SeamlessM4TModel`], [`~SeamlessM4TForTextToSpeech`] or
            [`~SeamlessM4TForTextToText`].
        t2u_vocab_size (`int`, *optional*, defaults to 10082):
            Unit vocabulary size of the SeamlessM4T model. Defines the number of different unit tokens that can be
            represented by the `inputs_ids` passed when calling the Text-To-Units sub-model of [`~SeamlessM4TModel`],
            [`~SeamlessM4TForSpeechToSpeech`] or [`~SeamlessM4TForTextToSpeech`].

        > Parameters shared across sub-models

        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the "intermediate" layers in the architecture.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model text encoder and decoder might ever be used with. Typically set
            this to something large just in case (e.g., 512 or 1024 or 2048).
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.
        encoder_layerdrop (`float`, *optional*, defaults to 0.05):
            The LayerDrop probability for the encoders. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layerdrop (`float`, *optional*, defaults to 0.05):
            The LayerDrop probability for the decoders. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the decoder and feed-forward layers. If string,
            `"gelu"`, `"relu"`, `"selu"`, `"swish"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, decoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all attention layers.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all activation layers in the model.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Scale embeddings by diving by sqrt(d_model).

        > Text encoder and text decoder specific parameters

        encoder_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer text encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 8192):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer text encoder.
        decoder_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer text decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 8192):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer text decoder.
        decoder_start_token_id (`int`, *optional*, defaults to 3):
            If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token. Only
            applied in the text decoder.
        max_new_tokens (`int`, *optional*, defaults to 256):
            The maximum numbers of text tokens to generate, ignoring the number of tokens in the prompt.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the _padding_ text token. Only applied to the text-decoder model.
        bos_token_id (`int`, *optional*, defaults to 2):
            The id of the _beginning-of-stream_ text token. Only applied to the text-decoder model.
        eos_token_id (`int`, *optional*, defaults to 3):
            The id of the _end-of-stream_ text token. Only applied to the text-decoder model.

        > Speech encoder specific parameters

        speech_encoder_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer speech encoder.
        speech_encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer speech encoder.
        speech_encoder_intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer speech encoder.
        speech_encoder_hidden_act (`str` or `function`, *optional*, defaults to `"swish"`):
            The non-linear activation function (function or string) in the speech encoder. If string, `"gelu"`,
            `"relu"`, `"selu"`, `"swish"` and `"gelu_new"` are supported.
        speech_encoder_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all layers in the speech encoder.
        add_adapter (`bool`, *optional*, defaults to `True`):
            Add an adapter layer on top of the speech encoder.
        speech_encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability for the speech encoder. See the [LayerDrop paper](see
            https://arxiv.org/abs/1909.11556) for more details.
        feature_projection_input_dim (`int`, *optional*, defaults to 160):
            Input dimension of the input feature projection of the speech encoder, i.e the dimension after processing
            input audios with [`SeamlessM4TFeatureExtractor`].
        num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer of the speech encoder.
        num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer of the speech encoder.
        adaptor_kernel_size (`int`, *optional*, defaults to 8):
            Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        adaptor_stride (`int`, *optional*, defaults to 8):
            Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        adaptor_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all layers in the speech adapter.
        num_adapter_layers (`int`, *optional*, defaults to 1):
            Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is
            True`.
        position_embeddings_type (`str`, *optional*, defaults to `"relative"`):
            Can be specified to `relative` or `rotary` for relative or rotary position embeddings respectively. If left
            `None` no relative position embedding is applied. Only applied to the speech encoder.
        rotary_embedding_base (`int`, *optional*, defaults to 10000):
            If `"rotary"` position embeddings are used, defines the size of the embedding base. Only applied to the
            speech encoder.
        max_source_positions (`int`, *optional*, defaults to 4096):
            if `"relative"` position embeddings are used, defines the maximum source input positions. Only applied to
            the speech encoder.
        conv_depthwise_kernel_size (`int`, *optional*, defaults to 31):
            Kernel size of convolutional depthwise 1D layer in Conformer blocks. Only applied to the speech encoder.

        > Text-To-Unit (t2u) model specific parameters

        t2u_bos_token_id (`int`, *optional*, defaults to 0):
            The id of the _beginning-of-stream_ unit token. Only applied to the text-to-unit seq2seq model.
        t2u_pad_token_id (`int`, *optional*, defaults to 1):
            The id of the _padding_ unit token. Only applied to the text-to-unit seq2seq model.
        t2u_eos_token_id (`int`, *optional*, defaults to 2):
            The id of the _end-of-stream_ unit token. Only applied to the text-to-unit seq2seq model.
        t2u_decoder_start_token_id (`int`, *optional*, defaults to 2):
            If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token. Only
            applied to the text-to-unit seq2seq model.
        t2u_max_new_tokens (`int`, *optional*, defaults to 1024):
            The maximum numbers of unit tokens to generate, ignoring the number of tokens in the prompt. Only applied
            to the text-to-unit seq2seq model.
        t2u_encoder_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer text-to-unit encoder.
        t2u_encoder_ffn_dim (`int`, *optional*, defaults to 8192):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text-to-unit encoder.
        t2u_encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer text-to-unit encoder.
        t2u_decoder_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer text-to-unit decoder.
        t2u_decoder_ffn_dim (`int`, *optional*, defaults to 8192):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text-to-unit decoder.
        t2u_decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer text-to-unit decoder.
        t2u_max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model text-to-unit component might ever be used with. Typically set
            this to something large just in case (e.g., 512 or 1024 or 2048).

         > Hifi-Gan Vocoder specific parameters

        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the output audio will be generated, expressed in hertz (Hz).
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the hifi-gan upsampling network. Applies to the vocoder only.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[5, 4, 4, 2, 2]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the vocoder upsampling network.
            The length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*. Applies to the vocoder only.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[11, 8, 8, 4, 4]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the vocoder upsampling
            network. The length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match
            the length of *upsample_rates*. Applies to the vocoder only.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the vocoder 1D convolutional layers in the multi-receptive
            field fusion (MRF) module. Applies to the vocoder only.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the vocoder dilated 1D convolutional layers in
            the multi-receptive field fusion (MRF) module. Applies to the vocoder only.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation in the vocoder. Applies to the vocoder
            only.
        unit_hifi_gan_vocab_size (`int`, *optional*, defaults to 10000):
            Vocabulary size of the SeamlessM4T vocoder. Defines the number of different unit tokens that can be
            represented by the `inputs_ids` passed when calling the vocoder of [`~SeamlessM4TModel`],
            [`~SeamlessM4TForSpeechToSpeech`] or [`~SeamlessM4TForTextToSpeech`].
        unit_embed_dim (`int`, *optional*, defaults to 1280):
            The projection dimension of the input ids given to the hifi-gan vocoder. Applies to the vocoder only.
        lang_embed_dim (`int`, *optional*, defaults to 256):
            The projection dimension of the target language given to the hifi-gan vocoder. Applies to the vocoder only.
        spkr_embed_dim (`int`, *optional*, defaults to 256):
            The projection dimension of the speaker id given to the hifi-gan vocoder. Applies to the vocoder only.
        vocoder_num_langs (`int`, *optional*, defaults to 36):
            Number of langs supported by the vocoder. Might be different from `t2u_num_langs`.
        vocoder_num_spkrs (`int`, *optional*, defaults to 200):
            Number of speakers supported by the vocoder.
        variance_predictor_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the duration predictor. Applies to the vocoder only.
        var_pred_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability of the duration predictor. Applies to the vocoder only.
        vocoder_offset (`int`, *optional*, defaults to 4):
            Offset the unit token ids by this number to account for symbol tokens. Applies to the vocoder only.

    ```python
    >>> from transformers import SeamlessM4TModel, SeamlessM4TConfig

    >>> # Initializing a SeamlessM4T "facebook/hf-seamless-m4t-medium" style configuration
    >>> configuration = SeamlessM4TConfig()

    >>> # Initializing a model from the "facebook/hf-seamless-m4t-medium" style configuration
    >>> model = SeamlessM4TModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "seamless_m4t"

    def __init__(
        self,
        vocab_size=256102,
        t2u_vocab_size=10082,
        # shared config
        hidden_size=1024,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        max_position_embeddings=1024,
        is_encoder_decoder=True,
        encoder_layerdrop=0.05,
        decoder_layerdrop=0.05,
        activation_function="relu",
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        scale_embedding=True,
        # text encoder|decoder
        encoder_layers=24,
        encoder_ffn_dim=8192,
        encoder_attention_heads=16,
        decoder_layers=24,
        decoder_ffn_dim=8192,
        decoder_attention_heads=16,
        decoder_start_token_id=3,
        max_new_tokens=256,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        # speech_encoder
        speech_encoder_layers=24,
        speech_encoder_attention_heads=16,
        speech_encoder_intermediate_size=4096,
        speech_encoder_hidden_act="swish",
        speech_encoder_dropout=0.0,
        add_adapter=True,
        speech_encoder_layerdrop=0.1,
        feature_projection_input_dim=160,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_dropout=0.1,
        num_adapter_layers=1,
        position_embeddings_type="relative",
        rotary_embedding_base=10000,
        max_source_positions=4096,
        conv_depthwise_kernel_size=31,
        # t2u config
        t2u_bos_token_id=0,
        t2u_pad_token_id=1,
        t2u_eos_token_id=2,
        t2u_decoder_start_token_id=2,
        t2u_max_new_tokens=1024,
        t2u_encoder_layers=6,
        t2u_encoder_ffn_dim=8192,
        t2u_encoder_attention_heads=16,
        t2u_decoder_layers=6,
        t2u_decoder_ffn_dim=8192,
        t2u_decoder_attention_heads=16,
        t2u_max_position_embeddings=2048,
        # hifi-gan vocoder config
        sampling_rate=16000,
        upsample_initial_channel=512,
        upsample_rates=[5, 4, 4, 2, 2],
        upsample_kernel_sizes=[11, 8, 8, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        leaky_relu_slope=0.1,
        # specific to Code Hifi-Gan
        unit_hifi_gan_vocab_size=10000,
        unit_embed_dim=1280,
        lang_embed_dim=256,
        spkr_embed_dim=256,
        vocoder_num_langs=36,
        vocoder_num_spkrs=200,
        variance_predictor_kernel_size=3,
        var_pred_dropout=0.5,
        vocoder_offset=4,
        **kwargs,
    ):
        # overall_config
        self.vocab_size = vocab_size
        self.t2u_vocab_size = t2u_vocab_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.max_new_tokens = max_new_tokens
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.scale_embedding = scale_embedding
        # for proper config init
        self.num_attention_heads = decoder_attention_heads
        self.num_hidden_layers = decoder_layers

        # text|unit encoder|decoder
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads

        # speech_encoder
        self.speech_encoder_layers = speech_encoder_layers
        self.speech_encoder_hidden_act = speech_encoder_hidden_act
        self.speech_encoder_dropout = speech_encoder_dropout
        self.speech_encoder_attention_heads = speech_encoder_attention_heads
        self.speech_encoder_layerdrop = speech_encoder_layerdrop
        self.speech_encoder_intermediate_size = speech_encoder_intermediate_size
        self.feature_projection_input_dim = feature_projection_input_dim
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.adaptor_kernel_size = adaptor_kernel_size
        self.adaptor_stride = adaptor_stride
        self.adaptor_dropout = adaptor_dropout
        self.num_adapter_layers = num_adapter_layers
        self.position_embeddings_type = position_embeddings_type
        self.rotary_embedding_base = rotary_embedding_base
        self.max_source_positions = max_source_positions
        self.conv_depthwise_kernel_size = conv_depthwise_kernel_size
        self.add_adapter = add_adapter

        # t2u config
        self.t2u_bos_token_id = t2u_bos_token_id
        self.t2u_pad_token_id = t2u_pad_token_id
        self.t2u_eos_token_id = t2u_eos_token_id
        self.t2u_decoder_start_token_id = t2u_decoder_start_token_id
        self.t2u_max_new_tokens = t2u_max_new_tokens
        self.t2u_encoder_layers = t2u_encoder_layers
        self.t2u_encoder_ffn_dim = t2u_encoder_ffn_dim
        self.t2u_encoder_attention_heads = t2u_encoder_attention_heads
        self.t2u_decoder_layers = t2u_decoder_layers
        self.t2u_decoder_ffn_dim = t2u_decoder_ffn_dim
        self.t2u_decoder_attention_heads = t2u_decoder_attention_heads
        self.t2u_max_position_embeddings = t2u_max_position_embeddings

        # hifi-gan vocoder config
        # original parameters specific to Hifi-Gan
        self.sampling_rate = sampling_rate
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.leaky_relu_slope = leaky_relu_slope

        # specific to Code Hifi-Gan
        self.unit_hifi_gan_vocab_size = unit_hifi_gan_vocab_size
        self.unit_embed_dim = unit_embed_dim
        self.lang_embed_dim = lang_embed_dim
        self.spkr_embed_dim = spkr_embed_dim
        self.vocoder_num_langs = vocoder_num_langs
        self.vocoder_num_spkrs = vocoder_num_spkrs
        self.variance_predictor_kernel_size = variance_predictor_kernel_size
        self.var_pred_dropout = var_pred_dropout
        self.vocoder_offset = vocoder_offset

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            max_position_embeddings=max_position_embeddings,
            **kwargs,
        )


__all__ = ["SeamlessM4TConfig"]

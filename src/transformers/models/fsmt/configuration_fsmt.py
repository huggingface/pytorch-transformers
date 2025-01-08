# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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
"""FSMT configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class DecoderConfig(PretrainedConfig):
    r"""
    Configuration class for FSMT's decoder specific things. note: this is a private helper class
    """

    model_type = "fsmt_decoder"

    def __init__(self, vocab_size=0, bos_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id


class FSMTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FSMTModel`]. It is used to instantiate a FSMT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the FSMT
    [facebook/wmt19-en-ru](https://huggingface.co/facebook/wmt19-en-ru) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        langs (`List[str]`):
            A list with source language and target_language (e.g., ['en', 'ru']).
        src_vocab_size (`int`):
            Vocabulary size of the encoder. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method in the encoder.
        tgt_vocab_size (`int`):
            Vocabulary size of the decoder. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed to the forward method in the decoder.
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `Callable`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Scale embeddings by diving by sqrt(d_model).
        bos_token_id (`int`, *optional*, defaults to 0)
            Beginning of stream token id.
        pad_token_id (`int`, *optional*, defaults to 1)
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 2)
            End of stream token id.
        decoder_start_token_id (`int`, *optional*):
            This model starts decoding with `eos_token_id`
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            Google "layerdrop arxiv", as its not explainable in one line.
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            Google "layerdrop arxiv", as its not explainable in one line.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether this is an encoder/decoder model.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        num_beams (`int`, *optional*, defaults to 5)
            Number of beams for beam search that will be used by default in the `generate` method of the model. 1 means
            no beam search.
        length_penalty (`float`, *optional*, defaults to 1)
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        early_stopping (`bool`, *optional*, defaults to `False`)
            Flag that will be used by default in the `generate` method of the model. Whether to stop the beam search
            when at least `num_beams` sentences are finished per batch or not.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        forced_eos_token_id (`int`, *optional*, defaults to 2):
            The id of the token to force as the last generated token when `max_length` is reached. Usually set to
            `eos_token_id`.

    Examples:

    ```python
    >>> from transformers import FSMTConfig, FSMTModel

    >>> # Initializing a FSMT facebook/wmt19-en-ru style configuration
    >>> config = FSMTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FSMTModel(config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fsmt"
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    # update the defaults from config file
    def __init__(
        self,
        langs=["en", "de"],
        src_vocab_size=42024,
        tgt_vocab_size=42024,
        activation_function="relu",
        d_model=1024,
        max_length=200,
        max_position_embeddings=1024,
        encoder_ffn_dim=4096,
        encoder_layers=12,
        encoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_ffn_dim=4096,
        decoder_layers=12,
        decoder_attention_heads=16,
        decoder_layerdrop=0.0,
        attention_dropout=0.0,
        dropout=0.1,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=2,
        is_encoder_decoder=True,
        scale_embedding=True,
        tie_word_embeddings=False,
        num_beams=5,
        length_penalty=1.0,
        early_stopping=False,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        forced_eos_token_id=2,
        **common_kwargs,
    ):
        self.langs = langs
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model  # encoder_embed_dim and decoder_embed_dim

        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)
        self.activation_function = activation_function

        self.decoder = DecoderConfig(vocab_size=tgt_vocab_size, bos_token_id=eos_token_id)
        if "decoder" in common_kwargs:
            del common_kwargs["decoder"]

        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=tie_word_embeddings,
            forced_eos_token_id=forced_eos_token_id,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            **common_kwargs,
        )


__all__ = ["FSMTConfig"]

# coding=utf-8
# Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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
"""ProphetNet model configuration"""

from typing import Callable, Optional, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class ProphetNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ProphetNetModel`]. It is used to instantiate a
    ProphetNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ProphetNet
    [microsoft/prophetnet-large-uncased](https://huggingface.co/microsoft/prophetnet-large-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the ProphetNET model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`ProphetNetModel`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        num_encoder_layers (`int`, *optional*, defaults to 12):
            Number of encoder layers.
        num_encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the `intermediate` (often named feed-forward) layer in decoder.
        num_decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        num_decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        add_cross_attention (`bool`, *optional*, defaults to `True`):
            Whether cross-attention layers should be added to the model.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether this is an encoder/decoder model.
        pad_token_id (`int`, *optional*, defaults to 1)
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0)
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2)
            End of stream token id.
        ngram (`int`, *optional*, defaults to 2)
            Number of future tokens to predict. Set to 1 to be same as traditional Language model to predict next first
            token.
        num_buckets (`int`, *optional*, defaults to 32)
            The number of buckets to use for each attention layer. This is for relative position calculation. See the
            [T5 paper](see https://arxiv.org/abs/1910.10683) for more details.
        relative_max_distance (`int`, *optional*, defaults to 128)
            Relative distances greater than this number will be put into the last same bucket. This is for relative
            position calculation. See the [T5 paper](see https://arxiv.org/abs/1910.10683) for more details.
        disable_ngram_loss (`bool`, *optional*, defaults to `False`):
            Whether be trained predicting only the next first token.
        eps (`float`, *optional*, defaults to 0.0):
            Controls the `epsilon` parameter value for label smoothing in the loss calculation. If set to 0, no label
            smoothing is performed.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """

    model_type = "prophetnet"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "num_encoder_attention_heads",
    }

    def __init__(
        self,
        activation_dropout: Optional[float] = 0.1,
        activation_function: Optional[Union[str, Callable]] = "gelu",
        vocab_size: Optional[int] = 30522,
        hidden_size: Optional[int] = 1024,
        encoder_ffn_dim: Optional[int] = 4096,
        num_encoder_layers: Optional[int] = 12,
        num_encoder_attention_heads: Optional[int] = 16,
        decoder_ffn_dim: Optional[int] = 4096,
        num_decoder_layers: Optional[int] = 12,
        num_decoder_attention_heads: Optional[int] = 16,
        attention_dropout: Optional[float] = 0.1,
        dropout: Optional[float] = 0.1,
        max_position_embeddings: Optional[int] = 512,
        init_std: Optional[float] = 0.02,
        is_encoder_decoder: Optional[bool] = True,
        add_cross_attention: Optional[bool] = True,
        decoder_start_token_id: Optional[int] = 0,
        ngram: Optional[int] = 2,
        num_buckets: Optional[int] = 32,
        relative_max_distance: Optional[int] = 128,
        disable_ngram_loss: Optional[bool] = False,
        eps: Optional[float] = 0.0,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = 0,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_attention_heads = num_encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_attention_heads = num_decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)
        self.activation_function = activation_function

        # parameters for prophetnet
        self.ngram = ngram
        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.disable_ngram_loss = disable_ngram_loss
        self.eps = eps

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            add_cross_attention=add_cross_attention,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )

    @property
    def num_hidden_layers(self) -> int:
        return self.num_encoder_layers + self.num_decoder_layers

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        raise NotImplementedError(
            "This model does not support the setting of `num_hidden_layers`. Please set `num_encoder_layers` and"
            " `num_decoder_layers`."
        )


__all__ = ["ProphetNetConfig"]

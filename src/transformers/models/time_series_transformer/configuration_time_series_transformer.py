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
""" Time Series Transformer model configuration"""
from typing import List, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/tst-ett": "https://huggingface.co/huggingface/tst-ett/resolve/main/config.json",
    # See all TimeSeriesTransformer models at https://huggingface.co/models?filter=time_series_transformer
}


class TimeSeriesTransformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TimeSeriesTransformerModel`]. It is used to
    instantiate a Time Series Transformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Time Series
    Transformer [huggingface/tst-ett](https://huggingface.co/huggingface/tst-ett) architecture.

    Configuration objects inherit from [`PretrainedConfig`] can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        prediction_length (`int`):
            The prediction horizon for the model.
        context_length (`int`, *optional*):
            The context length for the encoder. If `None`, the context length will be the same as the
            `prediction_length`.
        distribution_output (`string`, *optional* defaults to `"student_t"`):
            The distribution emission head for the model.
        loss (`string`, *optional* defaults to `"nll"`):
            The loss function for the model corresponding to the `distribution_output` head. For parametric
            distributions it is negative log likelihood.
        input_size (`int`, *optional* defaults to 1):
            The size of the target variable which by default is 1 for univariate targets.
        scaling (`bool`, *optional* defaults to `True`):
            Whether to scale the input targets.
        lags_seq (`list` of `int`  *optional* defaults to `[1, 2, 3, 4, 5, 6, 7]`):
            The lags of the input time series as covariates often dictated by the frequency. Default is `[1, 2, 3, 4,
            5, 6, 7]`.
        num_time_features (`int`, *optional* defaults to 0):
            The number of time features in the input time series.
        num_feat_dynamic_real (`int`, *optional* defaults to 0):
            The number of dynamic real valued features.
        num_feat_static_cat (`int`, *optional* defaults to 0):
            The number of static categorical features.
        num_feat_static_real (`int`, *optional* defaults to 0):
            The number of static real valued features.
        cardinality (`list` of `int`, *optional*):
            The cardinality of the categorical features. Cannot be `None` if `num_feat_static_cat` is `> 0`.
        embedding_dimension (`list` of `int`, *optional*):
            The dimension of the embedding for the categorical features. Cannot be `None` if `num_feat_static_cat` is
            `> 0`.
        encoder_layers (`int`, *optional*, defaults to `2`):
            Number of encoder layers.
        decoder_layers (`int`, *optional*, defaults to `2`):
            Number of decoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to `2`):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (`int`, *optional*, defaults to `2`):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_ffn_dim (`int`, *optional*, defaults to `32`):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        decoder_ffn_dim (`int`, *optional*, defaults to `32`):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and decoder. If string, `"gelu"` and
            `"relu"` are supported.
        dropout (`float`, *optional*, defaults to `0.1`):
            The dropout probability for all fully connected layers in the encoder, and decoder.
        encoder_layerdrop (`float`, *optional*, defaults to `0.1`):
            The dropout probability for the attention and fully connected layers for each encoder layer.
        decoder_layerdrop (`float`, *optional*, defaults to `0.1`):
            The dropout probability for the attention and fully connected layers for each decoder layer.
        attention_dropout (`float`, *optional*, defaults to `0.1`):
            The dropout probability for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to `0.1`):
            The dropout probability used between the two layers of the feed-forward networks.
        num_parallel_samples (`int`, *optional*, defaults to `100`):
            The number of samples to generate in parallel for each time step of inference.
        init_std (`float`, *optional*, defaults to `0.02`):
            The standard deviation of the truncated normal weight initialization distribution.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.

        Example:

    ```python
    >>> from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

    >>> # Initializing a Time Series Transformer huggingface/tst-ett style configuration
    >>> configuration = TimeSeriesTransformerConfig()

    >>> # Initializing a model from the huggingface/tst-ett style configuration
    >>> model = TimeSeriesTransformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "time_series_transformer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }

    def __init__(
        self,
        input_size: int = 1,
        prediction_length: Optional[int] = None,
        context_length: Optional[int] = None,
        distribution_output: str = "student_t",
        loss: str = "nll",
        lags_seq: List[int] = [1, 2, 3, 4, 5, 6, 7],
        scaling: bool = True,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        num_time_features: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        encoder_ffn_dim: int = 32,
        decoder_ffn_dim: int = 32,
        encoder_attention_heads: int = 2,
        decoder_attention_heads: int = 2,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        is_encoder_decoder: bool = True,
        activation_function: str = "gelu",
        dropout: float = 0.1,
        encoder_layerdrop: float = 0.1,
        decoder_layerdrop: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        num_parallel_samples: int = 100,
        init_std: float = 0.02,
        use_cache=True,
        **kwargs
    ):
        # time series specific configuration
        self.prediction_length = prediction_length
        self.context_length = context_length or prediction_length
        self.distribution_output = distribution_output
        self.loss = loss
        self.input_size = input_size
        self.num_time_features = num_time_features
        self.lags_seq = lags_seq
        self.scaling = scaling
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_real = num_feat_static_real
        self.num_feat_static_cat = num_feat_static_cat
        self.cardinality = cardinality if cardinality and num_feat_static_cat > 0 else [1]
        self.embedding_dimension = embedding_dimension or [min(50, (cat + 1) // 2) for cat in self.cardinality]
        self.num_parallel_samples = num_parallel_samples

        # Transformer architecture configuration
        self.d_model = input_size * len(lags_seq) + self._number_of_features
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop

        self.activation_function = activation_function
        self.init_std = init_std

        self.output_attentions = False
        self.output_hidden_states = False

        self.use_cache = use_cache

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_time_features
            + max(1, self.num_feat_static_real)  # there is at least one dummy static real feature
            + 1  # the log(scale)
        )

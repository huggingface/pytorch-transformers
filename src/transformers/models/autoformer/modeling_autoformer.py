# coding=utf-8
# Copyright (c) 2021 THUML @ Tsinghua University
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""PyTorch Autoformer model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
    SampleTSPredictionOutput,
    Seq2SeqTSPredictionOutput,
)
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_autoformer import AutoformerConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "AutoformerConfig"


@dataclass
class AutoFormerDecoderOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        trend (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Trend tensor for each time series.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    trend: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class AutoformerModelOutput(ModelOutput):
    """
    Autoformer model output that contains the additional trend output.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        trend (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Trend tensor for each time series.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        loc (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*):
            Shift values of each time series' context window which is used to give the model inputs of the same
            magnitude and then used to shift back to the original magnitude.
        scale (`torch.FloatTensor` of shape `(batch_size,)` or `(batch_size, input_size)`, *optional*):
            Scaling values of each time series' context window which is used to give the model inputs of the same
            magnitude and then used to rescale back to the original magnitude.
        static_features: (`torch.FloatTensor` of shape `(batch_size, feature size)`, *optional*):
            Static features of each time series' in a batch which are copied to the covariates at inference time.
    """

    last_hidden_state: torch.FloatTensor = None
    trend: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None
    static_features: Optional[torch.FloatTensor] = None


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesFeatureEmbedder with TimeSeries->Autoformer
class AutoformerFeatureEmbedder(nn.Module):
    """
    Embed a sequence of categorical features.

    Args:
        cardinalities (`list[int]`):
            List of cardinalities of the categorical features.
        embedding_dims (`list[int]`):
            List of embedding dimensions of the categorical features.
    """

    def __init__(self, cardinalities: List[int], embedding_dims: List[int]) -> None:
        super().__init__()

        self.num_features = len(cardinalities)
        self.embedders = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, embedding_dims)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_features > 1:
            # we slice the last dimension, giving an array of length
            # self.num_features with shape (N,T) or (N)
            cat_feature_slices = torch.chunk(features, self.num_features, dim=-1)
        else:
            cat_feature_slices = [features]

        return torch.cat(
            [
                embed(cat_feature_slice.squeeze(-1))
                for embed, cat_feature_slice in zip(self.embedders, cat_feature_slices)
            ],
            dim=-1,
        )


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesStdScaler with TimeSeriesTransformer->Autoformer,TimeSeries->Autoformer
class AutoformerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesMeanScaler with TimeSeriesTransformer->Autoformer,TimeSeries->Autoformer
class AutoformerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesNOPScaler with TimeSeriesTransformer->Autoformer,TimeSeries->Autoformer
class AutoformerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return data, loc, scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.weighted_average
def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.

    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return input_tensor.mean(dim=dim)


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.nll
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->Autoformer
class AutoformerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesValueEmbedding with TimeSeries->Autoformer
class AutoformerValueEmbedding(nn.Module):
    def __init__(self, feature_size, d_model):
        super().__init__()
        self.value_projection = nn.Linear(in_features=feature_size, out_features=d_model, bias=False)

    def forward(self, x):
        return self.value_projection(x)


# Class based on
# https://github.com/thuml/Autoformer/blob/c6a0694ff484753f2d986cc0bb1f99ee850fc1a8/layers/Autoformer_EncDec.py#L39
# where AutoformerSeriesDecompositionLayer is series_decomp + moving_average
class AutoformerSeriesDecompositionLayer(nn.Module):
    """
    Returns the trend and the seasonal parts of the time series. Calculated as:

        x_trend = AvgPool(Padding(X)) and x_seasonal = X - x_trend
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.kernel_size = config.moving_average
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x):
        """Input shape: Batch x Time x EMBED_DIM"""
        # padding on the both ends of time series
        num_of_pads = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[:, -1:, :].repeat(1, num_of_pads, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # calculate the trend and seasonal part of the series
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend


# Class based on
# https://github.com/thuml/Autoformer/blob/c6a0694ff484753f2d986cc0bb1f99ee850fc1a8/layers/Autoformer_EncDec.py#L6
# where AutoformerLayernorm is my_Layernorm
class AutoformerLayernorm(nn.Module):
    """
    Special designed layer normalization for the seasonal part, calculated as: AutoformerLayernorm(x) = nn.LayerNorm(x)
    - torch.mean(nn.LayerNorm(x))
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class AutoformerAttention(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
        (1) period-based dependencies discovery (2) time delay aggregation
    This block replace the canonical self-attention mechanism.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        autocorrelation_factor: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.autocorrelation_factor = autocorrelation_factor

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        # (1) period-based dependencies discovery
        # Resize (truncation or zero filling)
        queries_time_length = query_states.size(1)
        values_time_length = value_states.size(1)
        if queries_time_length > values_time_length:
            query_states = query_states[:, : (queries_time_length - values_time_length), :]
            zeros = torch.zeros_like(query_states).float()
            value_states = torch.cat([value_states, zeros], dim=1)
            key_states = torch.cat([key_states, zeros], dim=1)
        else:
            value_states = value_states[:, :queries_time_length, :]
            key_states = key_states[:, :queries_time_length, :]

        query_states_fft = torch.fft.rfft(query_states, n=tgt_len, dim=1)
        key_states_fft = torch.fft.rfft(key_states, n=tgt_len, dim=1)
        attn_weights = query_states_fft * torch.conj(key_states_fft)
        attn_weights = torch.fft.irfft(attn_weights, n=tgt_len, dim=1)  # Autocorrelation(Q,K)

        src_len = key_states.size(1)
        channel = key_states.size(2)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, channel):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, channel)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, channel)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, channel)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, channel)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, channel)
        else:
            attn_weights_reshaped = None

        # time delay aggregation
        time_length = value_states.size(1)
        autocorrelations = attn_weights.view(bsz, self.num_heads, tgt_len, channel)

        # find top k autocorrelations delays
        top_k = int(self.autocorrelation_factor * math.log(time_length))
        autocorrelations_mean_on_head_channel = torch.mean(autocorrelations, dim=(1, -1))  # bsz x tgt_len
        if self.training:
            autocorrelations_mean_on_bsz = torch.mean(autocorrelations_mean_on_head_channel, dim=0)
            _, top_k_delays_index = torch.topk(autocorrelations_mean_on_bsz, top_k)
            top_k_autocorrelations = torch.stack(
                [autocorrelations_mean_on_head_channel[:, top_k_delays_index[i]] for i in range(top_k)], dim=-1
            )
        else:
            top_k_autocorrelations, top_k_delays_index = torch.topk(
                autocorrelations_mean_on_head_channel, top_k, dim=1
            )

        top_k_autocorrelations = torch.softmax(top_k_autocorrelations, dim=-1)  # bsz x top_k

        # compute aggregation: value_states.roll(delay) * top_k_autocorrelations(delay)
        if not self.training:
            # used for compute values_states.roll(delay) in inference
            tmp_values = value_states.repeat(1, 2, 1)
            init_index = (
                torch.arange(time_length)
                .view(1, -1, 1)
                .repeat(bsz * self.num_heads, 1, channel)
                .to(value_states.device)
            )

        delays_agg = torch.zeros_like(value_states).float()  # bsz x time_length x channel
        for i in range(top_k):
            # compute value_states roll delay
            if not self.training:
                tmp_delay = init_index + top_k_delays_index[:, i].view(-1, 1, 1).repeat(
                    self.num_heads, tgt_len, channel
                )
                value_states_roll_delay = torch.gather(tmp_values, dim=1, index=tmp_delay)
            else:
                value_states_roll_delay = value_states.roll(shifts=-int(top_k_delays_index[i]), dims=1)

            # aggregation
            top_k_autocorrelations_at_delay = (
                top_k_autocorrelations[:, i].view(-1, 1, 1).repeat(self.num_heads, tgt_len, channel)
            )
            delays_agg += value_states_roll_delay * top_k_autocorrelations_at_delay

        attn_output = delays_agg.contiguous()

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class AutoformerEncoderLayer(nn.Module):
    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = AutoformerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            autocorrelation_factor=config.autocorrelation_factor,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = AutoformerLayernorm(config)
        self.decomp1 = AutoformerSeriesDecompositionLayer(config)
        self.decomp2 = AutoformerSeriesDecompositionLayer(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # added layer norm here as an improvement
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.decomp1(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states, _ = self.decomp2(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class AutoformerDecoderLayer(nn.Module):
    def __init__(self, config: AutoformerConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = AutoformerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            autocorrelation_factor=config.autocorrelation_factor,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = AutoformerAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            autocorrelation_factor=config.autocorrelation_factor,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = AutoformerLayernorm(config)

        self.decomp1 = AutoformerSeriesDecompositionLayer(config)
        self.decomp2 = AutoformerSeriesDecompositionLayer(config)
        self.decomp3 = AutoformerSeriesDecompositionLayer(config)

        # source: https://github.com/thuml/Autoformer/blob/e6371e24f2ae2dd53e472edefdd5814c5176f864/layers/Autoformer_EncDec.py#L128
        self.trend_projection = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=config.feature_size,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache: (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the `present_key_value` state to be used for subsequent
                decoding.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states, trend1 = self.decomp1(hidden_states)
        # added layer norm here as an improvement
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states, trend2 = self.decomp2(hidden_states)
            # added layer norm here as an improvement
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states, trend3 = self.decomp3(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        if encoder_hidden_states is not None:
            residual_trend = trend1 + trend2 + trend3
        else:
            residual_trend = trend1 + trend3
        residual_trend = self.trend_projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        outputs = ((hidden_states, residual_trend),)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class AutoformerPreTrainedModel(PreTrainedModel):
    config_class = AutoformerConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, AutoformerSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


AUTOFORMER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`AutoformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

AUTOFORMER_INPUTS_DOCSTRING = r"""
    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Past values of the time series, that serve as context in order to predict the future. These values may
            contain lags, i.e. additional values from the past which are added in order to serve as "extra context".
            The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
            `static_categorical_features`, `static_real_features`, `past_time_features`).

            The sequence length here is equal to `context_length` + `max(config.lags_sequence)`.

            Missing values need to be replaced with zeros.

        past_time_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`, *optional*):
            Optional time features, which the model internally will add to `past_values`. These could be things like
            "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These
            could also be so-called "age" features, which basically help the model know "at which point in life" a
            time-series is. Age features have small values for distant past time steps and increase monotonically the
            more we approach the current time step.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional time features.

            The Autoformer only learns additional embeddings for `static_categorical_features`.

        past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected in
            `[0, 1]`:

            - 1 for values that are **observed**,
            - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        static_categorical_features (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*):
            Optional static categorical features for which the model will learn an embedding, which it will add to the
            values of the time series.

            Static categorical features are features which have the same value for all time steps (static over time).

            A typical example of a static categorical feature is a time series ID.

        static_real_features (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*):
            Optional static real features which the model will add to the values of the time series.

            Static real features are features which have the same value for all time steps (static over time).

            A typical example of a static real feature is promotion information.

        future_values (`torch.FloatTensor` of shape `(batch_size, prediction_length)`):
            Future values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs to learn to output, given the `past_values`.

            See the demo notebook and code snippets for details.

            Missing values need to be replaced with zeros.

        future_time_features (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`, *optional*):
            Optional time features, which the model internally will add to `future_values`. These could be things like
            "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These
            could also be so-called "age" features, which basically help the model know "at which point in life" a
            time-series is. Age features have small values for distant past time steps and increase monotonically the
            more we approach the current time step.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional features.

            The Autoformer only learns additional embeddings for `static_categorical_features`.

        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on certain token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
            make sure the model can only look at previous inputs in order to predict the future.

        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of `last_hidden_state`, `hidden_states` (*optional*) and `attentions` (*optional*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` (*optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesTransformerEncoder with TimeSeriesTransformer->Autoformer,TimeSeries->Autoformer
class AutoformerEncoder(AutoformerPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`AutoformerEncoderLayer`].

    Args:
        config: AutoformerConfig
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        self.value_embedding = AutoformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        self.embed_positions = AutoformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        self.layers = nn.ModuleList([AutoformerEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(inputs_embeds.size())

        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class AutoformerDecoder(AutoformerPreTrainedModel):
    """
    Transformer decoder consisting of `config.decoder_layers` layers. Each layer is a [`AutoformerDecoderLayer`]

    Args:
        config: AutoformerConfig
    """

    def __init__(self, config: AutoformerConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        if config.prediction_length is None:
            raise ValueError("The `prediction_length` config needs to be specified.")

        self.value_embedding = AutoformerValueEmbedding(feature_size=config.feature_size, d_model=config.d_model)
        self.embed_positions = AutoformerSinusoidalPositionalEmbedding(
            config.context_length + config.prediction_length, config.d_model
        )
        self.layers = nn.ModuleList([AutoformerDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # https://github.com/thuml/Autoformer/blob/e6371e24f2ae2dd53e472edefdd5814c5176f864/models/Autoformer.py#L74
        self.seasonality_projection = nn.Linear(config.d_model, config.feature_size)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        trend: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AutoFormerDecoderOutput]:
        r"""
        Args:
            trend (`torch.FloatTensor` of shape `(batch_size, prediction_length, feature_size)`, *optional*):
                The trend sequence to be fed to the decoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            use_cache (`bool`, *optional*):
                If `use_cache` is True, `past_key_values` key value states are returned and can be used to speed up
                decoding (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = inputs_embeds.size()[:-1]

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        hidden_states = self.value_embedding(inputs_embeds)
        embed_pos = self.embed_positions(
            inputs_embeds.size(), past_key_values_length=self.config.context_length - self.config.label_length
        )
        hidden_states = self.layernorm_embedding(hidden_states + embed_pos)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            (hidden_states, residual_trend) = layer_outputs[0]
            trend = trend + residual_trend

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # project seasonality representation
        hidden_states = self.seasonality_projection(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, trend, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return AutoFormerDecoderOutput(
            last_hidden_state=hidden_states,
            trend=trend,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare Autoformer Model outputting raw hidden-states without any specific head on top.",
    AUTOFORMER_START_DOCSTRING,
)
class AutoformerModel(AutoformerPreTrainedModel):
    def __init__(self, config: AutoformerConfig):
        super().__init__(config)

        if config.scaling == "mean" or config.scaling is True:
            self.scaler = AutoformerMeanScaler(config)
        elif config.scaling == "std":
            self.scaler = AutoformerStdScaler(config)
        else:
            self.scaler = AutoformerNOPScaler(config)

        if config.num_static_categorical_features > 0:
            self.embedder = AutoformerFeatureEmbedder(
                cardinalities=config.cardinality, embedding_dims=config.embedding_dimension
            )

        # transformer encoder-decoder and mask initializer
        self.encoder = AutoformerEncoder(config)
        self.decoder = AutoformerDecoder(config)

        # used for decoder seasonal and trend initialization
        self.decomposition_layer = AutoformerSeriesDecompositionLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def _past_length(self) -> int:
        return self.config.context_length + max(self.config.lags_sequence)

    def get_lagged_subsequences(
        self, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence. Returns a tensor of shape (batch_size, subsequences_length,
        feature_size, indices_length), containing lagged subsequences. Specifically, lagged[i, j, :, k] = sequence[i,
        -indices[k]-subsequences_length+j, :].

        Args:
            sequence (`torch.Tensor` or shape `(batch_size, context_length,
                feature_size)`): The sequence from which lagged subsequences should be extracted.
            subsequences_length (`int`):
                Length of the subsequences to be extracted.
            shift (`int`, *optional* defaults to 0):
                Shift the lags by this amount back in the time index.
        """

        # calculates the indices of the lags by subtracting the shift value from the given lags_sequence
        indices = [lag - shift for lag in self.config.lags_sequence]

        # checks if the maximum lag plus the length of the subsequences exceeds the length of the input sequence
        sequence_length = sequence.shape[1]
        if max(indices) + subsequences_length > sequence_length:
            raise ValueError(
                f"lags cannot go further than history length, found lag {max(indices)} "
                f"while history length is only {sequence_length}"
            )

        # extracts the lagged subsequences from the input sequence using the calculated indices
        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])

        # return as stacked tensor in the feature dimension
        return torch.stack(lagged_values, dim=-1)

    def create_network_inputs(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates the inputs for the network given the past and future values, time features, and static features.

        Args:
            past_values (`torch.Tensor`):
                A tensor of shape `(batch_size, past_length, input_size)` containing the past values.
            past_time_features (`torch.Tensor`):
                A tensor of shape `(batch_size, past_length, num_features)` containing the past time features.
            static_categorical_features (`Optional[torch.Tensor]`):
                An optional tensor of shape `(batch_size, num_categorical_features)` containing the static categorical
                features.
            static_real_features (`Optional[torch.Tensor]`):
                An optional tensor of shape `(batch_size, num_real_features)` containing the static real features.
            past_observed_mask (`Optional[torch.Tensor]`):
                An optional tensor of shape `(batch_size, past_length, input_size)` containing the mask of observed
                values in the past.
            future_values (`Optional[torch.Tensor]`):
                An optional tensor of shape `(batch_size, future_length, input_size)` containing the future values.

        Returns:
            A tuple containing the following tensors:
            - reshaped_lagged_sequence (`torch.Tensor`): A tensor of shape `(batch_size, sequence_length, num_lags *
              input_size)` containing the lagged subsequences of the inputs.
            - features (`torch.Tensor`): A tensor of shape `(batch_size, sequence_length, num_features)` containing the
              concatenated static and time features.
            - loc (`torch.Tensor`): A tensor of shape `(batch_size, input_size)` containing the mean of the input
              values.
            - scale (`torch.Tensor`): A tensor of shape `(batch_size, input_size)` containing the std of the input
              values.
            - static_feat (`torch.Tensor`): A tensor of shape `(batch_size, num_static_features)` containing the
              concatenated static features.
        """
        # time feature
        time_feat = (
            torch.cat(
                (
                    past_time_features[:, self._past_length - self.config.context_length :, ...],
                    future_time_features,
                ),
                dim=1,
            )
            if future_values is not None
            else past_time_features[:, self._past_length - self.config.context_length :, ...]
        )

        # target
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        context = past_values[:, -self.config.context_length :]
        observed_context = past_observed_mask[:, -self.config.context_length :]
        _, loc, scale = self.scaler(context, observed_context)

        inputs = (
            (torch.cat((past_values, future_values), dim=1) - loc) / scale
            if future_values is not None
            else (past_values - loc) / scale
        )

        # static features
        log_abs_loc = loc.abs().log1p() if self.config.input_size == 1 else loc.squeeze(1).abs().log1p()
        log_scale = scale.log() if self.config.input_size == 1 else scale.squeeze(1).log()
        static_feat = torch.cat((log_abs_loc, log_scale), dim=1)

        if static_real_features is not None:
            static_feat = torch.cat((static_real_features, static_feat), dim=1)
        if static_categorical_features is not None:
            embedded_cat = self.embedder(static_categorical_features)
            static_feat = torch.cat((embedded_cat, static_feat), dim=1)
        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)

        # all features
        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        # lagged features
        subsequences_length = (
            self.config.context_length + self.config.prediction_length
            if future_values is not None
            else self.config.context_length
        )
        lagged_sequence = self.get_lagged_subsequences(sequence=inputs, subsequences_length=subsequences_length)
        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

        if reshaped_lagged_sequence.shape[1] != time_feat.shape[1]:
            raise ValueError(
                f"input length {reshaped_lagged_sequence.shape[1]} and time feature lengths {time_feat.shape[1]} does not match"
            )
        return reshaped_lagged_sequence, features, loc, scale, static_feat

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(AUTOFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AutoformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[AutoformerModelOutput, Tuple]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import AutoformerModel

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> model = AutoformerModel.from_pretrained("huggingface/autoformer-tourism-monthly")

        >>> # during training, one provides both past and future values
        >>> # as well as possible additional features
        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     past_time_features=batch["past_time_features"],
        ...     past_observed_mask=batch["past_observed_mask"],
        ...     static_categorical_features=batch["static_categorical_features"],
        ...     future_values=batch["future_values"],
        ...     future_time_features=batch["future_time_features"],
        ... )

        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_inputs, temporal_features, loc, scale, static_feat = self.create_network_inputs(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            future_values=future_values,
            future_time_features=future_time_features,
        )

        if encoder_outputs is None:
            enc_input = torch.cat(
                (
                    transformer_inputs[:, : self.config.context_length, ...],
                    temporal_features[:, : self.config.context_length, ...],
                ),
                dim=-1,
            )
            encoder_outputs = self.encoder(
                inputs_embeds=enc_input,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if future_values is not None:
            # Decoder inputs
            # seasonality and trend from context length
            seasonal_input, trend_input = self.decomposition_layer(
                transformer_inputs[:, : self.config.context_length, ...]
            )
            mean = (
                torch.mean(transformer_inputs[:, : self.config.context_length, ...], dim=1)
                .unsqueeze(1)
                .repeat(1, self.config.prediction_length, 1)
            )
            zeros = torch.zeros(
                [transformer_inputs.shape[0], self.config.prediction_length, transformer_inputs.shape[2]],
                device=enc_input.device,
            )

            decoder_input = torch.cat(
                (
                    torch.cat((seasonal_input[:, -self.config.label_length :, ...], zeros), dim=1),
                    temporal_features[:, self.config.context_length - self.config.label_length :, ...],
                ),
                dim=-1,
            )
            trend_init = torch.cat(
                (
                    torch.cat((trend_input[:, -self.config.label_length :, ...], mean), dim=1),
                    temporal_features[:, self.config.context_length - self.config.label_length :, ...],
                ),
                dim=-1,
            )

            decoder_outputs = self.decoder(
                trend=trend_init,
                inputs_embeds=decoder_input,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            decoder_outputs = AutoFormerDecoderOutput()

        if not return_dict:
            return decoder_outputs + encoder_outputs + (loc, scale, static_feat)

        return AutoformerModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            trend=decoder_outputs.trend,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            loc=loc,
            scale=scale,
            static_features=static_feat,
        )


@add_start_docstrings(
    "The Autoformer Model with a distribution head on top for time-series forecasting.",
    AUTOFORMER_START_DOCSTRING,
)
class AutoformerForPrediction(AutoformerPreTrainedModel):
    def __init__(self, config: AutoformerConfig):
        super().__init__(config)
        self.model = AutoformerModel(config)
        if config.distribution_output == "student_t":
            self.distribution_output = StudentTOutput(dim=config.input_size)
        elif config.distribution_output == "normal":
            self.distribution_output = NormalOutput(dim=config.input_size)
        elif config.distribution_output == "negative_binomial":
            self.distribution_output = NegativeBinomialOutput(dim=config.input_size)
        else:
            raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.parameter_projection = self.distribution_output.get_parameter_projection(self.model.config.feature_size)
        self.target_shape = self.distribution_output.event_shape

        if config.loss == "nll":
            self.loss = nll
        else:
            raise ValueError(f"Unknown loss function {config.loss}")

        # Initialize weights of distribution_output and apply final processing
        self.post_init()

    def output_params(self, decoder_output):
        return self.parameter_projection(decoder_output[:, -self.config.prediction_length :, :])

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    @torch.jit.ignore
    def output_distribution(self, params, loc=None, scale=None, trailing_n=None) -> torch.distributions.Distribution:
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distribution_output.distribution(sliced_params, loc=loc, scale=scale)

    @add_start_docstrings_to_model_forward(AUTOFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqTSPredictionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqTSPredictionOutput, Tuple]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import AutoformerForPrediction

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly")

        >>> # during training, one provides both past and future values
        >>> # as well as possible additional features
        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     past_time_features=batch["past_time_features"],
        ...     past_observed_mask=batch["past_observed_mask"],
        ...     static_categorical_features=batch["static_categorical_features"],
        ...     future_values=batch["future_values"],
        ...     future_time_features=batch["future_time_features"],
        ... )

        >>> loss = outputs.loss
        >>> loss.backward()

        >>> # during inference, one only provides past values
        >>> # as well as possible additional features
        >>> # the model autoregressively generates future values
        >>> outputs = model.generate(
        ...     past_values=batch["past_values"],
        ...     past_time_features=batch["past_time_features"],
        ...     past_observed_mask=batch["past_observed_mask"],
        ...     static_categorical_features=batch["static_categorical_features"],
        ...     future_time_features=batch["future_time_features"],
        ... )

        >>> mean_prediction = outputs.sequences.mean(dim=1)
        ```

        <Tip>

        The AutoformerForPrediction can also use static_real_features. To do so, set num_static_real_features in
        AutoformerConfig based on number of such features in the dataset (in case of tourism_monthly dataset it
        is equal to 1), initialize the model and call as shown below:

        ```
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import AutoformerConfig, AutoformerForPrediction

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> # check number of static real features
        >>> num_static_real_features = batch["static_real_features"].shape[-1]

        >>> # load configuration of pretrained model and override num_static_real_features
        >>> configuration = AutoformerConfig.from_pretrained(
        ...     "huggingface/autoformer-tourism-monthly",
        ...     num_static_real_features=num_static_real_features,
        ... )
        >>> # we also need to update feature_size as it is not recalculated
        >>> configuration.feature_size += num_static_real_features

        >>> model = AutoformerForPrediction(configuration)

        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     past_time_features=batch["past_time_features"],
        ...     past_observed_mask=batch["past_observed_mask"],
        ...     static_categorical_features=batch["static_categorical_features"],
        ...     static_real_features=batch["static_real_features"],
        ...     future_values=batch["future_values"],
        ...     future_time_features=batch["future_time_features"],
        ... )
        ```

        </Tip>
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if future_values is not None:
            use_cache = False

        outputs = self.model(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            future_values=future_values,
            future_time_features=future_time_features,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        prediction_loss = None
        params = None
        if future_values is not None:
            # outputs.last_hidden_state and trend
            # loc is 4rd last and scale is 3rd last output
            params = self.output_params(outputs[0] + outputs[1])
            distribution = self.output_distribution(params, loc=outputs[-3], scale=outputs[-2])

            loss = self.loss(distribution, future_values)

            if future_observed_mask is None:
                future_observed_mask = torch.ones_like(future_values)

            if len(self.target_shape) == 0:
                loss_weights = future_observed_mask
            else:
                loss_weights, _ = future_observed_mask.min(dim=-1, keepdim=False)

            prediction_loss = weighted_average(loss, weights=loss_weights)

        if not return_dict:
            outputs = ((params,) + outputs[2:]) if params is not None else outputs[2:]
            return ((prediction_loss,) + outputs) if prediction_loss is not None else outputs

        return Seq2SeqTSPredictionOutput(
            loss=prediction_loss,
            params=params,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            loc=outputs.loc,
            scale=outputs.scale,
            static_features=outputs.static_features,
        )

    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SampleTSPredictionOutput:
        r"""
        Greedily generate sequences of sample predictions from a model with a probability distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`):
                Past values of the time series, that serve as context in order to predict the future. The sequence size
                of this tensor must be larger than the `context_length` of the model, since the model will use the
                larger size to construct lag features, i.e. additional values from the past which are added in order to
                serve as "extra context".

                The `sequence_length` here is equal to `config.context_length` + `max(config.lags_sequence)`, which if
                no `lags_sequence` is configured, is equal to `config.context_length` + 7 (as by default, the largest
                look-back index in `config.lags_sequence` is 7). The property `_past_length` returns the actual length
                of the past.

                The `past_values` is what the Transformer encoder gets as input (with optional additional features,
                such as `static_categorical_features`, `static_real_features`, `past_time_features` and lags).

                Optionally, missing values need to be replaced with zeros and indicated via the `past_observed_mask`.

                For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number
                of variates in the time series per time step.
            past_time_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`):
                Required time features, which the model internally will add to `past_values`. These could be things
                like "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features).
                These could also be so-called "age" features, which basically help the model know "at which point in
                life" a time-series is. Age features have small values for distant past time steps and increase
                monotonically the more we approach the current time step. Holiday features are also a good example of
                time features.

                These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT,
                where the position encodings are learned from scratch internally as parameters of the model, the Time
                Series Transformer requires to provide additional time features. The Time Series Transformer only
                learns additional embeddings for `static_categorical_features`.

                Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these
                features must but known at prediction time.

                The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
            future_time_features (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`):
                Required time features for the prediction window, which the model internally will add to sampled
                predictions. These could be things like "month of year", "day of the month", etc. encoded as vectors
                (for instance as Fourier features). These could also be so-called "age" features, which basically help
                the model know "at which point in life" a time-series is. Age features have small values for distant
                past time steps and increase monotonically the more we approach the current time step. Holiday features
                are also a good example of time features.

                These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT,
                where the position encodings are learned from scratch internally as parameters of the model, the Time
                Series Transformer requires to provide additional time features. The Time Series Transformer only
                learns additional embeddings for `static_categorical_features`.

                Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these
                features must but known at prediction time.

                The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

            static_categorical_features (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*):
                Optional static categorical features for which the model will learn an embedding, which it will add to
                the values of the time series.

                Static categorical features are features which have the same value for all time steps (static over
                time).

                A typical example of a static categorical feature is a time series ID.
            static_real_features (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*):
                Optional static real features which the model will add to the values of the time series.

                Static real features are features which have the same value for all time steps (static over time).

                A typical example of a static real feature is promotion information.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

        Return:
            [`SampleTSPredictionOutput`] where the outputs `sequences` tensor will have shape `(batch_size, number of
            samples, prediction_length)` or `(batch_size, number of samples, prediction_length, input_size)` for
            multivariate predictions.
        """
        outputs = self(
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            past_time_features=past_time_features,
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_time_features=None,
            future_values=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=False,
        )

        decoder = self.model.get_decoder()
        enc_last_hidden = outputs.encoder_last_hidden_state
        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features

        num_parallel_samples = self.config.num_parallel_samples
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_values = (
            past_values.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc
        ) / repeated_scale

        time_features = torch.cat((past_time_features, future_time_features), dim=1)

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_features.shape[1], -1)
        features = torch.cat((expanded_static_feat, time_features), dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_enc_last_hidden = enc_last_hidden.repeat_interleave(repeats=num_parallel_samples, dim=0)

        lagged_sequence = self.model.get_lagged_subsequences(
            sequence=repeated_past_values, subsequences_length=self.config.context_length
        )
        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)
        seasonal_input, trend_input = self.model.decomposition_layer(reshaped_lagged_sequence)

        mean = torch.mean(reshaped_lagged_sequence, dim=1).unsqueeze(1).repeat(1, self.config.prediction_length, 1)
        zeros = torch.zeros(
            [reshaped_lagged_sequence.shape[0], self.config.prediction_length, reshaped_lagged_sequence.shape[2]],
            device=reshaped_lagged_sequence.device,
        )

        decoder_input = torch.cat(
            (
                torch.cat((seasonal_input[:, -self.config.label_length :, ...], zeros), dim=1),
                repeated_features[:, -self.config.prediction_length - self.config.label_length :, ...],
            ),
            dim=-1,
        )
        trend_init = torch.cat(
            (
                torch.cat((trend_input[:, -self.config.label_length :, ...], mean), dim=1),
                repeated_features[:, -self.config.prediction_length - self.config.label_length :, ...],
            ),
            dim=-1,
        )
        decoder_outputs = decoder(
            trend=trend_init, inputs_embeds=decoder_input, encoder_hidden_states=repeated_enc_last_hidden
        )
        decoder_last_hidden = decoder_outputs.last_hidden_state
        trend = decoder_outputs.trend
        params = self.output_params(decoder_last_hidden + trend)
        distr = self.output_distribution(params, loc=repeated_loc, scale=repeated_scale)
        future_samples = distr.sample()
        scaling_params = {"loc": loc, "scale": scale}

        return SampleTSPredictionOutput(
            sequences=future_samples.reshape(
                (-1, num_parallel_samples, self.config.prediction_length) + self.target_shape,
            ),
            params=dict(zip(self.distribution_output.args_dim, params)),
            distribution=self.config.distribution_output,
            scaling_params=scaling_params,
        )

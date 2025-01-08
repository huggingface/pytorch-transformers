# coding=utf-8
# Copyright 2023 IBM and HuggingFace Inc. team. All Rights Reserved.
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
"""PyTorch PatchTSMixer model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_patchtsmixer import PatchTSMixerConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PatchTSMixerConfig"


PATCHTSMIXER_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSMixerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        mask_input (`bool`, *optional*, defaults to `False`):
            If True, Masking will be enabled. False otherwise.
"""

PATCHTSMIXER_INPUTS_DOCSTRING = r"""
    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a pretraining task, this denotes the input time series to predict
            the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
            for classification or regression tasks, it denotes the appropriate context values of the time series.

            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
            greater than 1.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class PatchTSMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTBatchNorm with PatchTST->PatchTSMixer
class PatchTSMixerBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        output = inputs.transpose(1, 2)  # output: (batch_size, d_model, sequence_length)
        output = self.batchnorm(output)
        return output.transpose(1, 2)


class PatchTSMixerPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        # positional encoding: [num_patches x d_model]
        if config.use_positional_encoding:
            self.position_enc = self._init_pe(config)
        else:
            self.position_enc = nn.Parameter(torch.zeros(config.num_patches, config.d_model))

    @staticmethod
    def _init_pe(config: PatchTSMixerConfig) -> nn.Parameter:
        # Positional encoding
        if config.positional_encoding_type == "random":
            position_enc = nn.Parameter(torch.randn(config.num_patches, config.d_model), requires_grad=True)
        elif config.positional_encoding_type == "sincos":
            position_enc = torch.zeros(config.num_patches, config.d_model)
            position = torch.arange(0, config.num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            raise ValueError(
                f"{config.positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'."
            )
        return position_enc

    def forward(self, patch_input: torch.Tensor):
        # hidden_state: [bs x num_channels x num_patches x d_model]
        hidden_state = patch_input + self.position_enc
        return hidden_state


class PatchTSMixerNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.norm_mlp = config.norm_mlp

        if "batch" in config.norm_mlp.lower():
            self.norm = PatchTSMixerBatchNorm(config)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """
        if "batch" in self.norm_mlp.lower():
            # reshape the data
            inputs_reshaped = torch.reshape(
                inputs,
                (
                    inputs.shape[0] * inputs.shape[1],
                    inputs.shape[2],
                    inputs.shape[3],
                ),
            )  # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]

            # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]
            inputs_reshaped = self.norm(inputs_reshaped)

            # put back data to the original shape
            inputs = torch.reshape(inputs_reshaped, inputs.shape)

        else:
            inputs = self.norm(inputs)

        return inputs


class PatchTSMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        num_hidden = in_features * config.expansion_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the MLP layer.
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class PatchTSMixerChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.norm = PatchTSMixerNormLayer(config)
        self.gated_attn = config.gated_attn
        self.mlp = PatchTSMixerMLP(
            in_features=config.num_input_channels,
            out_features=config.num_input_channels,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(
                in_size=config.num_input_channels, out_size=config.num_input_channels
            )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                input to the MLP layer
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        residual = inputs
        inputs = self.norm(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        if self.gated_attn:
            inputs = self.gating_block(inputs)

        inputs = self.mlp(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        out = inputs + residual
        return out


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->PatchTSMixer
class PatchTSMixerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[PatchTSMixerConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

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
        query_states = self.q_proj(hidden_states) * self.scaling
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
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

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


class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.norm = PatchTSMixerNormLayer(config)

        self.self_attn = config.self_attn
        self.gated_attn = config.gated_attn

        self.mlp = PatchTSMixerMLP(
            in_features=config.num_patches,
            out_features=config.num_patches,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(in_size=config.num_patches, out_size=config.num_patches)

        if config.self_attn:
            self.self_attn_layer = PatchTSMixerAttention(
                embed_dim=config.d_model,
                num_heads=config.self_attn_heads,
                dropout=config.dropout,
            )
            self.norm_attn = PatchTSMixerNormLayer(config)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state (`torch.Tensor`): Input tensor.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden_state

        hidden_state = self.norm(hidden_state)

        if self.self_attn:
            batch_size, n_vars, num_patches, d_model = hidden_state.shape
            hidden_state_reshaped = hidden_state.reshape(batch_size * n_vars, num_patches, d_model)

            x_attn, _, _ = self.self_attn_layer(hidden_state_reshaped, output_attentions=False)
            x_attn = x_attn.reshape(batch_size, n_vars, num_patches, d_model)

        # Transpose so that num_patches is the last dimension
        hidden_state = hidden_state.transpose(2, 3)
        hidden_state = self.mlp(hidden_state)

        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)

        # Transpose back
        hidden_state = hidden_state.transpose(2, 3)

        if self.self_attn:
            hidden_state = self.norm_attn(hidden_state + x_attn)

        out = hidden_state + residual
        return out


class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.

    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.norm = PatchTSMixerNormLayer(config)

        self.gated_attn = config.gated_attn

        self.mlp = PatchTSMixerMLP(
            in_features=config.d_model,
            out_features=config.d_model,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(in_size=config.d_model, out_size=config.d_model)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)

        if self.gated_attn:
            hidden = self.gating_block(hidden)

        out = hidden + residual
        return out


class PatchTSMixerLayer(nn.Module):
    """
    The `PatchTSMixer` layer that does all three kinds of mixing.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.

    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.patch_mixer = PatchMixerBlock(config=config)
        self.feature_mixer = FeatureMixerBlock(config=config)

        self.mode = config.mode

        if config.mode == "mix_channel":
            self.channel_feature_mixer = PatchTSMixerChannelFeatureMixerBlock(config=config)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        if self.mode == "mix_channel":
            hidden = self.channel_feature_mixer(hidden)

        hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)  # hidden: (batch_size x num_patches x d_model)
        return hidden


class PatchTSMixerBlock(nn.Module):
    """The main computing framework of the `PatchTSMixer` model.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        num_layers = config.num_layers

        self.mixers = nn.ModuleList([PatchTSMixerLayer(config=config) for _ in range(num_layers)])

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        all_hidden_states = []

        embedding = hidden_state

        for mod in self.mixers:
            embedding = mod(embedding)
            if output_hidden_states:
                all_hidden_states.append(embedding)

        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None


class PatchTSMixerForPredictionHead(nn.Module):
    """Prediction Head for Forecasting

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig, distribution_output=None):
        super().__init__()

        self.prediction_channel_indices = config.prediction_channel_indices

        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices.sort()

        self.dropout_layer = nn.Dropout(config.head_dropout)
        if distribution_output is None:
            self.base_forecast_block = nn.Linear((config.num_patches * config.d_model), config.prediction_length)
        else:
            self.base_forecast_block = distribution_output.get_parameter_projection(
                config.num_patches * config.d_model
            )

        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, hidden_features):
        """

        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size, num_patch, d_model)` in `flatten` mode
                or `(batch_size, n_vars, num_patch, d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size, prediction_length, nvars)`.

        """

        hidden_features = self.flatten(hidden_features)  # [batch_size x n_vars x num_patch * d_model]
        hidden_features = self.dropout_layer(hidden_features)  # [batch_size x n_vars x num_patch * d_model]
        forecast = self.base_forecast_block(hidden_features)  # [batch_size x n_vars x prediction_length]
        if isinstance(forecast, tuple):
            forecast = tuple(z.transpose(-1, -2) for z in forecast)
        else:
            forecast = forecast.transpose(-1, -2)  # [batch_size x prediction_length x n_vars]

        if self.prediction_channel_indices is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(z[..., self.prediction_channel_indices] for z in forecast)
            else:
                forecast = forecast[..., self.prediction_channel_indices]  # [batch_size x prediction_length x n_vars]

        return forecast


class PatchTSMixerLinearHead(nn.Module):
    """Linear head for Classification and Regression.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig, distribution_output=None):
        super().__init__()

        self.head_aggregation = config.head_aggregation
        self.output_range = config.output_range

        if config.head_aggregation is None:
            mul_factor = config.num_patches
        else:
            mul_factor = 1
        self.distribution_output = distribution_output
        if distribution_output is None:
            self.projection = nn.Linear(
                config.d_model * config.num_input_channels * mul_factor,
                config.num_targets,
            )
        else:
            self.projection = distribution_output.get_parameter_projection(
                config.d_model * config.num_input_channels * mul_factor
            )

        if config.head_aggregation is None:
            self.flatten = nn.Flatten(start_dim=-3)
        else:
            self.flatten = nn.Flatten(start_dim=-2)

        self.dropout = nn.Dropout(config.head_dropout)

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x d_model)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x num_targets)`.
        """

        # batch_size x d_model x num_patch or batch_size x n_vars x d_model x num_patch
        hidden_features = hidden_features.transpose(-1, -2)
        if self.head_aggregation == "use_last":
            # batch_size x d_model (flatten) or # batch_size x n_vars x d_model (common_channel)
            hidden_features = hidden_features[..., -1]
        elif self.head_aggregation == "max_pool":
            # batch_size x n_vars x d_model or batch_size x d_model
            hidden_features = hidden_features.max(dim=-1).values
        elif self.head_aggregation == "avg_pool":
            # batch_size x n_vars x d_model or batch_size x d_model
            hidden_features = hidden_features.mean(dim=-1)

        if self.flatten:
            hidden_features = self.flatten(hidden_features)
        hidden_features = self.dropout(hidden_features)
        hidden_features = self.projection(hidden_features)  # batch_size x num_targets

        if (self.distribution_output is None) and (self.output_range is not None):
            hidden_features = (
                torch.sigmoid(hidden_features) * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
            )
        return hidden_features


class PatchTSMixerPreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = PatchTSMixerConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, PatchTSMixerPositionalEncoding):
            # initialize positional encoding
            if self.config.positional_encoding_type == "random":
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PatchTSMixerBatchNorm):
            module.batchnorm.bias.data.zero_()
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()


class PatchTSMixerPretrainHead(nn.Module):
    """Pretraining head.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.dropout_layer = nn.Dropout(config.head_dropout)
        self.base_pt_block = nn.Linear(config.d_model, config.patch_length)

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x d_model)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x n_vars x num_patch x patch_length)`.
        """

        hidden_features = self.dropout_layer(hidden_features)
        forecast = self.base_pt_block(hidden_features)  # [batch_size x n_vars x num_patch x patch_length]
        return forecast


# Copied from transformers.models.patchtst.modeling_patchtst.random_masking
def random_masking(
    inputs: torch.Tensor,
    mask_ratio: float,
    unmasked_channel_indices: list = None,
    channel_consistent_masking: bool = False,
    mask_value: int = 0,
):
    """random_masking: Mask the input considering the control variables.

    Args:
        inputs (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, num_features)`):
            The input tensor to mask.
        mask_ratio (`float`):
            Masking ratio applied to mask the input data during random pretraining. It is the number between 0 and 1.
        unmasked_channel_indices (list, *optional*):
            Indices of channels that will not be masked.
        channel_consistent_masking (bool, *optional*, defaults to `False`):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels.
        mask_value (int, *optional*, defaults to 0):
            Define the value of masked patches for pretraining.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as input Tensor and mask tensor of shape [bs x c x
        n]
    """
    if mask_ratio < 0 or mask_ratio >= 1:
        raise ValueError(f"Mask ratio {mask_ratio} has to be between 0 and 1.")

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    device = inputs.device

    len_keep = int(sequence_length * (1 - mask_ratio))

    if channel_consistent_masking:
        noise = torch.rand(batch_size, 1, sequence_length, device=device)  # noise in [0, 1], bs x 1 x  L
        noise = noise.repeat(1, num_channels, 1)  # bs x num_channels x time
    else:
        # noise in [0, 1], bs x num_channels x L
        noise = torch.rand(batch_size, num_channels, sequence_length, device=device)

    # mask: [bs x num_channels x num_patch]
    mask = torch.ones(batch_size, num_channels, sequence_length, device=device)
    mask[:, :, :len_keep] = 0

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x num_channels x L]

    mask = torch.gather(mask, dim=-1, index=ids_restore)
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patches x patch_length]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]


# Copied from transformers.models.patchtst.modeling_patchtst.forecast_masking
def forecast_masking(
    inputs: torch.Tensor,
    num_forecast_mask_patches: Union[list, int],
    unmasked_channel_indices: list = None,
    mask_value: int = 0,
):
    """Forecast masking that masks the last K patches where K is from the num_forecast_mask_patches.
    If num_forecast_mask_patches is a list, samples in the batch will be randomly masked by numbers defined in the list.

    Parameters:
        inputs (`torch.Tensor`):
            Input of shape `(bs, num_channels, num_patch, patch_length)`
        num_forecast_mask_patches (`list`):
            Number of patches to be masked at the end of each batch sample. e.g. 4 or [3, 5].
        unmasked_channel_indices (`list`, *optional*):
            Indices of channels that are not masked.
        mask_value (`int`, *optional*, defaults to 0):
            Values in the masked patches will be filled by `mask_value`.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as inputs Tensor and Mask tensor of shape `(bs,
        num_channels , num_patch)` or `(bs, tsg1, tsg2, num_channels, num_patch)`
    """

    if isinstance(num_forecast_mask_patches, int):
        num_forecast_mask_patches = [num_forecast_mask_patches]
    forecast_mask_ratios = [1 for _ in num_forecast_mask_patches]

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    mask = torch.zeros(batch_size, num_channels, sequence_length, device=inputs.device)

    t_list = []
    total_length = 0
    total_ratio = sum(forecast_mask_ratios)

    for patch_length, ratio in zip(num_forecast_mask_patches, forecast_mask_ratios):
        if patch_length <= 0 or patch_length >= sequence_length:
            raise ValueError(
                f"num_forecast_mask_patches {patch_length} should be greater than 0 and less than total patches."
            )
        temp_len = int(batch_size * ratio / total_ratio)
        t_list.append([patch_length, ratio, temp_len])
        total_length += temp_len

    t_list = sorted(t_list, key=lambda x: x[2])

    if total_length < batch_size:
        t_list[0][2] = t_list[0][2] + (batch_size - total_length)
    elif total_length > batch_size:
        t_list[-1][2] = t_list[-1][2] + (total_length - batch_size)

    batch1 = 0
    for patch_len, _, temp_len in t_list:
        batch2 = batch1 + temp_len
        mask[batch1:batch2, :, -patch_len:] = 1
        batch1 = batch2

    perm = torch.randperm(mask.shape[0])
    mask = mask[perm]

    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patch x patch_len]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTPatchify with PatchTST->PatchTSMixer
class PatchTSMixerPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.sequence_length = config.context_length
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # get the number of patches
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        sequence_length = past_values.shape[-2]
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )
        # output: [bs x new_sequence_length x num_channels]
        output = past_values[:, self.sequence_start :, :]
        # output: [bs x num_patches x num_input_channels x patch_length]
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # output: [bs x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()
        return output


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTMasking with PatchTST->PatchTSMixer
class PatchTSMixerMasking(nn.Module):
    """
    Class to perform random or forecast masking.

    Parameters:
        config (`PatchTSMixerConfig`): model config
    Returns:
        x_mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
            Masked patched input
        mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
            Bool tensor indicating True on masked points
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.random_mask_ratio = config.random_mask_ratio
        self.channel_consistent_masking = config.channel_consistent_masking
        self.mask_type = config.mask_type
        self.num_forecast_mask_patches = config.num_forecast_mask_patches
        self.unmasked_channel_indices = config.unmasked_channel_indices
        self.mask_value = config.mask_value
        if self.unmasked_channel_indices is not None:
            self.unmasked_channel_indices = sorted(self.unmasked_channel_indices)

    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input

        Return:
            masked_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
                Masked patched input
            mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
                Bool tensor indicating True on masked points

        """
        if self.mask_type == "random":
            masked_input, mask = random_masking(
                inputs=patch_input,
                mask_ratio=self.random_mask_ratio,
                unmasked_channel_indices=self.unmasked_channel_indices,
                channel_consistent_masking=self.channel_consistent_masking,
                mask_value=self.mask_value,
            )
        elif self.mask_type == "forecast":
            masked_input, mask = forecast_masking(
                inputs=patch_input,
                num_forecast_mask_patches=self.num_forecast_mask_patches,
                unmasked_channel_indices=self.unmasked_channel_indices,
                mask_value=self.mask_value,
            )
        else:
            raise ValueError(f"Invalid mask type {self.mask_type}.")

        # mask: [bs x num_input_channels x num_patch]
        mask = mask.bool()
        return masked_input, mask


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTStdScaler with PatchTST->PatchTSMixer
class PatchTSMixerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: PatchTSMixerConfig):
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


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTMeanScaler with PatchTST->PatchTSMixer
class PatchTSMixerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: PatchTSMixerConfig):
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


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTNOPScaler with PatchTST->PatchTSMixer
class PatchTSMixerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, config: PatchTSMixerConfig):
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


@dataclass
class PatchTSMixerEncoderOutput(ModelOutput):
    """
    Base class for `PatchTSMixerEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSMixerEncoder(PatchTSMixerPreTrainedModel):
    """
    Encoder for PatchTSMixer which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.use_return_dict = config.use_return_dict

        self.patcher = nn.Linear(config.patch_length, config.d_model)
        if config.use_positional_encoding:
            self.positional_encoder = PatchTSMixerPositionalEncoding(config=config)
        else:
            self.positional_encoder = None
        self.mlp_mixer_encoder = PatchTSMixerBlock(config=config)

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @replace_return_docstrings(output_type=PatchTSMixerEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSMixerEncoderOutput]:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series. For a pretraining task, this denotes the input time series to
                predict the masked portion. For a forecasting task, this denotes the history/past time series values.
                Similarly, for classification or regression tasks, it denotes the appropriate context values of the
                time series.

                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series,
                it is greater than 1.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, n_vars, num_patches, d_model)`
        """

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # flatten [bs x num_patch x d_model]. common_channel/mix_channel: [bs x n_vars x num_patch x d_model]
        patches = self.patcher(past_values)

        # add positional encoder
        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)

        last_hidden_state, hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states=output_hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    hidden_states,
                ]
            )

        return PatchTSMixerEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states)


@dataclass
class PatchTSMixerModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        patch_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Patched input data to the model.
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`,*optional*):
            Bool Tensor indicating True in masked patches and False otherwise.
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the mean of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the std dev of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    patch_input: torch.FloatTensor = None
    mask: Optional[torch.FloatTensor] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    "The PatchTSMixer Model for time-series forecasting.",
    PATCHTSMIXER_START_DOCSTRING,
)
class PatchTSMixerModel(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig, mask_input: bool = False):
        super().__init__(config)

        self.use_return_dict = config.use_return_dict
        self.encoder = PatchTSMixerEncoder(config)
        self.patching = PatchTSMixerPatchify(config)

        if mask_input is True:
            self.masking = PatchTSMixerMasking(config)
        else:
            self.masking = None

        if config.scaling == "mean":
            self.scaler = PatchTSMixerMeanScaler(config)
        elif config.scaling == "std" or config.scaling is True:
            self.scaler = PatchTSMixerStdScaler(config)
        else:
            self.scaler = PatchTSMixerNOPScaler(config)

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> PatchTSMixerModelOutput:
        r"""
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        mask = None
        if observed_mask is None:
            observed_mask = torch.ones_like(past_values)
        scaled_past_values, loc, scale = self.scaler(past_values, observed_mask)

        patched_x = self.patching(scaled_past_values)  # [batch_size x num_input_channels x num_patch x patch_length

        enc_input = patched_x
        if self.masking is not None:
            enc_input, mask = self.masking(patched_x)
            # enc_input: [batch_size x num_input_channels x num_patch x patch_length]
            # mask: [batch_size x num_input_channels x num_patch]

        encoder_output = self.encoder(
            enc_input,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(encoder_output, tuple):
            encoder_output = PatchTSMixerEncoderOutput(*encoder_output)

        if not return_dict:
            return tuple(
                v
                for v in [
                    encoder_output.last_hidden_state,
                    encoder_output.hidden_states,
                    patched_x,
                    mask,
                    loc,
                    scale,
                ]
            )

        return PatchTSMixerModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            patch_input=patched_x,
            mask=mask,
            loc=loc,
            scale=scale,
        )


@dataclass
class PatchTSMixerForPreTrainingOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForPreTrainingOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, patch_length)`):
            Prediction output from the pretrain head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSMixerForPretraining(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for mask pretraining.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.model = PatchTSMixerModel(config, mask_input=True)
        self.head = PatchTSMixerPretrainHead(config=config)
        self.masked_loss = config.masked_loss
        self.use_return_dict = config.use_return_dict

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> PatchTSMixerForPreTrainingOutput:
        r"""
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if self.masked_loss is True:
            loss = torch.nn.MSELoss(reduction="none")
        else:
            loss = torch.nn.MSELoss(reduction="mean")

        # past_values: tensor [batch_size x context_length x num_input_channels]
        model_output = self.model(
            past_values,
            observed_mask=observed_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # x.last_hidden_state: [batch_size x nvars x num_patch x d_model]
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)

        x_hat = self.head(model_output.last_hidden_state)  # tensor [batch_size x nvars x num_patch x patch_length]

        if return_loss is True:
            loss_val = loss(x_hat, model_output.patch_input)
        else:
            loss_val = None

        # calculate masked_loss
        if self.masked_loss is True and loss_val is not None:
            loss_val = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    x_hat,
                    model_output.last_hidden_state,
                    model_output.hidden_states,
                ]
            )

        return PatchTSMixerForPreTrainingOutput(
            loss=loss_val,
            prediction_outputs=x_hat,  # tensor [batch_size x nvars x num_patch x patch_length]
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x d_model]
            hidden_states=model_output.hidden_states,
        )


@dataclass
class PatchTSMixerForPredictionOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForPredictionOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_input_channels)`):
            Prediction output from the forecast head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
        loc (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input mean
        scale (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input std dev

    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


@dataclass
class SamplePatchTSMixerPredictionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length, number_channels)`):
            Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None


@dataclass
class SamplePatchTSMixerRegressionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, num_targets)`
                Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.nll
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


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


class PatchTSMixerForPrediction(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for forecasting application.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.loss = config.loss
        self.use_return_dict = config.use_return_dict
        self.prediction_channel_indices = config.prediction_channel_indices
        self.num_parallel_samples = config.num_parallel_samples

        if config.loss == "mse":
            self.distribution_output = None
        else:
            dim = config.prediction_length
            distribution_output_map = {
                "student_t": StudentTOutput,
                "normal": NormalOutput,
                "negative_binomial": NegativeBinomialOutput,
            }
            output_class = distribution_output_map.get(config.distribution_output, None)
            if output_class is not None:
                self.distribution_output = output_class(dim=dim)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerForPredictionHead(
            config=config,
            distribution_output=self.distribution_output,
        )

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForPredictionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> PatchTSMixerForPredictionOutput:
        r"""
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
        future_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,:
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """
        if self.loss == "mse":
            loss = nn.MSELoss(reduction="mean")
        elif self.loss == "nll":
            loss = nll
        else:
            raise ValueError("Invalid loss function: Allowed values: mse and nll")

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # past_values: tensor [batch_size x context_length x num_input_channels]
        model_output = self.model(
            past_values,
            observed_mask=observed_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # model_output: [batch_size x nvars x num_patch x d_model]
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)

        # tensor [batch_size x prediction_length x num_input_channels]
        y_hat = self.head(model_output.last_hidden_state)

        loss_val = None
        if self.prediction_channel_indices is not None:
            if self.distribution_output:
                distribution = self.distribution_output.distribution(
                    y_hat,
                    loc=model_output.loc[..., self.prediction_channel_indices],
                    scale=model_output.scale[..., self.prediction_channel_indices],
                )
                if future_values is not None and return_loss is True:
                    loss_val = loss(
                        distribution,
                        future_values[..., self.prediction_channel_indices],
                    )
                    # take average of the loss
                    loss_val = weighted_average(loss_val)
            else:
                y_hat = (
                    y_hat * model_output.scale[..., self.prediction_channel_indices]
                    + model_output.loc[..., self.prediction_channel_indices]
                )
                if future_values is not None and return_loss is True:
                    loss_val = loss(y_hat, future_values[..., self.prediction_channel_indices])
        else:
            if self.distribution_output:
                distribution = self.distribution_output.distribution(
                    y_hat, loc=model_output.loc, scale=model_output.scale
                )
                if future_values is not None and return_loss is True:
                    loss_val = loss(distribution, future_values)
                    loss_val = weighted_average(loss_val)
            else:
                y_hat = y_hat * model_output.scale + model_output.loc
                if future_values is not None and return_loss is True:
                    loss_val = loss(y_hat, future_values)

        if self.prediction_channel_indices is not None:
            loc = model_output.loc[..., self.prediction_channel_indices]
            scale = model_output.scale[..., self.prediction_channel_indices]
        else:
            loc = model_output.loc
            scale = model_output.scale

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    y_hat,
                    model_output.last_hidden_state,
                    model_output.hidden_states,
                    loc,
                    scale,
                ]
            )

        return PatchTSMixerForPredictionOutput(
            loss=loss_val,
            prediction_outputs=y_hat,  # tensor [batch_size x prediction_length x num_input_channels]
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x d_model]
            hidden_states=model_output.hidden_states,
            loc=loc,
            scale=scale,
        )

    def generate(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> SamplePatchTSMixerPredictionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

            observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSMixerPredictionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, prediction_length, num_input_channels)`.
        """
        # get number of samples
        num_parallel_samples = self.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            future_values=None,
            observed_mask=observed_mask,
            output_hidden_states=False,
        )

        # get distribution

        distribution = self.distribution_output.distribution(
            outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale
        )

        # get samples: list of [batch_size x prediction_length x num_channels]
        samples = [distribution.sample() for _ in range(num_parallel_samples)]

        # stack tensors
        samples = torch.stack(samples, dim=1)  # [batch_size x num_samples x prediction_length x num_channels]
        return SamplePatchTSMixerPredictionOutput(sequences=samples)


@dataclass
class PatchTSMixerForTimeSeriesClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForTimeSeriesClassificationOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Prediction output from the classfication head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSMixerForTimeSeriesClassification(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for classification application.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerLinearHead(
            config=config,
        )
        self.use_return_dict = config.use_return_dict
        if config.scaling in ["std", "mean", True]:
            self.inject_scale = InjectScalerStatistics4D(d_model=config.d_model, num_patches=config.num_patches)
        else:
            self.inject_scale = None

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=PatchTSMixerForTimeSeriesClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> PatchTSMixerForTimeSeriesClassificationOutput:
        r"""
        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `target_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.

            For a classification task, it has a shape of `(batch_size,)`.

            For a regression task, it has a shape of `(batch_size, num_targets)`.
        return_loss (`bool`, *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """

        loss = torch.nn.CrossEntropyLoss()

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        model_output = self.model(
            past_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # x: [batch_size x nvars x num_patch x d_model]
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)

        if self.inject_scale is not None:
            model_output.last_hidden_state = self.inject_scale(
                model_output.last_hidden_state,
                loc=model_output.loc,
                scale=model_output.scale,
            )  # x: [batch_size x nvars x num_patch x d_model]

        y_hat = self.head(model_output.last_hidden_state)  # tensor [batch_size x n_labels]

        if target_values is not None and return_loss is True:
            loss_val = loss(y_hat, target_values)
        else:
            loss_val = None

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    y_hat,
                    model_output.last_hidden_state,
                    model_output.hidden_states,
                ]
            )

        return PatchTSMixerForTimeSeriesClassificationOutput(
            loss=loss_val,
            prediction_outputs=y_hat,  # tensor [batch_size x n_labels]
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x d_model]
            hidden_states=model_output.hidden_states,
        )


@dataclass
class PatchTSMixerForRegressionOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForRegressionOutput`].

    Args:
        regression_outputs (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Prediction output from the regression head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    loss: Optional[torch.FloatTensor] = None
    regression_outputs: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class InjectScalerStatistics4D(nn.Module):
    def __init__(self, d_model: int, num_patches: int, expansion: int = 2):
        super().__init__()

        self.inverse_trans_expansion = nn.Linear(d_model + 2, expansion * d_model)
        self.inverse_trans_compression = nn.Linear(expansion * d_model, d_model)
        self.map_scale_expansion = nn.Linear(2, 2 * expansion)
        self.map_scale_compression = nn.Linear(2 * expansion, 2)
        self.num_patches = num_patches

    def forward(self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, d_model)`)
            loc (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
            scale (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
        Returns:
            `torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, d_model)`
        """

        mean = loc.transpose(-1, -2)  # [batch_size x n_channels x 1 ]
        mean = mean.unsqueeze(-2)  # [batch_size x n_channels x 1 x 1]
        mean = mean.repeat(1, 1, self.num_patches, 1)  # [batch_size x n_channels x num_patch x 1]

        stdev = scale.transpose(-1, -2)  # [batch_size x n_channels x 1 ]
        stdev = stdev.unsqueeze(-2)  # [batch_size x n_channels x 1 x 1]
        stdev = stdev.repeat(1, 1, self.num_patches, 1)  # [batch_size x n_channels x num_patch x 1]

        concat_stats = torch.cat([mean, stdev], dim=-1)  # [batch_size x n_channels x num_patch x 2]

        concat_stats = self.map_scale_expansion(concat_stats)  # [batch_size x n_channels x num_patch x (2*expansion)]
        concat_stats = self.map_scale_compression(concat_stats)  # [batch_size x n_channels x num_patch x 2]

        inputs = torch.cat([inputs, concat_stats], dim=-1)  # [batch_size x channels x num_patch x d_model+2]
        inputs = self.inverse_trans_expansion(inputs)  # [batch_size x channels x num_patch x (expansion*d_model)]
        inputs = self.inverse_trans_compression(inputs)  # [batch_size x channels x num_patch x d_model]

        return inputs


class PatchTSMixerForRegression(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for regression application.

    Args:
        config (`PatchTSMixerConfig`):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)

        self.loss = config.loss
        self.distribution_output = config.distribution_output

        self.use_return_dict = config.use_return_dict
        self.num_parallel_samples = config.num_parallel_samples

        if config.loss == "mse":
            self.distribution_output = None
        else:
            distribution_output_map = {
                "student_t": StudentTOutput,
                "normal": NormalOutput,
                "negative_binomial": NegativeBinomialOutput,
            }
            output_class = distribution_output_map.get(config.distribution_output)
            if output_class is not None:
                self.distribution_output = output_class(dim=config.num_targets)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        if config.scaling in ["std", "mean", True]:
            self.inject_scale = InjectScalerStatistics4D(d_model=config.d_model, num_patches=config.num_patches)
        else:
            self.inject_scale = None

        self.head = PatchTSMixerLinearHead(
            config=config,
            distribution_output=self.distribution_output,
        )

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForRegressionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> PatchTSMixerForRegressionOutput:
        r"""
        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `target_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.

            For a classification task, it has a shape of `(batch_size,)`.

            For a regression task, it has a shape of `(batch_size, num_targets)`.
        return_loss (`bool`, *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """

        if self.loss == "mse":
            loss = nn.MSELoss(reduction="mean")
        elif self.loss == "nll":
            loss = nll
        else:
            raise ValueError("Invalid loss function: Allowed values: mse and nll")

        return_dict = return_dict if return_dict is not None else self.use_return_dict
        model_output = self.model(
            past_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # model_output: [batch_size x nvars x num_patch x d_model]
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)

        if self.inject_scale is not None:
            model_output.last_hidden_state = self.inject_scale(
                model_output.last_hidden_state,
                loc=model_output.loc,
                scale=model_output.scale,
            )  # x: [batch_size x nvars x num_patch x d_model]

        y_hat = self.head(model_output.last_hidden_state)  # [batch_size x num_targets]

        if target_values is not None and return_loss is True:
            if self.distribution_output:
                if self.distribution_output == "negative_binomial" and torch.any(target_values < 0):
                    raise Exception("target_values cannot be negative for negative_binomial distribution.")
                distribution = self.distribution_output.distribution(y_hat)
                # y_hat should be a 2-tuple, each with dimension [bs, num_targets]
                y_hat = tuple([item.view(-1, self.config.num_targets) for item in y_hat])
                loss_val = loss(distribution, target_values)
                # take average of the loss
                loss_val = weighted_average(loss_val)
            else:
                loss_val = loss(y_hat, target_values)
        else:
            loss_val = None

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    y_hat,
                    model_output.last_hidden_state,
                    model_output.hidden_states,
                ]
            )

        return PatchTSMixerForRegressionOutput(
            loss=loss_val,
            regression_outputs=y_hat,  # tensor [batch_size x num_targets]
            last_hidden_state=model_output.last_hidden_state,  # [batch_size x nvars x num_patch x d_model]
            hidden_states=model_output.hidden_states,
        )

    def generate(
        self,
        past_values: torch.Tensor,
    ) -> SamplePatchTSMixerRegressionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the target values.

        Return:
            [`SamplePatchTSMixerRegressionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, num_targets)`.
        """
        # get number of samples
        num_parallel_samples = self.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            target_values=None,
            output_hidden_states=False,
        )

        # get distribution
        distribution = self.distribution_output.distribution(outputs.regression_outputs)

        # get samples
        samples = [
            distribution.sample() for _ in range(num_parallel_samples)
        ]  # samples: list of [batch_size x num_targets]
        # stack tensors
        # [batch_size x num_samples x num_targets]
        samples = torch.stack(samples, dim=1).view(-1, num_parallel_samples, self.config.num_targets)
        return SamplePatchTSMixerRegressionOutput(sequences=samples)


__all__ = [
    "PatchTSMixerPreTrainedModel",
    "PatchTSMixerModel",
    "PatchTSMixerForPretraining",
    "PatchTSMixerForPrediction",
    "PatchTSMixerForTimeSeriesClassification",
    "PatchTSMixerForRegression",
]

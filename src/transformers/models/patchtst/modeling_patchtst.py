# coding=utf-8
# Copyright 2023 IBM & Hugging Face. All rights reserved.
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
"""PyTorch PatchTST model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PatchTSTConfig"


# Copied from transformers.models.hubert.modeling_hubert.HubertAttention with Hubert->PatchTST
class PatchTSTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[PatchTSTConfig] = None,
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


class PatchTSTBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: PatchTSTConfig):
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


class PatchTSTPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: PatchTSTConfig):
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


class PatchTSTMasking(nn.Module):
    """
    Class to perform random or forecast masking.

    Parameters:
        config (`PatchTSTConfig`): model config
    Returns:
        x_mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
            Masked patched input
        mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
            Bool tensor indicating True on masked points
    """

    def __init__(self, config: PatchTSTConfig):
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


class PatchTSTEncoderLayer(nn.Module):
    """
    PatchTST encoder layer
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.channel_attention = config.channel_attention
        # Multi-Head attention
        self.self_attn = PatchTSTAttention(
            embed_dim=config.d_model,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
        )

        # Add & Norm of the sublayer 1
        self.dropout_path1 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
        if config.norm_type == "batchnorm":
            self.norm_sublayer1 = PatchTSTBatchNorm(config)
        elif config.norm_type == "layernorm":
            self.norm_sublayer1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        else:
            raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

        # Add & Norm of the sublayer 2
        if self.channel_attention:
            self.dropout_path2 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
            if config.norm_type == "batchnorm":
                self.norm_sublayer2 = PatchTSTBatchNorm(config)
            elif config.norm_type == "layernorm":
                self.norm_sublayer2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            else:
                raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim, bias=config.bias),
            ACT2CLS[config.activation_function](),
            nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
            nn.Linear(config.ffn_dim, config.d_model, bias=config.bias),
        )

        # Add & Norm of sublayer 3
        self.dropout_path3 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
        if config.norm_type == "batchnorm":
            self.norm_sublayer3 = PatchTSTBatchNorm(config)
        elif config.norm_type == "layernorm":
            self.norm_sublayer3 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        else:
            raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

        self.pre_norm = config.pre_norm

    def forward(self, hidden_state: torch.Tensor, output_attentions: Optional[bool] = None):
        """
        Parameters:
            hidden_state (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, d_model)`, *required*):
                Past values of the time series
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
        Return:
            `torch.Tensor` of shape `(batch_size, num_channels, sequence_length, d_model)`

        """
        batch_size, num_input_channels, sequence_length, d_model = hidden_state.shape

        # First sublayer: attention across time
        # hidden_states: [(bs*num_channels) x sequence_length x d_model]
        hidden_state = hidden_state.view(batch_size * num_input_channels, sequence_length, d_model)

        if self.pre_norm:
            ## Norm and Multi-Head attention and Add residual connection
            attn_output, attn_weights, _ = self.self_attn(
                hidden_states=self.norm_sublayer1(hidden_state), output_attentions=output_attentions
            )
            # Add: residual connection with residual dropout
            hidden_state = hidden_state + self.dropout_path1(attn_output)
        else:
            ## Multi-Head attention and Add residual connection and Norm - Standard Transformer from BERT
            attn_output, attn_weights, _ = self.self_attn(
                hidden_states=hidden_state, output_attentions=output_attentions
            )
            # hidden_states: [(bs*num_channels) x sequence_length x d_model]
            hidden_state = self.norm_sublayer1(hidden_state + self.dropout_path1(attn_output))

        # hidden_state: [bs x num_channels x sequence_length x d_model]
        hidden_state = hidden_state.reshape(batch_size, num_input_channels, sequence_length, d_model)

        # second sublayer: attention across variable at any given time
        if self.channel_attention:
            # hidden_state: [bs x sequence_length x num_channels x d_model]
            hidden_state = hidden_state.transpose(2, 1).contiguous()
            # hidden_state: [(bs*sequence_length) x num_channels x d_model]
            hidden_state = hidden_state.view(batch_size * sequence_length, num_input_channels, d_model)
            if self.pre_norm:
                ## Norm and Multi-Head attention and Add residual connection
                attn_output, channel_attn_weights, _ = self.self_attn(
                    hidden_states=self.norm_sublayer2(hidden_state), output_attentions=output_attentions
                )
                # Add: residual connection with residual dropout
                hidden_state = hidden_state + self.dropout_path2(attn_output)
            else:
                ## Multi-Head attention and Add residual connection and Norm
                attn_output, channel_attn_weights, _ = self.self_attn(
                    hidden_states=hidden_state, output_attentions=output_attentions
                )
                # hidden_states: [(bs*sequence_length) x num_channels x d_model]
                hidden_state = self.norm_sublayer2(hidden_state + self.dropout_path2(attn_output))

            # Reshape hidden state
            # hidden_state: [bs x sequence_length x num_channels x d_model]
            hidden_state = hidden_state.reshape(batch_size, sequence_length, num_input_channels, d_model)
            # hidden_state: [bs x num_channels x sequence_length x d_model]
            hidden_state = hidden_state.transpose(1, 2).contiguous()

        # Third sublayer: mixing across hidden
        # hidden_state: [(batch_size*num_channels) x sequence_length x d_model]
        hidden_state = hidden_state.view(batch_size * num_input_channels, sequence_length, d_model)
        if self.pre_norm:
            ## Norm and Position-wise Feed-Forward and Add residual connection
            # Add: residual connection with residual dropout
            hidden_state = hidden_state + self.dropout_path3(self.ff(self.norm_sublayer3(hidden_state)))
        else:
            ## Position-wise Feed-Forward and Add residual connection and Norm
            # Add: residual connection with residual dropout
            hidden_state = self.norm_sublayer3(hidden_state + self.dropout_path3(self.ff(hidden_state)))

        # [bs x num_channels x sequence_length x d_model]
        hidden_state = hidden_state.reshape(batch_size, num_input_channels, sequence_length, d_model)

        outputs = (hidden_state,)
        if output_attentions:
            outputs += (attn_weights, channel_attn_weights) if self.channel_attention else (attn_weights,)

        return outputs


class PatchTSTPreTrainedModel(PreTrainedModel):
    config_class = PatchTSTConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """
        Initialize weights
        """
        if isinstance(module, PatchTSTPositionalEncoding):
            # initialize cls_token
            if self.config.use_cls_token:
                nn.init.normal_(module.cls_token, std=0.02)
            # initialize positional encoding
            if self.config.positional_encoding_type == "random":
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PatchTSTBatchNorm):
            module.batchnorm.bias.data.zero_()
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PatchTSTEncoder)):
            module.gradient_checkpointing = value


class PatchTSTEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.num_input_channels = config.num_input_channels
        self.share_embedding = config.share_embedding
        # Input encoding: projection of feature vectors onto a d-dim vector space
        if self.share_embedding:
            self.input_embedding = nn.Linear(config.patch_length, config.d_model)
        else:
            self.input_embedding = nn.ModuleList()
            for _ in range(config.num_input_channels):
                self.input_embedding.append(nn.Linear(config.patch_length, config.d_model))

    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
        """
        # Input encoding
        num_input_channels = patch_input.shape[1]
        if num_input_channels != self.num_input_channels:
            raise ValueError(
                f"The defined number of input channels ({self.num_input_channels}) in the config "
                f"has to be the same as the number of channels in the batch input ({num_input_channels})"
            )
        if self.share_embedding:
            embeddings = self.input_embedding(patch_input)  # x: [bs x num_channels  x num_patches x d_model]
        else:
            embeddings = [self.input_embedding[i](patch_input[:, i, :, :]) for i in range(num_input_channels)]
            embeddings = torch.stack(embeddings, dim=1)
        return embeddings


class PatchTSTPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, config: PatchTSTConfig, num_patches: int):
        super().__init__()
        self.use_cls_token = config.use_cls_token
        self.num_input_channels = config.num_input_channels
        if config.use_cls_token:
            # cls_token: [1 x num_input_channels x 1 x d_model]
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, config.d_model))
            num_patches += 1
        # postional encoding: [num_patches x d_model]
        self.position_enc = self._init_pe(config, num_patches)
        # Positional dropout
        self.positional_dropout = (
            nn.Dropout(config.positional_dropout) if config.positional_dropout > 0 else nn.Identity()
        )

    @staticmethod
    def _init_pe(config: PatchTSTConfig, num_patches: int) -> nn.Parameter:
        # Positional encoding
        if config.positional_encoding_type == "random":
            position_enc = nn.Parameter(torch.randn(num_patches, config.d_model), requires_grad=True)
        elif config.positional_encoding_type == "sincos":
            position_enc = torch.zeros(num_patches, config.d_model)
            position = torch.arange(0, num_patches).unsqueeze(1)
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
        if self.use_cls_token:
            # patch_input: [bs x num_channels x num_patches x d_model]
            patch_input = self.positional_dropout(patch_input + self.position_enc[1:, :])
            # append cls token where cls_token: [1 x num_channels x 1 x d_model]
            cls_token = self.cls_token + self.position_enc[:1, :]
            # get the same copy of cls_token for all the samples in batch: [bs x num_channels x 1 x d_model]
            cls_tokens = cls_token.expand(patch_input.shape[0], self.num_input_channels, -1, -1)
            # hidden_state: [bs x num_channels x (num_patches+1) x d_model]
            hidden_state = torch.cat((cls_tokens, patch_input), dim=2)
        else:
            # hidden_state: [bs x num_channels x num_patches x d_model]
            hidden_state = self.positional_dropout(patch_input + self.position_enc)
        return hidden_state


class PatchTSTEncoder(PatchTSTPreTrainedModel):
    """
    PatchTST Encoder
    """

    def __init__(self, config: PatchTSTConfig, num_patches: int):
        super().__init__(config)
        self.gradient_checkpointing = False

        # Input embedding: projection of feature vectors onto a d-dim vector space
        self.embedder = PatchTSTEmbedding(config)
        # Positional encoding
        self.positional_encoder = PatchTSTPositionalEncoding(config, num_patches)
        # Encoder
        self.layers = nn.ModuleList([PatchTSTEncoderLayer(config) for i in range(config.num_hidden_layers)])

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        patch_input: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> BaseModelOutput:
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Past values of the time series
            output_hidden_states (bool, optional): Indicates if hidden states should be outputted.
            output_attentions (bool, optional): Indicates if attentions should be outputted.

        return:
            `BaseModelOutput`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Input embedding
        patch_input = self.embedder(patch_input)
        # Positional encoding
        hidden_state = self.positional_encoder(patch_input)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_state,)

            layer_outputs = encoder_layer(hidden_state=hidden_state, output_attentions=output_attentions)
            # get hidden state. hidden_state shape is [bs x num_channels x num_patches x d_model]
            # or [bs x num_channels x (num_patches+1) x d_model] if use cls_token
            hidden_state = layer_outputs[0]
            # append attention matrix at each layer
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        # return past_values, hidden_states
        return BaseModelOutput(last_hidden_state=hidden_state, hidden_states=encoder_states, attentions=all_attentions)


PATCHTST_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@dataclass
class PatchTSTModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Parameters:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`, *optional*)
            Bool masked tensor indicating which patches are masked
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            Mean of the input data (batch_size, sequence_length, num_channels) over the sequence_length
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            Std of the input data (batch_size, sequence_length, num_channels) over the sequence_length
        patch_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Patched input to the Transformer
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask: torch.FloatTensor = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None
    patch_input: torch.FloatTensor = None


@dataclass
class PatchTSTForPretrainingOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPretraining`].

    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction outputs of the time series modeling heads.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PatchTSTForRegressionOutput(ModelOutput):
    """
    Output type of [`PatchTSTForRegression`].

    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.
        regression_outputs (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Regression outputs of the time series modeling heads.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    regression_outputs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PatchTSTForPredictionOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPrediction`].

    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, prediction_length, -1)`):
            Prediction outputs of the time series modeling heads.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            Mean of the input data (batch_size, sequence_length, num_channels) over the sequence_length
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            Std of the input data (batch_size, sequence_length, num_channels) over the sequence_length
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


@dataclass
class PatchTSTForClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSTForClassification`].

    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Prediction scores of the PatchTST modeling head (scores before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SamplePatchTSTOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Parameters:
        sequences `(batch_size, num_samples, prediction_length, num_targets)`):
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


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesStdScaler with TimeSeriesTransformer->PatchTST,TimeSeries->PatchTST
class PatchTSTStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: PatchTSTConfig):
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


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesMeanScaler with TimeSeriesTransformer->PatchTST,TimeSeries->PatchTST
class PatchTSTMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: PatchTSTConfig):
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


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesNOPScaler with TimeSeriesTransformer->PatchTST,TimeSeries->PatchTST
class PatchTSTNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, config: PatchTSTConfig):
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


class PatchTSTScaler(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        if config.scaling == "mean" or config.scaling is True:
            self.scaler = PatchTSTMeanScaler(config)
        elif config.scaling == "std":
            self.scaler = PatchTSTStdScaler(config)
        else:
            self.scaler = PatchTSTNOPScaler(config)

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Input for scaler calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, um_input_channels)`)
        """
        data, loc, scale = self.scaler(data, observed_indicator)
        return data, loc, scale


@add_start_docstrings(
    "The bare PatchTST Model outputting raw hidden-states without any specific head.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTModel(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.scaler = PatchTSTScaler(config)
        self.patchifier = PatchTSTPatchify(config)
        self.do_mask_input = config.do_mask_input
        # get num_patches information from PatchTSTPatchify
        num_patches = self.patchifier.num_patches

        if self.do_mask_input:
            self.masking = PatchTSTMasking(config)
        else:
            self.masking = nn.Identity()
        self.encoder = PatchTSTEncoder(config, num_patches=num_patches)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSTModelOutput]:
        r"""
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            future_values (`torch.BoolTensor` of shape `(batch_size, prediction_length, num_input_channels)`, *optional*):
                Future target values associated with the `past_values`
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTModelOutput` or tuple of `torch.Tensor` (if `return_dict`=False or `config.return_dict`=False)

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import PatchTSTModel

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> model = PatchTSTModel.from_pretrained("namctin/patchtst_etth1_pretrain")

        >>> # during training, one provides both past and future values
        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     future_values=batch["future_values"],
        ... )

        >>> last_hidden_state = outputs.last_hidden_state
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        # x: tensor [bs x sequence_length x num_input_channels]
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)

        # patched_values: [bs x num_input_channels x num_patches x patch_length] for pretrain
        patched_values = self.patchifier(scaled_past_values)
        if self.do_mask_input:
            masked_values, mask = self.masking(patched_values)
        else:
            masked_values, mask = self.masking(patched_values), None

        encoder_output = self.encoder(
            patch_input=masked_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )

        if not return_dict:
            outputs = (encoder_output.last_hidden_state, encoder_output.hidden_states, encoder_output.attentions)
            outputs = outputs + (mask, loc, scale, patched_values)
            return tuple(v for v in outputs if v is not None)

        return PatchTSTModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
            mask=mask,
            loc=loc,
            scale=scale,
            patch_input=patched_values,
        )


class PatchTSTMaskPretrainHead(nn.Module):
    """
    Pretraining head for mask modelling
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        self.linear = nn.Linear(config.d_model, config.patch_length)
        self.use_cls_token = config.use_cls_token

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                    `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                            `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True

        """
        embedding = self.linear(self.dropout(embedding))  # [bs x num_channels x num_patches x patch_length]
        if self.use_cls_token:
            embedding = embedding[:, :, 1:, :]  # remove the first cls token
        return embedding


@add_start_docstrings(
    "The PatchTST for pretrain model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTForPretraining(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        config.do_mask_input = True
        self.model = PatchTSTModel(config=config)
        self.head = PatchTSTMaskPretrainHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSTForPretrainingOutput]:
        r"""
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*): Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForPretrainingOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import PatchTSTConfig, PatchTSTForPretraining

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> # Config for random mask pretraining
        >>> config = PatchTSTConfig(
        ...     num_input_channels=7,
        ...     context_length=512,
        ...     patch_length=12,
        ...     stride=12,
        ...     mask_type='random',
        ...     random_mask_ratio=0.4,
        ...     use_cls_token=True,
        ... )
        >>> # Config for forecast mask pretraining
        >>> config = PatchTSTConfig(
        ...     num_input_channels=7,
        ...     context_length=512,
        ...     patch_length=12,
        ...     stride=12,
        ...     mask_type='forecast',
        ...     num_forecast_mask_patches=5,
        ...     use_cls_token=True,
        ... )
        >>> model = PatchTSTForPretraining(config)

        >>> # during training, one provides both past and future values
        >>> outputs = model(past_values=batch["past_values"])

        >>> loss = outputs.loss
        >>> loss.backward()
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # past_values: [bs x num_channels x num_patches x d_model] or
        # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # last_hidden_state: [bs x num_channels x num_patches x patch_length] or
        # [bs x num_channels x (num_patches+1) x patch_length] if use cls_token
        x_hat = self.head(model_output.last_hidden_state)

        # calculate masked_loss
        loss = nn.MSELoss(reduction="none")
        loss_val = loss(x_hat, model_output.patch_input)
        masked_loss = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)

        encoder_states = model_output.hidden_states
        if not return_dict:
            outputs = (x_hat,) + model_output[1:-4]
            outputs = (masked_loss,) + outputs if masked_loss is not None else outputs
            return outputs
        return PatchTSTForPretrainingOutput(
            loss=masked_loss, prediction_output=x_hat, hidden_states=encoder_states, attentions=model_output.attentions
        )


class PatchTSTClassificationHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.use_cls_token = config.use_cls_token
        self.pooling_type = config.pooling_type
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        self.linear = nn.Linear(config.num_input_channels * config.d_model, config.num_targets)

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                     `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, num_targets)`

        """
        if self.use_cls_token:
            # use the first output token, pooled_embedding: bs x num_channels x d_model
            pooled_embedding = embedding[:, :, 0, :]
        elif self.pooling_type == "mean":
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.mean(dim=2)
        elif self.pooling_type == "max":
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.max(dim=2).values
        else:
            raise ValueError(f"pooling operator {self.pooling_type} is not implemented yet")
        # pooled_embedding: bs x num_channels * d_model
        pooled_embedding = self.flatten(pooled_embedding)
        # output: bs x n_classes
        output = self.linear(self.dropout(pooled_embedding))
        return output


@add_start_docstrings(
    "The PatchTST for classification model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTForClassification(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        # Turn off masking
        if config.do_mask_input:
            logger.warning("Setting `do_mask_input` parameter to False.")
            config.do_mask_input = False

        self.model = PatchTSTModel(config)
        self.head = PatchTSTClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        past_observed_mask: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, PatchTSTForClassificationOutput]:
        r"""
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            target_values (`torch.Tensor`, *optional*):
                Labels associates with the `past_values`
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForClassificationOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        Examples:

        ```python
        >>> from transformers import PatchTSTConfig, PatchTSTForClassification

        >>> # classification task with two input channel2 and 3 classes
        >>> config = PatchTSTConfig(
        ...     num_input_channels=2,
        ...     num_targets=3,
        ...     context_length=512,
        ...     patch_length=12,
        ...     stride=12,
        ...     use_cls_token=True,
        ... )
        >>> model = PatchTSTForClassification(config=config)

        >>> # during inference, one only provides past values
        >>> past_values = torch.randn(20, 512, 2)
        >>> outputs = model(past_values=past_values)
        >>> labels = outputs.prediction_logits
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        y_hat = self.head(model_output.last_hidden_state)

        loss_val = None
        if target_values is not None:
            loss = nn.CrossEntropyLoss()
            loss_val = loss(y_hat, target_values)

        if not return_dict:
            outputs = (y_hat,) + model_output[1:-3]
            outputs = (loss_val,) + outputs if loss_val is not None else outputs
            return outputs
        return PatchTSTForClassificationOutput(
            loss=loss_val,
            prediction_logits=y_hat,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
        )


@add_start_docstrings(
    "The PatchTST for regression Model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTPredictionHead(nn.Module):
    def __init__(self, config: PatchTSTConfig, num_patches, distribution_output=None):
        super().__init__()

        self.share_projection = config.share_projection
        self.num_input_channels = config.num_input_channels
        self.use_cls_token = config.use_cls_token
        self.pooling_type = config.pooling_type
        if self.pooling_type or self.use_cls_token:
            head_dim = config.d_model
        else:
            head_dim = config.d_model * num_patches

        if not self.share_projection:
            # if each channel has its own head
            self.projections = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.num_input_channels):
                self.flattens.append(nn.Flatten(start_dim=2))
                if distribution_output is None:
                    # use linear head
                    self.projections.append(nn.Linear(head_dim, config.prediction_length))
                else:
                    # use distribution head
                    self.projections.append(distribution_output.get_parameter_projection(head_dim))
                self.dropouts.append(nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity())
        else:
            # all the channels share the same head
            self.flatten = nn.Flatten(start_dim=2)
            if distribution_output is None:
                # use linear head
                self.projection = nn.Linear(head_dim, config.prediction_length)
            else:
                # use distribution head
                self.projection = distribution_output.get_parameter_projection(head_dim)
            self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                     `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, forecast_len, num_channels)`

        """
        if self.use_cls_token:
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding[:, :, 0, :]
        else:
            if self.pooling_type == "mean":
                # pooled_embedding: [bs x num_channels x d_model]
                pooled_embedding = embedding.mean(dim=2)
            elif self.pooling_type == "max":
                # pooled_embedding: [bs x num_channels x d_model]
                pooled_embedding = embedding.max(dim=2).values
            else:
                # pooled_embedding: [bs x num_channels x num_patches x d_model]
                pooled_embedding = embedding

        if not self.share_projection:
            output = []
            for i in range(self.num_input_channels):
                # pooled_embedding: [bs x (d_model * num_patches)] or [bs x d_model)]
                pooled_embedding = self.flattens[i](pooled_embedding[:, i, :])
                pooled_embedding = self.dropouts[i](pooled_embedding)
                # pooled_embedding: [bs x forecast_len]
                #  or tuple ([bs x forecast_len], [bs x forecast_len]) if using distribution head
                pooled_embedding = self.projections[i](pooled_embedding)
                output.append(pooled_embedding)
            # output: [bs x num_channels x forecast_len]
            output = torch.stack(output, dim=1)
        else:
            # pooled_embedding: [bs x num_channels x (d_model * num_patches)] or [bs x num_channels x d_model)]
            pooled_embedding = self.flatten(pooled_embedding)
            pooled_embedding = self.dropout(pooled_embedding)
            # output: [bs x num_channels x forecast_len] or
            # tuple ([bs x num_channels x forecast_len], [bs x num_channels x forecast_len]) if using distribution head
            output = self.projection(pooled_embedding)

        if isinstance(output, tuple):
            # output: ([bs x forecast_len x num_channels], [bs x forecast_len x num_channels])
            output = tuple(z.transpose(2, 1) for z in output)
        else:
            output = output.transpose(2, 1)  # [bs x forecast_len x num_channels]
        return output


@add_start_docstrings(
    "The PatchTST for prediction model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        # Turn off masking
        if config.do_mask_input:
            logger.warning("Setting `do_mask_input` parameter to False.")
            config.do_mask_input = False

        self.model = PatchTSTModel(config)

        if config.loss == "mse":
            self.distribution_output = None
        else:
            if config.distribution_output == "student_t":
                self.distribution_output = StudentTOutput(dim=config.prediction_length)
            elif config.distribution_output == "normal":
                self.distribution_output = NormalOutput(dim=config.prediction_length)
            elif config.distribution_output == "negative_binomial":
                self.distribution_output = NegativeBinomialOutput(dim=config.prediction_length)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.head = PatchTSTPredictionHead(
            config, self.model.patchifier.num_patches, distribution_output=self.distribution_output
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSTForPredictionOutput]:
        r"""
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            future_values (`torch.Tensor` of shape `(bs, forecast_len, num_input_channels)`, *optional*):
                Future target values associated with the `past_values`
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForPredictionOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        Examples:

        ```python
        >>> from huggingface_hub import hf_hub_download
        >>> import torch
        >>> from transformers import PatchTSTConfig, PatchTSTForPrediction

        >>> file = hf_hub_download(
        ...     repo_id="hf-internal-testing/etth1-hourly-batch", filename="train-batch.pt", repo_type="dataset"
        ... )
        >>> batch = torch.load(file)

        >>> # Prediction task with 7 input channels and prediction length is 96
        >>> model = PatchTSTForPrediction.from_pretrained("namctin/patchtst_etth1_forecast")

        >>> # during training, one provides both past and future values
        >>> outputs = model(
        ...     past_values=batch["past_values"],
        ...     future_values=batch["future_values"],
        ... )

        >>> loss = outputs.loss
        >>> loss.backward()

        >>> # during inference, one only provides past values, the model outputs future values
        >>> outputs = model(past_values=batch["past_values"])
        >>> prediction_outputs = outputs.prediction_outputs
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # get model output
        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        # get output head
        y_hat = self.head(model_output.last_hidden_state)

        loss_val = None

        if self.distribution_output:
            y_hat_out = y_hat
        else:
            y_hat_out = y_hat * model_output.scale + model_output.loc

        if future_values is not None:
            if self.distribution_output:
                distribution = self.distribution_output.distribution(
                    y_hat, loc=model_output.loc, scale=model_output.scale
                )
                loss_val = nll(distribution, future_values)
                # take average of the loss
                loss_val = weighted_average(loss_val)
            else:
                loss = nn.MSELoss(reduction="mean")
                loss_val = loss(y_hat_out, future_values)

        loc = model_output.loc
        scale = model_output.scale

        if not return_dict:
            outputs = (y_hat_out,) + model_output[1:-1]
            outputs = (loss_val,) + outputs if loss_val is not None else outputs
            return outputs
        return PatchTSTForPredictionOutput(
            loss=loss_val,
            prediction_outputs=y_hat_out,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
            loc=loc,
            scale=scale,
        )

    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
    ) -> SamplePatchTSTOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSTOutput`] where the outputs `sequences` tensor will have shape `(batch_size, number of
            samples, prediction_length, 1)` or `(batch_size, number of samples, prediction_length, num_input_channels)`
            for multivariate predictions.
        """
        # get number of samples
        num_parallel_samples = self.config.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            future_values=None,
            past_observed_mask=past_observed_mask,
            output_hidden_states=False,
        )
        if self.distribution_output:
            # get distribution
            distribution = self.distribution_output.distribution(
                outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale
            )
            # get samples: list of [bs x forecast_len x num_channels]
            samples = [distribution.sample() for _ in range(num_parallel_samples)]
            # samples: [bs x num_samples x forecast_len x num_channels]
            samples = torch.stack(samples, dim=1)
        else:
            samples = outputs.prediction_outputs.unsqueeze(1)

        return SamplePatchTSTOutput(sequences=samples)


class PatchTSTRegressionHead(nn.Module):
    """
    Regression head
    """

    def __init__(self, config: PatchTSTConfig, distribution_output=None):
        super().__init__()
        self.y_range = config.output_range
        self.use_cls_token = config.use_cls_token
        self.pooling_type = config.pooling_type
        self.distribution_output = distribution_output

        head_dim = config.num_input_channels * config.d_model

        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

        if distribution_output is None:
            self.projection = nn.Linear(head_dim, config.num_targets)
        else:
            self.projection = distribution_output.get_parameter_projection(head_dim)

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                    `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, output_dim)`

        """
        if self.use_cls_token:
            # use the first output token, pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding[:, :, 0, :]
        elif self.pooling_type == "mean":
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.mean(dim=2)
        elif self.pooling_type == "max":
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.max(dim=2).values
        else:
            raise ValueError(f"pooling operator {self.pooling_type} is not implemented yet")
        # flatten the input
        # pooled_embedding: bs x (num_channels * d_model)
        pooled_embedding = self.dropout(self.flatten(pooled_embedding))
        # projection
        # output: bs x output_dim or a tuple of this shape for distribution head
        output = self.projection(pooled_embedding)
        # apply sigmoid to bound the output if required
        if (self.distribution_output is None) & (self.y_range is not None):  # linear head
            output = torch.sigmoid(output) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return output


@add_start_docstrings(
    "The PatchTST for regression model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTForRegression(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        # Turn off masking
        if config.do_mask_input:
            logger.warning("Setting `do_mask_input` parameter to False.")
            config.do_mask_input = False

        self.model = PatchTSTModel(config)
        if config.loss == "mse":
            self.distribution_output = None
        else:
            if config.distribution_output == "student_t":
                self.distribution_output = StudentTOutput(dim=config.num_targets)
            elif config.distribution_output == "normal":
                self.distribution_output = NormalOutput(dim=config.num_targets)
            elif config.distribution_output == "negative_binomial":
                self.distribution_output = NegativeBinomialOutput(dim=config.num_targets)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.head = PatchTSTRegressionHead(config, self.distribution_output)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, PatchTSTForRegressionOutput]:
        r"""
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            target_values (`torch.Tensor` of shape `(bs, num_input_channels)`):
                Target values associates with the `past_values`
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForRegressionOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        Examples:

        ```python
        >>> from transformers import PatchTSTConfig, PatchTSTForRegression

        >>> # Regression task with 6 input channels and regress 2 targets
        >>> model = PatchTSTForRegression.from_pretrained("namctin/patchtst_etth1_regression")

        >>> # during inference, one only provides past values, the model outputs future values
        >>> past_values = torch.randn(20, 512, 6)
        >>> outputs = model(past_values=past_values)
        >>> regression_outputs = outputs.regression_outputs
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        # get output head. y_hat is of shape [bs x num_targets] or tuple of this shape
        y_hat = self.head(model_output.last_hidden_state)

        loss = None
        if target_values is not None:
            if self.distribution_output:
                distribution = self.distribution_output.distribution(y_hat)
                # y_hat should be a 2-tuple, each with dimension [bs, num_targets]
                y_hat = tuple([item.view(-1, self.config.num_targets) for item in y_hat])
                loss = nll(distribution, target_values)
                # take average of the loss
                loss = weighted_average(loss)
            else:
                loss = nn.MSELoss(reduction="mean")
                loss = loss(y_hat, target_values)

        if not return_dict:
            # hidden_states, attentions, mask
            outputs = (y_hat,) + model_output[1:-3]
            outputs = (loss,) + outputs if loss is not None else outputs
            return outputs
        return PatchTSTForRegressionOutput(
            loss=loss,
            regression_outputs=y_hat,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
        )

    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
    ) -> SamplePatchTSTOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSTOutput`] where the outputs `sequences` tensor will have shape `(batch_size, number of
            samples, num_targets)`.
        """
        # get number of samples
        num_parallel_samples = self.config.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            target_values=None,
            past_observed_mask=past_observed_mask,
            output_hidden_states=False,
        )

        # get distribution
        distribution = self.distribution_output.distribution(outputs.regression_outputs)
        # get samples: list of [bs x num_targets]
        samples = [distribution.sample() for _ in range(num_parallel_samples)]
        # samples: [bs x num_samples x num_targets]
        samples = torch.stack(samples, dim=1).view(-1, num_parallel_samples, self.config.num_targets)
        return SamplePatchTSTOutput(sequences=samples)


__all__ = [
    "PatchTSTModel",
    "PatchTSTPreTrainedModel",
    "PatchTSTForPrediction",
    "PatchTSTForPretraining",
    "PatchTSTForRegression",
    "PatchTSTForClassification",
]

# coding=utf-8
# Copyright 2023 The Kakao Enterprise Authors and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch VITS model."""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "VitsConfig"


@dataclass
class VitsModelOutput(ModelOutput):
    """
    Describes the outputs for the VITS model, with potential hidden states and attentions.

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The final audio waveform predicted by the model.
        sequence_lengths  (`torch.FloatTensor` of shape `(batch_size,)`):
            The length in samples of each element in the `waveform` batch.
        spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`):
            The log-mel spectrogram predicted at the output of the flow model. This spectrogram is passed to the Hi-Fi
            GAN decoder model to obtain the final audio waveform.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attention weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    waveform: torch.FloatTensor = None
    sequence_lengths: torch.FloatTensor = None
    spectrogram: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class VitsTextEncoderOutput(ModelOutput):
    """
    Describes the outputs for the VITS text encoder model, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        prior_means (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The predicted mean values of the prior distribution for the latent text variables.
        prior_log_variances (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The predicted log-variance values of the prior distribution for the latent text variables.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attention weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    prior_means: torch.FloatTensor = None
    prior_log_variances: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels):
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :num_channels, :])
    s_act = torch.sigmoid(in_act[:, num_channels:, :])
    acts = t_act * s_act
    return acts


def _unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse=False,
    tail_bound=5.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    """
    This transformation represents a monotonically increasing piecewise rational quadratic function. Outside of the
    `tail_bound`, the transform behaves as an identity function.

    Args:
        inputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Second half of the hidden-states input to the Vits convolutional flow module.
        unnormalized_widths (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            First `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_heights (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Second `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_derivatives (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Third `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        reverse (`bool`, *optional*, defaults to `False`):
            Whether the model is being run in reverse mode.
        tail_bound (`float`, *optional* defaults to 5):
            Upper and lower limit bound for the rational quadratic function. Outside of this `tail_bound`, the
            transform behaves as an identity function.
        min_bin_width (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the width dimension for the piecewise rational quadratic function.
        min_bin_height (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the height dimension for the piecewise rational quadratic function.
        min_derivative (`float`, *optional*, defaults to 1e-3):
            Minimum bin value across the derivatives for the piecewise rational quadratic function.
    Returns:
        outputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Hidden-states as transformed by the piecewise rational quadratic function with the `tail_bound` limits
            applied.
        log_abs_det (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Logarithm of the absolute value of the determinants corresponding to the `outputs` with the `tail_bound`
            limits applied.
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    log_abs_det = torch.zeros_like(inputs)
    constant = np.log(np.exp(1 - min_derivative) - 1)

    unnormalized_derivatives = nn.functional.pad(unnormalized_derivatives, pad=(1, 1))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    log_abs_det[outside_interval_mask] = 0.0

    outputs[inside_interval_mask], log_abs_det[inside_interval_mask] = _rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        reverse=reverse,
        tail_bound=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    return outputs, log_abs_det


def _rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse,
    tail_bound,
    min_bin_width,
    min_bin_height,
    min_derivative,
):
    """
    This transformation represents a monotonically increasing piecewise rational quadratic function. Unlike the
    function `_unconstrained_rational_quadratic_spline`, the function behaves the same across the `tail_bound`.

    Args:
        inputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Second half of the hidden-states input to the Vits convolutional flow module.
        unnormalized_widths (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            First `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_heights (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Second `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_derivatives (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Third `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        reverse (`bool`):
            Whether the model is being run in reverse mode.
        tail_bound (`float`):
            Upper and lower limit bound for the rational quadratic function. Outside of this `tail_bound`, the
            transform behaves as an identity function.
        min_bin_width (`float`):
            Minimum bin value across the width dimension for the piecewise rational quadratic function.
        min_bin_height (`float`):
            Minimum bin value across the height dimension for the piecewise rational quadratic function.
        min_derivative (`float`):
            Minimum bin value across the derivatives for the piecewise rational quadratic function.
    Returns:
        outputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Hidden-states as transformed by the piecewise rational quadratic function.
        log_abs_det (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Logarithm of the absolute value of the determinants corresponding to the `outputs`.
    """
    upper_bound = tail_bound
    lower_bound = -tail_bound

    if torch.min(inputs) < lower_bound or torch.max(inputs) > upper_bound:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError(f"Minimal bin width {min_bin_width} too large for the number of bins {num_bins}")
    if min_bin_height * num_bins > 1.0:
        raise ValueError(f"Minimal bin height {min_bin_height} too large for the number of bins {num_bins}")

    widths = nn.functional.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = nn.functional.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (upper_bound - lower_bound) * cumwidths + lower_bound
    cumwidths[..., 0] = lower_bound
    cumwidths[..., -1] = upper_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + nn.functional.softplus(unnormalized_derivatives)

    heights = nn.functional.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = nn.functional.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (upper_bound - lower_bound) * cumheights + lower_bound
    cumheights[..., 0] = lower_bound
    cumheights[..., -1] = upper_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_locations = cumheights if reverse else cumwidths
    bin_locations[..., -1] += 1e-6
    bin_idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
    bin_idx = bin_idx[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    intermediate1 = input_derivatives + input_derivatives_plus_one - 2 * input_delta
    if not reverse:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, log_abs_det
    else:
        # find the roots of a quadratic equation
        intermediate2 = inputs - input_cumheights
        intermediate3 = intermediate2 * intermediate1
        a = input_heights * (input_delta - input_derivatives) + intermediate3
        b = input_heights * input_derivatives - intermediate3
        c = -input_delta * intermediate2

        discriminant = b.pow(2) - 4 * a * c
        if not (discriminant >= 0).all():
            raise RuntimeError(f"invalid discriminant {discriminant}")

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -log_abs_det


class VitsWaveNet(torch.nn.Module):
    def __init__(self, config: VitsConfig, num_layers: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_layers

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(config.wavenet_dropout)

        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        else:
            weight_norm = nn.utils.weight_norm

        if config.speaker_embedding_size != 0:
            cond_layer = torch.nn.Conv1d(config.speaker_embedding_size, 2 * config.hidden_size * num_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name="weight")

        for i in range(num_layers):
            dilation = config.wavenet_dilation_rate**i
            padding = (config.wavenet_kernel_size * dilation - dilation) // 2
            in_layer = torch.nn.Conv1d(
                in_channels=config.hidden_size,
                out_channels=2 * config.hidden_size,
                kernel_size=config.wavenet_kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < num_layers - 1:
                res_skip_channels = 2 * config.hidden_size
            else:
                res_skip_channels = config.hidden_size

            res_skip_layer = torch.nn.Conv1d(config.hidden_size, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        outputs = torch.zeros_like(inputs)
        num_channels_tensor = torch.IntTensor([self.hidden_size])

        if global_conditioning is not None:
            global_conditioning = self.cond_layer(global_conditioning)

        for i in range(self.num_layers):
            hidden_states = self.in_layers[i](inputs)

            if global_conditioning is not None:
                cond_offset = i * 2 * self.hidden_size
                global_states = global_conditioning[:, cond_offset : cond_offset + 2 * self.hidden_size, :]
            else:
                global_states = torch.zeros_like(hidden_states)

            acts = fused_add_tanh_sigmoid_multiply(hidden_states, global_states, num_channels_tensor[0])
            acts = self.dropout(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_size, :]
                inputs = (inputs + res_acts) * padding_mask
                outputs = outputs + res_skip_acts[:, self.hidden_size :, :]
            else:
                outputs = outputs + res_skip_acts

        return outputs * padding_mask

    def remove_weight_norm(self):
        if self.speaker_embedding_size != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)


class VitsPosteriorEncoder(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.out_channels = config.flow_size

        self.conv_pre = nn.Conv1d(config.spectrogram_bins, config.hidden_size, 1)
        self.wavenet = VitsWaveNet(config, num_layers=config.posterior_encoder_num_wavenet_layers)
        self.conv_proj = nn.Conv1d(config.hidden_size, self.out_channels * 2, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        inputs = self.conv_pre(inputs) * padding_mask
        inputs = self.wavenet(inputs, padding_mask, global_conditioning)
        stats = self.conv_proj(inputs) * padding_mask
        mean, log_stddev = torch.split(stats, self.out_channels, dim=1)
        sampled = (mean + torch.randn_like(mean) * torch.exp(log_stddev)) * padding_mask
        return sampled, mean, log_stddev


# Copied from transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock
class HifiGanResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        for layer in self.convs1:
            weight_norm(layer)
        for layer in self.convs2:
            weight_norm(layer)

    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


class VitsHifiGan(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(
            config.flow_size,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3, bias=False)

        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.upsample_initial_channel, 1)

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        for layer in self.upsampler:
            weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()

    def remove_weight_norm(self):
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()

    def forward(
        self, spectrogram: torch.FloatTensor, global_conditioning: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        r"""
        Converts a spectrogram into a speech waveform.

        Args:
            spectrogram (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`):
                Tensor containing the spectrograms.
            global_conditioning (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_size, 1)`, *optional*):
                Tensor containing speaker embeddings, for multispeaker models.

        Returns:
            `torch.FloatTensor`: Tensor of shape shape `(batch_size, 1, num_frames)` containing the speech waveform.
        """
        hidden_states = self.conv_pre(spectrogram)

        if global_conditioning is not None:
            hidden_states = hidden_states + self.cond(global_conditioning)

        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        waveform = torch.tanh(hidden_states)
        return waveform


class VitsResidualCouplingLayer(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.half_channels = config.flow_size // 2

        self.conv_pre = nn.Conv1d(self.half_channels, config.hidden_size, 1)
        self.wavenet = VitsWaveNet(config, num_layers=config.prior_encoder_num_wavenet_layers)
        self.conv_post = nn.Conv1d(config.hidden_size, self.half_channels, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)
        hidden_states = self.conv_pre(first_half) * padding_mask
        hidden_states = self.wavenet(hidden_states, padding_mask, global_conditioning)
        mean = self.conv_post(hidden_states) * padding_mask
        log_stddev = torch.zeros_like(mean)

        if not reverse:
            second_half = mean + second_half * torch.exp(log_stddev) * padding_mask
            outputs = torch.cat([first_half, second_half], dim=1)
            log_determinant = torch.sum(log_stddev, [1, 2])
            return outputs, log_determinant
        else:
            second_half = (second_half - mean) * torch.exp(-log_stddev) * padding_mask
            outputs = torch.cat([first_half, second_half], dim=1)
            return outputs, None


class VitsResidualCouplingBlock(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(config.prior_encoder_num_flows):
            self.flows.append(VitsResidualCouplingLayer(config))

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                inputs, _ = flow(inputs, padding_mask, global_conditioning)
                inputs = torch.flip(inputs, [1])
        else:
            for flow in reversed(self.flows):
                inputs = torch.flip(inputs, [1])
                inputs, _ = flow(inputs, padding_mask, global_conditioning, reverse=True)
        return inputs


class VitsDilatedDepthSeparableConv(nn.Module):
    def __init__(self, config: VitsConfig, dropout_rate=0.0):
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        channels = config.hidden_size
        self.num_layers = config.depth_separable_num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.convs_dilated = nn.ModuleList()
        self.convs_pointwise = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_dilated.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_pointwise.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(nn.LayerNorm(channels))
            self.norms_2.append(nn.LayerNorm(channels))

    def forward(self, inputs, padding_mask, global_conditioning=None):
        if global_conditioning is not None:
            inputs = inputs + global_conditioning

        for i in range(self.num_layers):
            hidden_states = self.convs_dilated[i](inputs * padding_mask)
            hidden_states = self.norms_1[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.convs_pointwise[i](hidden_states)
            hidden_states = self.norms_2[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.dropout(hidden_states)
            inputs = inputs + hidden_states

        return inputs * padding_mask


class VitsConvFlow(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.filter_channels = config.hidden_size
        self.half_channels = config.depth_separable_channels // 2
        self.num_bins = config.duration_predictor_flow_bins
        self.tail_bound = config.duration_predictor_tail_bound

        self.conv_pre = nn.Conv1d(self.half_channels, self.filter_channels, 1)
        self.conv_dds = VitsDilatedDepthSeparableConv(config)
        self.conv_proj = nn.Conv1d(self.filter_channels, self.half_channels * (self.num_bins * 3 - 1), 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)

        hidden_states = self.conv_pre(first_half)
        hidden_states = self.conv_dds(hidden_states, padding_mask, global_conditioning)
        hidden_states = self.conv_proj(hidden_states) * padding_mask

        batch_size, channels, length = first_half.shape
        hidden_states = hidden_states.reshape(batch_size, channels, -1, length).permute(0, 1, 3, 2)

        unnormalized_widths = hidden_states[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = hidden_states[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = hidden_states[..., 2 * self.num_bins :]

        second_half, log_abs_det = _unconstrained_rational_quadratic_spline(
            second_half,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            reverse=reverse,
            tail_bound=self.tail_bound,
        )

        outputs = torch.cat([first_half, second_half], dim=1) * padding_mask
        if not reverse:
            log_determinant = torch.sum(log_abs_det * padding_mask, [1, 2])
            return outputs, log_determinant
        else:
            return outputs, None


class VitsElementwiseAffine(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.channels = config.depth_separable_channels
        self.translate = nn.Parameter(torch.zeros(self.channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(self.channels, 1))

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        if not reverse:
            outputs = self.translate + torch.exp(self.log_scale) * inputs
            outputs = outputs * padding_mask
            log_determinant = torch.sum(self.log_scale * padding_mask, [1, 2])
            return outputs, log_determinant
        else:
            outputs = (inputs - self.translate) * torch.exp(-self.log_scale) * padding_mask
            return outputs, None


class VitsStochasticDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.speaker_embedding_size
        filter_channels = config.hidden_size

        self.conv_pre = nn.Conv1d(filter_channels, filter_channels, 1)
        self.conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )

        if embed_dim != 0:
            self.cond = nn.Conv1d(embed_dim, filter_channels, 1)

        self.flows = nn.ModuleList()
        self.flows.append(VitsElementwiseAffine(config))
        for _ in range(config.duration_predictor_num_flows):
            self.flows.append(VitsConvFlow(config))

        self.post_conv_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )

        self.post_flows = nn.ModuleList()
        self.post_flows.append(VitsElementwiseAffine(config))
        for _ in range(config.duration_predictor_num_flows):
            self.post_flows.append(VitsConvFlow(config))

    def forward(self, inputs, padding_mask, global_conditioning=None, durations=None, reverse=False, noise_scale=1.0):
        inputs = torch.detach(inputs)
        inputs = self.conv_pre(inputs)

        if global_conditioning is not None:
            global_conditioning = torch.detach(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)

        inputs = self.conv_dds(inputs, padding_mask)
        inputs = self.conv_proj(inputs) * padding_mask

        if not reverse:
            hidden_states = self.post_conv_pre(durations)
            hidden_states = self.post_conv_dds(hidden_states, padding_mask)
            hidden_states = self.post_conv_proj(hidden_states) * padding_mask

            random_posterior = (
                torch.randn(durations.size(0), 2, durations.size(2)).to(device=inputs.device, dtype=inputs.dtype)
                * padding_mask
            )
            log_determinant_posterior_sum = 0
            latents_posterior = random_posterior
            for flow in self.post_flows:
                latents_posterior, log_determinant = flow(
                    latents_posterior, padding_mask, global_conditioning=inputs + hidden_states
                )
                latents_posterior = torch.flip(latents_posterior, [1])
                log_determinant_posterior_sum += log_determinant

            first_half, second_half = torch.split(latents_posterior, [1, 1], dim=1)

            log_determinant_posterior_sum += torch.sum(
                (nn.functional.logsigmoid(first_half) + nn.functional.logsigmoid(-first_half)) * padding_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (random_posterior**2)) * padding_mask, [1, 2])
                - log_determinant_posterior_sum
            )

            first_half = (durations - torch.sigmoid(first_half)) * padding_mask
            first_half = torch.log(torch.clamp_min(first_half, 1e-5)) * padding_mask
            log_determinant_sum = torch.sum(-first_half, [1, 2])

            latents = torch.cat([first_half, second_half], dim=1)
            for flow in self.flows:
                latents, log_determinant = flow(latents, padding_mask, global_conditioning=inputs)
                latents = torch.flip(latents, [1])
                log_determinant_sum += log_determinant

            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (latents**2)) * padding_mask, [1, 2]) - log_determinant_sum
            return nll + logq
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow

            latents = (
                torch.randn(inputs.size(0), 2, inputs.size(2)).to(device=inputs.device, dtype=inputs.dtype)
                * noise_scale
            )
            for flow in flows:
                latents = torch.flip(latents, [1])
                latents, _ = flow(latents, padding_mask, global_conditioning=inputs, reverse=True)

            log_duration, _ = torch.split(latents, [1, 1], dim=1)
            return log_duration


class VitsDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        filter_channels = config.duration_predictor_filter_channels

        self.dropout = nn.Dropout(config.duration_predictor_dropout)
        self.conv_1 = nn.Conv1d(config.hidden_size, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.hidden_size, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        inputs = torch.detach(inputs)

        if global_conditioning is not None:
            global_conditioning = torch.detach(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)

        inputs = self.conv_1(inputs * padding_mask)
        inputs = torch.relu(inputs)
        inputs = self.norm_1(inputs.transpose(1, -1)).transpose(1, -1)
        inputs = self.dropout(inputs)

        inputs = self.conv_2(inputs * padding_mask)
        inputs = torch.relu(inputs)
        inputs = self.norm_2(inputs.transpose(1, -1)).transpose(1, -1)
        inputs = self.dropout(inputs)

        inputs = self.proj(inputs * padding_mask)
        return inputs * padding_mask


class VitsAttention(nn.Module):
    """Multi-headed attention with relative positional representation."""

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.window_size = config.window_size

        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.embed_dim}"
                f" and `num_attention_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)

        if self.window_size:
            self.emb_rel_k = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)
            self.emb_rel_v = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, src_len)
            relative_logits = torch.matmul(query_states, key_relative_embeddings.transpose(-2, -1))
            rel_pos_bias = self._relative_position_to_absolute_position(relative_logits)
            attn_weights += rel_pos_bias

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
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if self.window_size is not None:
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, src_len)
            relative_weights = self._absolute_position_to_relative_position(attn_probs)
            rel_pos_bias = torch.matmul(relative_weights, value_relative_embeddings)
            attn_output += rel_pos_bias

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        if pad_length > 0:
            relative_embeddings = nn.functional.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])

        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        return relative_embeddings[:, slice_start_position:slice_end_position]

    def _relative_position_to_absolute_position(self, x):
        batch_heads, length, _ = x.size()

        # Concat columns of pad to shift from relative to absolute indexing.
        x = nn.functional.pad(x, [0, 1, 0, 0, 0, 0])

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch_heads, length * 2 * length])
        x_flat = nn.functional.pad(x_flat, [0, length - 1, 0, 0])

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch_heads, length + 1, 2 * length - 1])
        x_final = x_final[:, :length, length - 1 :]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch_heads, length, _ = x.size()

        # Pad along column
        x = nn.functional.pad(x, [0, length - 1, 0, 0, 0, 0])
        x_flat = x.view([batch_heads, length * (2 * length - 1)])

        # Add 0's in the beginning that will skew the elements after reshape
        x_flat = nn.functional.pad(x_flat, [length, 0, 0, 0])
        x_final = x_flat.view([batch_heads, length, 2 * length])[:, :, 1:]
        return x_final


class VitsFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_1 = nn.Conv1d(config.hidden_size, config.ffn_dim, config.ffn_kernel_size)
        self.conv_2 = nn.Conv1d(config.ffn_dim, config.hidden_size, config.ffn_kernel_size)
        self.dropout = nn.Dropout(config.activation_dropout)

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        if config.ffn_kernel_size > 1:
            pad_left = (config.ffn_kernel_size - 1) // 2
            pad_right = config.ffn_kernel_size // 2
            self.padding = [pad_left, pad_right, 0, 0, 0, 0]
        else:
            self.padding = None

    def forward(self, hidden_states, padding_mask):
        hidden_states = hidden_states.permute(0, 2, 1)
        padding_mask = padding_mask.permute(0, 2, 1)

        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_2(hidden_states)
        hidden_states = hidden_states * padding_mask

        hidden_states = hidden_states.permute(0, 2, 1)
        return hidden_states


class VitsEncoderLayer(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.attention = VitsAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = VitsFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states, attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.feed_forward(hidden_states, padding_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.final_layer_norm(residual + hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class VitsEncoder(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([VitsEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.layerdrop = config.layerdrop

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        hidden_states = hidden_states * padding_mask

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = self.training and (dropout_probability < self.layerdrop)
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        padding_mask,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = hidden_states * padding_mask

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class VitsTextEncoder(nn.Module):
    """
    Transformer encoder that uses relative positional representation instead of absolute positional encoding.
    """

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.encoder = VitsEncoder(config)
        self.project = nn.Conv1d(config.hidden_size, config.flow_size * 2, kernel_size=1)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], VitsTextEncoderOutput]:
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)

        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state

        stats = self.project(last_hidden_state.transpose(1, 2)).transpose(1, 2) * padding_mask
        prior_means, prior_log_variances = torch.split(stats, self.config.flow_size, dim=2)

        if not return_dict:
            outputs = (last_hidden_state, prior_means, prior_log_variances) + encoder_outputs[1:]
            return outputs

        return VitsTextEncoderOutput(
            last_hidden_state=last_hidden_state,
            prior_means=prior_means,
            prior_log_variances=prior_log_variances,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class VitsPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VitsConfig
    base_model_prefix = "vits"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


VITS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VitsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


VITS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        speaker_id (`int`, *optional*):
            Which speaker embedding to use. Only used for multispeaker models.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The complete VITS model, for text-to-speech synthesis.",
    VITS_START_DOCSTRING,
)
class VitsModel(VitsPreTrainedModel):
    def __init__(self, config: VitsConfig):
        super().__init__(config)
        self.config = config
        self.text_encoder = VitsTextEncoder(config)
        self.flow = VitsResidualCouplingBlock(config)
        self.decoder = VitsHifiGan(config)

        if config.use_stochastic_duration_prediction:
            self.duration_predictor = VitsStochasticDurationPredictor(config)
        else:
            self.duration_predictor = VitsDurationPredictor(config)

        if config.num_speakers > 1:
            self.embed_speaker = nn.Embedding(config.num_speakers, config.speaker_embedding_size)

        # This is used only for training.
        self.posterior_encoder = VitsPosteriorEncoder(config)

        # These parameters control the synthesised speech properties
        self.speaking_rate = config.speaking_rate
        self.noise_scale = config.noise_scale
        self.noise_scale_duration = config.noise_scale_duration

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.text_encoder

    @add_start_docstrings_to_model_forward(VITS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VitsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[Any], VitsModelOutput]:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`, *optional*):
            Float values of target spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss
            computation.

        Returns:

        Example:

        ```python
        >>> from transformers import VitsTokenizer, VitsModel, set_seed
        >>> import torch

        >>> tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
        >>> model = VitsModel.from_pretrained("facebook/mms-tts-eng")

        >>> inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

        >>> set_seed(555)  # make deterministic

        >>> with torch.no_grad():
        ...     outputs = model(inputs["input_ids"])
        >>> outputs.waveform.shape
        torch.Size([1, 45824])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            raise NotImplementedError("Training of VITS is not supported yet.")

        if attention_mask is not None:
            input_padding_mask = attention_mask.unsqueeze(-1).float()
        else:
            input_padding_mask = torch.ones_like(input_ids).unsqueeze(-1).float()

        if self.config.num_speakers > 1 and speaker_id is not None:
            if not 0 <= speaker_id < self.config.num_speakers:
                raise ValueError(f"Set `speaker_id` in the range 0-{self.config.num_speakers - 1}.")
            if isinstance(speaker_id, int):
                speaker_id = torch.full(size=(1,), fill_value=speaker_id, device=self.device)
            speaker_embeddings = self.embed_speaker(speaker_id).unsqueeze(-1)
        else:
            speaker_embeddings = None

        text_encoder_output = self.text_encoder(
            input_ids=input_ids,
            padding_mask=input_padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = text_encoder_output[0] if not return_dict else text_encoder_output.last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        input_padding_mask = input_padding_mask.transpose(1, 2)
        prior_means = text_encoder_output[1] if not return_dict else text_encoder_output.prior_means
        prior_log_variances = text_encoder_output[2] if not return_dict else text_encoder_output.prior_log_variances

        if self.config.use_stochastic_duration_prediction:
            log_duration = self.duration_predictor(
                hidden_states,
                input_padding_mask,
                speaker_embeddings,
                reverse=True,
                noise_scale=self.noise_scale_duration,
            )
        else:
            log_duration = self.duration_predictor(hidden_states, input_padding_mask, speaker_embeddings)

        length_scale = 1.0 / self.speaking_rate
        duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

        # Create a padding mask for the output lengths of shape (batch, 1, max_output_length)
        indices = torch.arange(predicted_lengths.max(), dtype=predicted_lengths.dtype, device=predicted_lengths.device)
        output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
        output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

        # Reconstruct an attention tensor of shape (batch, 1, out_length, in_length)
        attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
        batch_size, _, output_length, input_length = attn_mask.shape
        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        indices = torch.arange(output_length, dtype=duration.dtype, device=duration.device)
        valid_indices = indices.unsqueeze(0) < cum_duration
        valid_indices = valid_indices.to(attn_mask.dtype).view(batch_size, input_length, output_length)
        padded_indices = valid_indices - nn.functional.pad(valid_indices, [0, 0, 1, 0, 0, 0])[:, :-1]
        attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

        # Expand prior distribution
        prior_means = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)
        prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances).transpose(1, 2)

        prior_latents = prior_means + torch.randn_like(prior_means) * torch.exp(prior_log_variances) * self.noise_scale
        latents = self.flow(prior_latents, output_padding_mask, speaker_embeddings, reverse=True)

        spectrogram = latents * output_padding_mask
        waveform = self.decoder(spectrogram, speaker_embeddings)
        waveform = waveform.squeeze(1)
        sequence_lengths = predicted_lengths * np.prod(self.config.upsample_rates)

        if not return_dict:
            outputs = (waveform, sequence_lengths, spectrogram) + text_encoder_output[3:]
            return outputs

        return VitsModelOutput(
            waveform=waveform,
            sequence_lengths=sequence_lengths,
            spectrogram=spectrogram,
            hidden_states=text_encoder_output.hidden_states,
            attentions=text_encoder_output.attentions,
        )


__all__ = ["VitsModel", "VitsPreTrainedModel"]

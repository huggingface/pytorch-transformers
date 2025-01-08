# coding=utf-8
# Copyright 2023 The Espnet authors, IMS Toucan authors, and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch FastSpeech2Conformer model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
    FastSpeech2ConformerConfig,
    FastSpeech2ConformerHifiGanConfig,
    FastSpeech2ConformerWithHifiGanConfig,
)


logger = logging.get_logger(__name__)


@dataclass
class FastSpeech2ConformerModelOutput(ModelOutput):
    """
    Output type of [`FastSpeech2ConformerModel`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Spectrogram generation loss.
        spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`):
            The predicted spectrogram.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        duration_outputs (`torch.LongTensor` of shape `(batch_size, max_text_length + 1)`, *optional*):
            Outputs of the duration predictor.
        pitch_outputs (`torch.FloatTensor` of shape `(batch_size, max_text_length + 1, 1)`, *optional*):
            Outputs of the pitch predictor.
        energy_outputs (`torch.FloatTensor` of shape `(batch_size, max_text_length + 1, 1)`, *optional*):
            Outputs of the energy predictor.

    """

    loss: Optional[torch.FloatTensor] = None
    spectrogram: torch.FloatTensor = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    duration_outputs: torch.LongTensor = None
    pitch_outputs: torch.FloatTensor = None
    energy_outputs: torch.FloatTensor = None


@dataclass
class FastSpeech2ConformerWithHifiGanOutput(FastSpeech2ConformerModelOutput):
    """
    Output type of [`FastSpeech2ConformerWithHifiGan`].

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, audio_length)`):
            Speech output as a result of passing the predicted mel spectrogram through the vocoder.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Spectrogram generation loss.
        spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`):
            The predicted spectrogram.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        duration_outputs (`torch.LongTensor` of shape `(batch_size, max_text_length + 1)`, *optional*):
            Outputs of the duration predictor.
        pitch_outputs (`torch.FloatTensor` of shape `(batch_size, max_text_length + 1, 1)`, *optional*):
            Outputs of the pitch predictor.
        energy_outputs (`torch.FloatTensor` of shape `(batch_size, max_text_length + 1, 1)`, *optional*):
            Outputs of the energy predictor.
    """

    waveform: torch.FloatTensor = None


_CONFIG_FOR_DOC = "FastSpeech2ConformerConfig"

FASTSPEECH2_CONFORMER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FastSpeech2ConformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


HIFIGAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FastSpeech2ConformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FASTSPEECH2_CONFORMER_WITH_HIFIGAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FastSpeech2ConformerWithHifiGanConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


def length_regulator(encoded_embeddings, duration_labels, speaking_speed=1.0):
    """
    Length regulator for feed-forward Transformer.

    This is the length regulator module described in `FastSpeech: Fast, Robust and Controllable Text to Speech`
    https://arxiv.org/pdf/1905.09263.pdf. The length regulator expands char or phoneme-level embedding features to
    frame-level by repeating each feature based on the corresponding predicted durations.

    Args:
        encoded_embeddings (`torch.Tensor` of shape `(batch_size, max_text_length, embedding_dim)`):
            Batch of sequences of char or phoneme embeddings.
        duration_labels (`torch.LongTensor` of shape `(batch_size, time)`):
            Batch of durations of each frame.
        speaking_speed (`float`, *optional*, defaults to 1.0):
            Value to control speed of speech.

    Returns:
        `torch.Tensor`:
            Replicated input tensor based on durations (batch_size, time*, embedding_dim).
    """

    if speaking_speed <= 0:
        raise ValueError("`speaking_speed` must be greater than 0.")
    elif speaking_speed != 1.0:
        duration_labels = torch.round(duration_labels.float() * speaking_speed).long()

    if duration_labels.sum() == 0:
        duration_labels[duration_labels.sum(dim=1).eq(0)] = 1

    # Calculate the maximum length needed
    max_len = torch.sum(duration_labels, dim=1).max()

    # Create a padded tensor to hold the results
    hidden_states = torch.zeros(
        (encoded_embeddings.size(0), max_len, encoded_embeddings.size(2)),
        dtype=torch.float,
        device=encoded_embeddings.device,
    )

    # Loop through the batch and fill in the data
    for i, (encoded_embedding, target_duration) in enumerate(zip(encoded_embeddings, duration_labels)):
        repeated = torch.repeat_interleave(encoded_embedding, target_duration, dim=0)
        hidden_states[i, : repeated.size(0)] = repeated

    return hidden_states


class FastSpeech2ConformerDurationPredictor(nn.Module):
    """
    Duration predictor module.

    This is a module of duration predictor described in the paper 'FastSpeech: Fast, Robust and Controllable Text to
    Speech' https://arxiv.org/pdf/1905.09263.pdf The duration predictor predicts a duration of each frame in log domain
    from the hidden embeddings of encoder.

    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`, the
        outputs are calculated in log domain but in `inference`, those are calculated in linear domain.

    """

    def __init__(self, config: FastSpeech2ConformerConfig):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.log_domain_offset = 1.0

        for layer_idx in range(config.duration_predictor_layers):
            num_chans = config.duration_predictor_channels
            input_channels = config.hidden_size if layer_idx == 0 else num_chans
            layer = FastSpeech2ConformerPredictorLayer(
                input_channels,
                num_chans,
                config.duration_predictor_kernel_size,
                config.duration_predictor_dropout_rate,
            )
            self.conv_layers.append(layer)
        self.linear = nn.Linear(config.duration_predictor_channels, 1)

    def forward(self, encoder_hidden_states):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                Batch of input sequences.
            padding_masks (`torch.ByteTensor` of shape `(batch_size, max_text_length)`, *optional*):
                Batch of masks indicating padded part.

        Returns:
            `torch.Tensor`: Batch of predicted durations in log domain `(batch_size, max_text_length)`.

        """
        # (batch_size, input_dim, max_text_length)
        hidden_states = encoder_hidden_states.transpose(1, -1)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)

        # NOTE: calculate in log domain, (batch_size, max_text_length)
        hidden_states = self.linear(hidden_states.transpose(1, -1)).squeeze(-1)

        if not self.training:
            # NOTE: calculate in linear domain
            hidden_states = torch.clamp(torch.round(hidden_states.exp() - self.log_domain_offset), min=0).long()

        return hidden_states


# Copied from transformers.models.speecht5.modeling_speecht5.SpeechT5BatchNormConvLayer
class FastSpeech2ConformerBatchNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()

        if layer_id == 0:
            in_conv_dim = config.num_mel_bins
        else:
            in_conv_dim = config.speech_decoder_postnet_units

        if layer_id == config.speech_decoder_postnet_layers - 1:
            out_conv_dim = config.num_mel_bins
        else:
            out_conv_dim = config.speech_decoder_postnet_units

        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=config.speech_decoder_postnet_kernel,
            stride=1,
            padding=(config.speech_decoder_postnet_kernel - 1) // 2,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(out_conv_dim)

        if layer_id < config.speech_decoder_postnet_layers - 1:
            self.activation = nn.Tanh()
        else:
            self.activation = None

        self.dropout = nn.Dropout(config.speech_decoder_postnet_dropout)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class FastSpeech2ConformerSpeechDecoderPostnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feat_out = nn.Linear(config.hidden_size, config.num_mel_bins * config.reduction_factor)
        self.layers = nn.ModuleList(
            [FastSpeech2ConformerBatchNormConvLayer(config, i) for i in range(config.speech_decoder_postnet_layers)]
        )

    def forward(self, hidden_states: torch.Tensor):
        outputs_before_postnet = self.feat_out(hidden_states).view(hidden_states.size(0), -1, self.config.num_mel_bins)
        layer_output = outputs_before_postnet.transpose(1, 2)
        for layer in self.layers:
            layer_output = layer(layer_output)
        outputs_after_postnet = outputs_before_postnet + layer_output.transpose(1, 2)
        return outputs_before_postnet, outputs_after_postnet


class FastSpeech2ConformerPredictorLayer(nn.Module):
    def __init__(self, input_channels, num_chans, kernel_size, dropout_rate):
        super().__init__()
        self.conv = nn.Conv1d(
            input_channels,
            num_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(num_chans)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)

        # Perform layer norm on dimension 1
        hidden_states = hidden_states.transpose(1, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, -1)

        hidden_states = self.dropout(hidden_states)

        return hidden_states


class FastSpeech2ConformerVariancePredictor(nn.Module):
    def __init__(
        self,
        config: FastSpeech2ConformerConfig,
        num_layers=2,
        num_chans=384,
        kernel_size=3,
        dropout_rate=0.5,
    ):
        """
        Initilize variance predictor module.

        Args:
            input_dim (`int`): Input dimension.
            num_layers (`int`, *optional*, defaults to 2): Number of convolutional layers.
            num_chans (`int`, *optional*, defaults to 384): Number of channels of convolutional layers.
            kernel_size (`int`, *optional*, defaults to 3): Kernel size of convolutional layers.
            dropout_rate (`float`, *optional*, defaults to 0.5): Dropout rate.
        """
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for idx in range(num_layers):
            input_channels = config.hidden_size if idx == 0 else num_chans
            layer = FastSpeech2ConformerPredictorLayer(input_channels, num_chans, kernel_size, dropout_rate)
            self.conv_layers.append(layer)
        self.linear = nn.Linear(num_chans, 1)

    def forward(self, encoder_hidden_states, padding_masks=None):
        """
        Calculate forward propagation.

        Args:
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                Batch of input sequences.
            padding_masks (`torch.ByteTensor` of shape `(batch_size, max_text_length)`, *optional*):
                Batch of masks indicating padded part.

        Returns:
            Tensor: Batch of predicted sequences `(batch_size, max_text_length, 1)`.
        """
        # (batch_size, input_dim, max_text_length)
        hidden_states = encoder_hidden_states.transpose(1, -1)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.linear(hidden_states.transpose(1, 2))

        if padding_masks is not None:
            hidden_states = hidden_states.masked_fill(padding_masks, 0.0)

        return hidden_states


class FastSpeech2ConformerVarianceEmbedding(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=384,
        kernel_size=1,
        padding=0,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class FastSpeech2ConformerAttention(nn.Module):
    """
    Multi-Head attention layer with relative position encoding. Details can be found in
    https://github.com/espnet/espnet/pull/2816. Paper: https://arxiv.org/abs/1901.02860.
    """

    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        """Construct an FastSpeech2ConformerAttention object."""
        super().__init__()
        # We assume d_v always equals dim_key
        self.num_heads = module_config["num_attention_heads"]
        self.hidden_size = config.hidden_size
        self.dim_key = self.hidden_size // self.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=module_config["attention_dropout_rate"])

        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))

    def shift_relative_position_tensor(self, pos_tensor):
        """
        Args:
            pos_tensor (torch.Tensor of shape (batch_size, head, time1, 2*time1-1)): Input tensor.
        """
        zero_pad = torch.zeros((*pos_tensor.size()[:3], 1), device=pos_tensor.device, dtype=pos_tensor.dtype)
        pos_tensor_padded = torch.cat([zero_pad, pos_tensor], dim=-1)

        pos_tensor_padded = pos_tensor_padded.view(*pos_tensor.size()[:2], pos_tensor.size(3) + 1, pos_tensor.size(2))
        # only keep the positions from 0 to time2
        pos_tensor = pos_tensor_padded[:, :, 1:].view_as(pos_tensor)[:, :, :, : pos_tensor.size(-1) // 2 + 1]

        return pos_tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time2, size)`): Values of the hidden states
            attention_mask (`torch.Tensor` of shape `(batch, time1, time2)`): Mask tensor.
            pos_emb (`torch.Tensor` of shape `(batch, 2*time1-1, size)`): Positional embedding tensor.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time1, d_model)`.
        """
        bsz, q_len, _ = hidden_states.size()
        query_states = self.linear_q(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
        key_states = self.linear_k(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)
        value_states = self.linear_v(hidden_states).view(bsz, -1, self.num_heads, self.head_dim)

        bsz_pos = pos_emb.size(0)
        pos_encoding = self.linear_pos(pos_emb).view(bsz_pos, -1, self.num_heads, self.head_dim)

        # (batch_size, head, time1, dim_key)
        query_with_bias_u = (query_states + self.pos_bias_u).transpose(1, 2)
        # (batch_size, head, time1, dim_key)
        query_with_bias_v = (query_states + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch_size, head, time1, time2)
        matrix_ac = torch.matmul(query_with_bias_u, key_states.permute(0, 2, 3, 1))

        # compute matrix b and matrix d
        # (batch_size, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(query_with_bias_v, pos_encoding.permute(0, 2, 3, 1))
        matrix_bd = self.shift_relative_position_tensor(matrix_bd)

        # (batch_size, head, time1, time2)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.dim_key)

        # Forward attention
        if attention_mask is not None:
            expected_size = (bsz, 1, q_len)
            if attention_mask.size() != expected_size:
                raise ValueError(f"Attention mask should be of size {expected_size}, but is {attention_mask.size()}")
            attention_mask = attention_mask.unsqueeze(1).eq(0)
            min_value = float(torch.finfo(scores.dtype).min)
            scores = scores.masked_fill(attention_mask, min_value)
            attn_weights = torch.softmax(scores, dim=-1).masked_fill(attention_mask, 0.0)
        else:
            attn_weights = torch.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states.transpose(1, 2))
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        attn_output = self.linear_out(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class FastSpeech2ConformerConvolutionModule(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        super().__init__()
        # kernel_size should be an odd number for 'SAME' padding
        channels = config.hidden_size
        kernel_size = module_config["kernel_size"]
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=True
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states):
        """
        Compute convolution module.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time, channels)`): Input tensor.

        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time, channels)`.

        """
        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism, (batch_size, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # (batch_size, channel, dim)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states * torch.sigmoid(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)

        return hidden_states.transpose(1, 2)


class FastSpeech2ConformerEncoderLayer(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        super().__init__()

        # self-attention module definition
        self.self_attn = FastSpeech2ConformerAttention(config, module_config)

        # feed-forward module definition
        self.feed_forward = FastSpeech2ConformerMultiLayeredConv1d(config, module_config)

        self.macaron_style = config.use_macaron_style_in_conformer
        if self.macaron_style:
            self.feed_forward_macaron = FastSpeech2ConformerMultiLayeredConv1d(config, module_config)
            self.ff_macaron_layer_norm = nn.LayerNorm(config.hidden_size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        # convolution module definition
        self.use_cnn_module = config.use_cnn_in_conformer
        if self.use_cnn_module:
            self.conv_module = FastSpeech2ConformerConvolutionModule(config, module_config)
            self.conv_layer_norm = nn.LayerNorm(config.hidden_size)
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)

        self.ff_layer_norm = nn.LayerNorm(config.hidden_size)

        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        self.dropout = nn.Dropout(module_config["dropout_rate"])
        self.size = config.hidden_size
        self.normalize_before = module_config["normalize_before"]
        self.concat_after = module_config["concat_after"]
        if self.concat_after:
            self.concat_linear = nn.Linear(config.hidden_size + config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = False,
    ):
        """
        Compute encoded features.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time, size)`): Input tensor.
            pos_emb (`torch.Tensor` of shape `(1, time, size)`): Positional embeddings tensor.
            attention_mask (`torch.Tensor` of shape `(batch, time)`): Attention mask tensor for the input.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time, size)`.

        """
        # whether to use macaron style
        if self.macaron_style:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.ff_macaron_layer_norm(hidden_states)
            hidden_states = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(hidden_states))
            if not self.normalize_before:
                hidden_states = self.ff_macaron_layer_norm(hidden_states)

        # multi-headed self-attention module
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        attention_output, attention_scores = self.self_attn(
            hidden_states, attention_mask=attention_mask, pos_emb=pos_emb, output_attentions=output_attentions
        )

        if self.concat_after:
            x_concat = torch.cat((hidden_states, attention_output), dim=-1)
            hidden_states = self.concat_linear(x_concat)
            hidden_states = residual + hidden_states
        else:
            hidden_states = self.dropout(attention_output)
            hidden_states = residual + hidden_states
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # convolution module
        if self.use_cnn_module:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.conv_layer_norm(hidden_states)
            hidden_states = self.conv_module(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states
            if not self.normalize_before:
                hidden_states = self.conv_layer_norm(hidden_states)

        # feed forward module
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.ff_layer_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + self.ff_scale * hidden_states
        if not self.normalize_before:
            hidden_states = self.ff_layer_norm(hidden_states)

        if self.conv_module is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_scores,)

        return outputs


class FastSpeech2ConformerMultiLayeredConv1d(nn.Module):
    """
    Multi-layered conv1d for Transformer block.

    This is a module of multi-layered conv1d designed to replace positionwise feed-forward network in Transformer
    block, which is introduced in 'FastSpeech: Fast, Robust and Controllable Text to Speech'
    https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        """
        Initialize FastSpeech2ConformerMultiLayeredConv1d module.

        Args:
            input_channels (`int`): Number of input channels.
            hidden_channels (`int`): Number of hidden channels.
            kernel_size (`int`): Kernel size of conv1d.
            dropout_rate (`float`): Dropout rate.
        """
        super().__init__()
        input_channels = config.hidden_size
        hidden_channels = module_config["linear_units"]
        kernel_size = config.positionwise_conv_kernel_size
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(hidden_channels, input_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = nn.Dropout(module_config["dropout_rate"])

    def forward(self, hidden_states):
        """
        Calculate forward propagation.

        Args:
            hidden_states (torch.Tensor): Batch of input tensors (batch_size, time, input_channels).

        Returns:
            torch.Tensor: Batch of output tensors (batch_size, time, hidden_channels).
        """
        hidden_states = hidden_states.transpose(-1, 1)
        hidden_states = self.conv1(hidden_states)
        hidden_states = torch.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.transpose(-1, 1)
        return hidden_states


class FastSpeech2ConformerRelPositionalEncoding(nn.Module):
    """
    Args:
    Relative positional encoding module (new implementation). Details can be found in
    https://github.com/espnet/espnet/pull/2816. See : Appendix Batch in https://arxiv.org/abs/1901.02860
        config (`FastSpeech2ConformerConfig`):
            FastSpeech2ConformerConfig instance.
        module_config (`dict`):
            Dictionary containing the encoder or decoder module configuration from the `FastSpeech2ConformerConfig`.
    """

    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        """
        Construct an PositionalEncoding object.
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.input_scale = math.sqrt(self.embed_dim)
        self.dropout = nn.Dropout(p=module_config["positional_dropout_rate"])
        self.pos_enc = None
        self.max_len = 5000
        self.extend_pos_enc(torch.tensor(0.0).expand(1, self.max_len))

    def extend_pos_enc(self, x):
        """Reset the positional encodings."""
        if self.pos_enc is not None:
            # self.pos_enc contains both positive and negative parts
            # the length of self.pos_enc is 2 * input_len - 1
            if self.pos_enc.size(1) >= x.size(1) * 2 - 1:
                if self.pos_enc.dtype != x.dtype or self.pos_enc.device != x.device:
                    self.pos_enc = self.pos_enc.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pos_enc_positive = torch.zeros(x.size(1), self.embed_dim)
        pos_enc_negative = torch.zeros(x.size(1), self.embed_dim)
        position = torch.arange(0, x.size(1), dtype=torch.int64).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.int64).float() * -(math.log(10000.0) / self.embed_dim)
        )
        pos_enc_positive[:, 0::2] = torch.sin(position * div_term)
        pos_enc_positive[:, 1::2] = torch.cos(position * div_term)
        pos_enc_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pos_enc_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pos_enc_positive = torch.flip(pos_enc_positive, [0]).unsqueeze(0)
        pos_enc_negative = pos_enc_negative[1:].unsqueeze(0)
        pos_enc = torch.cat([pos_enc_positive, pos_enc_negative], dim=1)
        self.pos_enc = pos_enc.to(device=x.device, dtype=x.dtype)

    def forward(self, feature_representation):
        """
        Args:
            feature_representation (`torch.Tensor` of shape (batch_size, time, `*`)):
                Input tensor.

        Returns:
            `torch.Tensor`: Encoded tensor (batch_size, time, `*`).
        """
        self.extend_pos_enc(feature_representation)
        hidden_states = feature_representation * self.input_scale
        center_idx = self.pos_enc.size(1) // 2
        pos_emb = self.pos_enc[:, center_idx - hidden_states.size(1) + 1 : center_idx + hidden_states.size(1)]
        return self.dropout(hidden_states), self.dropout(pos_emb)


class FastSpeech2ConformerEncoder(nn.Module):
    """
    FastSpeech2ConformerEncoder encoder module.

    Args:
        config (`FastSpeech2ConformerConfig`):
            FastSpeech2ConformerConfig instance.
        module_config (`dict`):
            Dictionary containing the encoder or decoder module configuration from the `FastSpeech2ConformerConfig`.
        use_encoder_input_layer (`bool`, *optional*, defaults to `False`):
            Input layer type.
    """

    def __init__(
        self,
        config: FastSpeech2ConformerConfig,
        module_config,
        use_encoder_input_layer=False,
    ):
        super().__init__()

        self.embed = None
        if use_encoder_input_layer:
            self.embed = nn.Embedding(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=0
            )

        self.pos_enc = FastSpeech2ConformerRelPositionalEncoding(config, module_config)

        self.conformer_layers = nn.ModuleList(
            [FastSpeech2ConformerEncoderLayer(config, module_config) for _ in range(module_config["layers"])]
        )

    def forward(
        self,
        input_tensor: torch.LongTensor,
        attention_mask: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        Returns:
            `torch.Tensor`:
                Output tensor of shape `(batch, time, attention_dim)`.
        """
        feature_representation = input_tensor
        if self.embed is not None:
            feature_representation = self.embed(feature_representation)

        hidden_states, pos_emb = self.pos_enc(feature_representation)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for conformer_layer in self.conformer_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = conformer_layer(hidden_states, pos_emb, attention_mask, output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )


class FastSpeech2ConformerLoss(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig):
        super().__init__()

        use_masking = config.use_masking
        use_weighted_masking = config.use_weighted_masking

        if use_masking and use_weighted_masking:
            raise ValueError("Either use_masking or use_weighted_masking can be True, but not both.")

        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = nn.L1Loss(reduction=reduction)
        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.duration_criterion = nn.MSELoss(reduction=reduction)
        self.log_domain_offset = 1.0

    def forward(
        self,
        outputs_after_postnet,
        outputs_before_postnet,
        duration_outputs,
        pitch_outputs,
        energy_outputs,
        spectrogram_labels,
        duration_labels,
        pitch_labels,
        energy_labels,
        duration_mask,
        spectrogram_mask,
    ):
        """
        Args:
            outputs_after_postnet (`torch.Tensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`):
                Batch of outputs after postnet.
            outputs_before_postnet (`torch.Tensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`):
                Batch of outputs before postnet.
            duration_outputs (`torch.LongTensor` of shape `(batch_size, max_text_length)`):
                Batch of outputs of duration predictor.
            pitch_outputs (`torch.Tensor` of shape `(batch_size, max_text_length, 1)`):
                Batch of outputs of pitch predictor.
            energy_outputs (`torch.Tensor` of shape `(batch_size, max_text_length, 1)`):
                Batch of outputs of energy predictor.
            spectrogram_labels (`torch.Tensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`):
                Batch of target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, max_text_length)`): Batch of durations.
            pitch_labels (`torch.Tensor` of shape `(batch_size, max_text_length, 1)`):
                Batch of target token-averaged pitch.
            energy_labels (`torch.Tensor` of shape `(batch_size, max_text_length, 1)`):
                Batch of target token-averaged energy.
            duration_mask (`torch.LongTensor`):
                Mask used to discern which values the duration loss should be calculated for.
            spectrogram_mask (`torch.LongTensor`):
                Mask used to discern which values the spectrogam loss should be calculated for.

        Returns:
            `tuple(torch.FloatTensor)`: Tuple of tensors containing, in order, the L1 loss value, duration predictor
            loss value, pitch predictor loss value, and energy predictor loss value.

        """
        pitch_and_energy_masks = duration_mask.unsqueeze(-1)

        # apply mask to remove padded part
        if self.use_masking:
            outputs_before_postnet = outputs_before_postnet.masked_select(spectrogram_mask)
            if outputs_after_postnet is not None:
                outputs_after_postnet = outputs_after_postnet.masked_select(spectrogram_mask)
            spectrogram_labels = spectrogram_labels.masked_select(spectrogram_mask)
            duration_outputs = duration_outputs.masked_select(duration_mask)
            duration_labels = duration_labels.masked_select(duration_mask)
            pitch_outputs = pitch_outputs.masked_select(pitch_and_energy_masks)
            energy_outputs = energy_outputs.masked_select(pitch_and_energy_masks)
            pitch_labels = pitch_labels.masked_select(pitch_and_energy_masks)
            energy_labels = energy_labels.masked_select(pitch_and_energy_masks)

        # calculate loss
        l1_loss = self.l1_criterion(outputs_before_postnet, spectrogram_labels)
        if outputs_after_postnet is not None:
            l1_loss = l1_loss + self.l1_criterion(outputs_after_postnet, spectrogram_labels)
        duration_labels = torch.log(duration_labels.float() + self.log_domain_offset)
        duration_loss = self.duration_criterion(duration_outputs, duration_labels)
        pitch_loss = self.mse_criterion(pitch_outputs, pitch_labels)
        energy_loss = self.mse_criterion(energy_outputs, energy_labels)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            spectrogram_mask = nn.functional.pad(
                spectrogram_mask.transpose(1, 2),
                [0, spectrogram_labels.size(1) - spectrogram_mask.size(1), 0, 0, 0, 0],
                value=False,
            ).transpose(1, 2)

            out_weights = spectrogram_mask.float() / spectrogram_mask.sum(dim=1, keepdim=True).float()
            out_weights /= spectrogram_labels.size(0) * spectrogram_labels.size(2)
            duration_weights = duration_mask.float() / duration_mask.sum(dim=1, keepdim=True).float()
            duration_weights /= duration_labels.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(spectrogram_mask).sum()
            duration_loss = duration_loss.mul(duration_weights).masked_select(duration_mask).sum()
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_and_energy_masks).sum()
            energy_loss = energy_loss.mul(pitch_weights).masked_select(pitch_and_energy_masks).sum()

        return l1_loss + duration_loss + pitch_loss + energy_loss


class FastSpeech2ConformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FastSpeech2ConformerConfig
    base_model_prefix = "fastspeech2_conformer"

    main_input_name = "input_ids"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                key = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-key, b=key)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_()
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, FastSpeech2ConformerAttention):
            nn.init.xavier_uniform_(module.pos_bias_u)
            nn.init.xavier_uniform_(module.pos_bias_v)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FastSpeech2ConformerEncoder):
            module.gradient_checkpointing = value


@add_start_docstrings(
    """FastSpeech2Conformer Model.""",
    FASTSPEECH2_CONFORMER_START_DOCSTRING,
)
class FastSpeech2ConformerModel(FastSpeech2ConformerPreTrainedModel):
    """
    FastSpeech 2 module.

    This is a module of FastSpeech 2 described in 'FastSpeech 2: Fast and High-Quality End-to-End Text to Speech'
    https://arxiv.org/abs/2006.04558. Instead of quantized pitch and energy, we use token-averaged value introduced in
    FastPitch: Parallel Text-to-speech with Pitch Prediction. The encoder and decoder are Conformers instead of regular
    Transformers.
    """

    def __init__(self, config: FastSpeech2ConformerConfig):
        super().__init__(config)
        self.config = config

        # store hyperparameters
        self.vocab_size = config.vocab_size
        self.num_mel_bins = config.num_mel_bins
        self.hidden_size = config.hidden_size
        self.reduction_factor = config.reduction_factor
        self.stop_gradient_from_pitch_predictor = config.stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = config.stop_gradient_from_energy_predictor

        self.multilingual_model = config.num_languages is not None and config.num_languages > 1
        if self.multilingual_model:
            self.language_id_embedding = torch.nn.Embedding(config.num_languages, self.hidden_size)

        self.multispeaker_model = config.num_speakers is not None and config.num_speakers > 1
        if self.multispeaker_model:
            self.speaker_id_embedding = torch.nn.Embedding(config.num_speakers, config.hidden_size)

        self.speaker_embed_dim = config.speaker_embed_dim
        if self.speaker_embed_dim:
            self.projection = nn.Linear(config.hidden_size + self.speaker_embed_dim, config.hidden_size)

        self.encoder = FastSpeech2ConformerEncoder(config, config.encoder_config, use_encoder_input_layer=True)

        self.duration_predictor = FastSpeech2ConformerDurationPredictor(config)

        self.pitch_predictor = FastSpeech2ConformerVariancePredictor(
            config,
            num_layers=config.pitch_predictor_layers,
            num_chans=config.pitch_predictor_channels,
            kernel_size=config.pitch_predictor_kernel_size,
            dropout_rate=config.pitch_predictor_dropout,
        )
        # continuous pitch + FastPitch style avg
        self.pitch_embed = FastSpeech2ConformerVarianceEmbedding(
            out_channels=self.hidden_size,
            kernel_size=config.pitch_embed_kernel_size,
            padding=(config.pitch_embed_kernel_size - 1) // 2,
            dropout_rate=config.pitch_embed_dropout,
        )

        self.energy_predictor = FastSpeech2ConformerVariancePredictor(
            config,
            num_layers=config.energy_predictor_layers,
            num_chans=config.energy_predictor_channels,
            kernel_size=config.energy_predictor_kernel_size,
            dropout_rate=config.energy_predictor_dropout,
        )
        # continuous energy + FastPitch style avg
        self.energy_embed = FastSpeech2ConformerVarianceEmbedding(
            out_channels=self.hidden_size,
            kernel_size=config.energy_embed_kernel_size,
            padding=(config.energy_embed_kernel_size - 1) // 2,
            dropout_rate=config.energy_embed_dropout,
        )

        # The decoder is an encoder
        self.decoder = FastSpeech2ConformerEncoder(config, config.decoder_config, use_encoder_input_layer=False)

        self.speech_decoder_postnet = FastSpeech2ConformerSpeechDecoderPostnet(config)

        self.criterion = FastSpeech2ConformerLoss(config)

        self.post_init()

    @replace_return_docstrings(output_type=FastSpeech2ConformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        spectrogram_labels: Optional[torch.FloatTensor] = None,
        duration_labels: Optional[torch.LongTensor] = None,
        pitch_labels: Optional[torch.FloatTensor] = None,
        energy_labels: Optional[torch.FloatTensor] = None,
        speaker_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        speaker_embedding: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, FastSpeech2ConformerModelOutput]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`: 0 for tokens that are **masked**, 1 for tokens that are **not masked**.
            spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`, *optional*, defaults to `None`):
                Batch of padded target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length + 1)`, *optional*, defaults to `None`):
                Batch of padded durations.
            pitch_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged pitch.
            energy_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged energy.
            speaker_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Speaker ids used to condition features of speech output by the model.
            lang_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Language ids used to condition features of speech output by the model.
            speaker_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`, *optional*, defaults to `None`):
                Embedding containing conditioning signals for the features of the speech.
            return_dict (`bool`, *optional*, defaults to `None`):
                Whether or not to return a [`FastSpeech2ConformerModelOutput`] instead of a plain tuple.
            output_attentions (`bool`, *optional*, defaults to `None`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*, defaults to `None`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.

        Returns:

        Example:

        ```python
        >>> from transformers import (
        ...     FastSpeech2ConformerTokenizer,
        ...     FastSpeech2ConformerModel,
        ...     FastSpeech2ConformerHifiGan,
        ... )

        >>> tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        >>> inputs = tokenizer("some text to convert to speech", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]

        >>> model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
        >>> output_dict = model(input_ids, return_dict=True)
        >>> spectrogram = output_dict["spectrogram"]

        >>> vocoder = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")
        >>> waveform = vocoder(spectrogram)
        >>> print(waveform.shape)
        torch.Size([1, 49664])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

        has_missing_labels = (
            spectrogram_labels is None or duration_labels is None or pitch_labels is None or energy_labels is None
        )
        if self.training and has_missing_labels:
            raise ValueError("All labels must be provided to run in training mode.")

        # forward encoder
        text_masks = attention_mask.unsqueeze(-2)

        encoder_outputs = self.encoder(
            input_ids,
            text_masks,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]

        # Integrate with language id, speaker id, and speaker embedding
        if self.multispeaker_model and speaker_ids is not None:
            speaker_id_embeddings = self.speaker_id_embedding(speaker_ids.view(-1))
            hidden_states = hidden_states + speaker_id_embeddings.unsqueeze(1)

        if self.multilingual_model and lang_ids is not None:
            language_id_embbedings = self.language_id_embedding(lang_ids.view(-1))
            hidden_states = hidden_states + language_id_embbedings.unsqueeze(1)

        if self.speaker_embed_dim is not None and speaker_embedding is not None:
            embeddings_expanded = (
                nn.functional.normalize(speaker_embedding).unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            )
            hidden_states = self.projection(torch.cat([hidden_states, embeddings_expanded], dim=-1))

        # forward duration predictor and variance predictors
        duration_mask = ~attention_mask.bool()

        if self.stop_gradient_from_pitch_predictor:
            pitch_predictions = self.pitch_predictor(hidden_states.detach(), duration_mask.unsqueeze(-1))
        else:
            pitch_predictions = self.pitch_predictor(hidden_states, duration_mask.unsqueeze(-1))

        if self.stop_gradient_from_energy_predictor:
            energy_predictions = self.energy_predictor(hidden_states.detach(), duration_mask.unsqueeze(-1))
        else:
            energy_predictions = self.energy_predictor(hidden_states, duration_mask.unsqueeze(-1))

        duration_predictions = self.duration_predictor(hidden_states)
        duration_predictions = duration_predictions.masked_fill(duration_mask, 0.0)

        if not self.training:
            # use prediction in inference
            embedded_pitch_curve = self.pitch_embed(pitch_predictions)
            embedded_energy_curve = self.energy_embed(energy_predictions)
            hidden_states = hidden_states + embedded_energy_curve + embedded_pitch_curve
            hidden_states = length_regulator(hidden_states, duration_predictions, self.config.speaking_speed)
        else:
            # use groundtruth in training
            embedded_pitch_curve = self.pitch_embed(pitch_labels)
            embedded_energy_curve = self.energy_embed(energy_labels)
            hidden_states = hidden_states + embedded_energy_curve + embedded_pitch_curve
            hidden_states = length_regulator(hidden_states, duration_labels)

        # forward decoder
        if not self.training:
            hidden_mask = None
        else:
            spectrogram_mask = (spectrogram_labels != -100).any(dim=-1)
            spectrogram_mask = spectrogram_mask.int()
            if self.reduction_factor > 1:
                length_dim = spectrogram_mask.shape[1] - spectrogram_mask.shape[1] % self.reduction_factor
                spectrogram_mask = spectrogram_mask[:, :, :length_dim]
            hidden_mask = spectrogram_mask.unsqueeze(-2)

        decoder_outputs = self.decoder(
            hidden_states,
            hidden_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        outputs_before_postnet, outputs_after_postnet = self.speech_decoder_postnet(decoder_outputs[0])

        loss = None
        if self.training:
            # calculate loss
            loss_duration_mask = ~duration_mask
            loss_spectrogram_mask = spectrogram_mask.unsqueeze(-1).bool()
            loss = self.criterion(
                outputs_after_postnet=outputs_after_postnet,
                outputs_before_postnet=outputs_before_postnet,
                duration_outputs=duration_predictions,
                pitch_outputs=pitch_predictions,
                energy_outputs=energy_predictions,
                spectrogram_labels=spectrogram_labels,
                duration_labels=duration_labels,
                pitch_labels=pitch_labels,
                energy_labels=energy_labels,
                duration_mask=loss_duration_mask,
                spectrogram_mask=loss_spectrogram_mask,
            )

        if not return_dict:
            postnet_outputs = (outputs_after_postnet,)
            audio_feature_predictions = (
                duration_predictions,
                pitch_predictions,
                energy_predictions,
            )
            outputs = postnet_outputs + encoder_outputs + decoder_outputs[1:] + audio_feature_predictions
            return ((loss,) + outputs) if loss is not None else outputs

        return FastSpeech2ConformerModelOutput(
            loss=loss,
            spectrogram=outputs_after_postnet,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            duration_outputs=duration_predictions,
            pitch_outputs=pitch_predictions,
            energy_outputs=energy_predictions,
        )


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


@add_start_docstrings(
    """HiFi-GAN vocoder.""",
    HIFIGAN_START_DOCSTRING,
)
# Copied from transformers.models.speecht5.modeling_speecht5.SpeechT5HifiGan with SpeechT5->FastSpeech2Conformer
class FastSpeech2ConformerHifiGan(PreTrainedModel):
    config_class = FastSpeech2ConformerHifiGanConfig
    main_input_name = "spectrogram"

    def __init__(self, config: FastSpeech2ConformerHifiGanConfig):
        super().__init__(config)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(
            config.model_in_dim,
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

        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

        self.register_buffer("mean", torch.zeros(config.model_in_dim))
        self.register_buffer("scale", torch.ones(config.model_in_dim))

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv_pre)
        for layer in self.upsampler:
            weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        weight_norm(self.conv_post)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)

    def forward(self, spectrogram: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`torch.FloatTensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

        Returns:
            `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """
        if self.config.normalize_before:
            spectrogram = (spectrogram - self.mean) / self.scale

        is_batched = spectrogram.dim() == 3
        if not is_batched:
            spectrogram = spectrogram.unsqueeze(0)

        hidden_states = spectrogram.transpose(2, 1)

        hidden_states = self.conv_pre(hidden_states)
        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        if not is_batched:
            # remove batch dim and collapse tensor to 1-d audio waveform
            waveform = hidden_states.squeeze(0).transpose(1, 0).view(-1)
        else:
            # remove seq-len dim since this collapses to 1
            waveform = hidden_states.squeeze(1)

        return waveform


@add_start_docstrings(
    "The FastSpeech2ConformerModel with a FastSpeech2ConformerHifiGan vocoder head that performs text-to-speech (waveform).",
    FASTSPEECH2_CONFORMER_WITH_HIFIGAN_START_DOCSTRING,
)
class FastSpeech2ConformerWithHifiGan(PreTrainedModel):
    config_class = FastSpeech2ConformerWithHifiGanConfig

    def __init__(self, config: FastSpeech2ConformerWithHifiGanConfig):
        super().__init__(config)

        self.model = FastSpeech2ConformerModel(config.model_config)
        self.vocoder = FastSpeech2ConformerHifiGan(config.vocoder_config)

        self.config = config

    @replace_return_docstrings(
        output_type=FastSpeech2ConformerWithHifiGanOutput, config_class=FastSpeech2ConformerWithHifiGanConfig
    )
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        spectrogram_labels: Optional[torch.FloatTensor] = None,
        duration_labels: Optional[torch.LongTensor] = None,
        pitch_labels: Optional[torch.FloatTensor] = None,
        energy_labels: Optional[torch.FloatTensor] = None,
        speaker_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        speaker_embedding: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, FastSpeech2ConformerModelOutput]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input sequence of text vectors.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`):
                Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
                `[0, 1]`: 0 for tokens that are **masked**, 1 for tokens that are **not masked**.
            spectrogram_labels (`torch.FloatTensor` of shape `(batch_size, max_spectrogram_length, num_mel_bins)`, *optional*, defaults to `None`):
                Batch of padded target features.
            duration_labels (`torch.LongTensor` of shape `(batch_size, sequence_length + 1)`, *optional*, defaults to `None`):
                Batch of padded durations.
            pitch_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged pitch.
            energy_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length + 1, 1)`, *optional*, defaults to `None`):
                Batch of padded token-averaged energy.
            speaker_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Speaker ids used to condition features of speech output by the model.
            lang_ids (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*, defaults to `None`):
                Language ids used to condition features of speech output by the model.
            speaker_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`, *optional*, defaults to `None`):
                Embedding containing conditioning signals for the features of the speech.
            return_dict (`bool`, *optional*, defaults to `None`):
                Whether or not to return a [`FastSpeech2ConformerModelOutput`] instead of a plain tuple.
            output_attentions (`bool`, *optional*, defaults to `None`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*, defaults to `None`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.

        Returns:

        Example:

        ```python
        >>> from transformers import (
        ...     FastSpeech2ConformerTokenizer,
        ...     FastSpeech2ConformerWithHifiGan,
        ... )

        >>> tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        >>> inputs = tokenizer("some text to convert to speech", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]

        >>> model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/fastspeech2_conformer_with_hifigan")
        >>> output_dict = model(input_ids, return_dict=True)
        >>> waveform = output_dict["waveform"]
        >>> print(waveform.shape)
        torch.Size([1, 49664])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.model_config.use_return_dict
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.model_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.model_config.output_hidden_states
        )

        model_outputs = self.model(
            input_ids,
            attention_mask,
            spectrogram_labels=spectrogram_labels,
            duration_labels=duration_labels,
            pitch_labels=pitch_labels,
            energy_labels=energy_labels,
            speaker_ids=speaker_ids,
            lang_ids=lang_ids,
            speaker_embedding=speaker_embedding,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if not return_dict:
            has_missing_labels = (
                spectrogram_labels is None or duration_labels is None or pitch_labels is None or energy_labels is None
            )
            if has_missing_labels:
                spectrogram = model_outputs[0]
            else:
                spectrogram = model_outputs[1]
        else:
            spectrogram = model_outputs["spectrogram"]
        waveform = self.vocoder(spectrogram)

        if not return_dict:
            return model_outputs + (waveform,)

        return FastSpeech2ConformerWithHifiGanOutput(waveform=waveform, **model_outputs)


__all__ = [
    "FastSpeech2ConformerWithHifiGan",
    "FastSpeech2ConformerHifiGan",
    "FastSpeech2ConformerModel",
    "FastSpeech2ConformerPreTrainedModel",
]

# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and affiliates, and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch EnCodec model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_encodec import EncodecConfig


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "EncodecConfig"


@dataclass
class EncodecOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_values (`torch.FlaotTensor` of shape `(batch_size, sequence_length)`, *optional*)
            Decoded audio values, obtained using the decoder part of Encodec.
    """

    audio_codes: torch.LongTensor = None
    audio_values: torch.FloatTensor = None


@dataclass
class EncodecEncoderOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
            Scaling factor for each `audio_codes` input. This is used to unscale each chunk of audio when decoding.
    """

    audio_codes: torch.LongTensor = None
    audio_scales: torch.FloatTensor = None


@dataclass
class EncodecDecoderOutput(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor`  of shape `(batch_size, segment_length)`, *optional*):
            Decoded audio values, obtained using the decoder part of Encodec.
    """

    audio_values: torch.FloatTensor = None


class EncodecConv1d(nn.Module):
    """Conv1d with asymmetric or causal padding and normalization."""

    def __init__(
        self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode
        self.norm_type = config.norm_type

        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(
                f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}'
            )

        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            logger.warning(
                "EncodecConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation)
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        if self.norm_type == "weight_norm":
            self.conv = weight_norm(self.conv)
        elif self.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels)

        kernel_size = self.conv.kernel_size[0]
        stride = torch.tensor(self.conv.stride[0], dtype=torch.int64)
        dilation = self.conv.dilation[0]

        # Effective kernel size with dilations.
        kernel_size = torch.tensor((kernel_size - 1) * dilation + 1, dtype=torch.int64)

        self.register_buffer("stride", stride, persistent=False)
        self.register_buffer("kernel_size", kernel_size, persistent=False)
        self.register_buffer("padding_total", torch.tensor(kernel_size - stride, dtype=torch.int64), persistent=False)

    def _get_extra_padding_for_conv1d(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """See `pad_for_conv1d`."""
        length = hidden_states.shape[-1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = torch.ceil(n_frames).to(torch.int64) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total

        return ideal_length - length

    @staticmethod
    def _pad1d(hidden_states: torch.Tensor, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0):
        """Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happens.
        """
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if not mode == "reflect":
            return nn.functional.pad(hidden_states, paddings, mode, value)

        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
        padded = nn.functional.pad(hidden_states, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    def forward(self, hidden_states):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_states)

        if self.causal:
            # Left padding for causal
            hidden_states = self._pad1d(hidden_states, (self.padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
            hidden_states = self._pad1d(
                hidden_states, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )

        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        return hidden_states


class EncodecConvTranspose1d(nn.Module):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""

    def __init__(self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio
        self.norm_type = config.norm_type
        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(
                f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}'
            )

        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        if config.norm_type == "weight_norm":
            self.conv = weight_norm(self.conv)
        elif config.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels)

        if not (self.causal or self.trim_right_ratio == 1.0):
            raise ValueError("`trim_right_ratio` != 1.0 only makes sense for causal convolutions")

    def forward(self, hidden_states):
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride

        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2

        padding_left = padding_total - padding_right

        # unpad
        end = hidden_states.shape[-1] - padding_right
        hidden_states = hidden_states[..., padding_left:end]
        return hidden_states


class EncodecLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data. Expects input as convolutional layout.
    """

    def __init__(self, config, dimension):
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, config.num_lstm_layers)

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(2, 0, 1)
        hidden_states = self.lstm(hidden_states)[0] + hidden_states
        hidden_states = hidden_states.permute(1, 2, 0)
        return hidden_states


class EncodecResnetBlock(nn.Module):
    """
    Residual block from SEANet model as used by EnCodec.
    """

    def __init__(self, config: EncodecConfig, dim: int, dilations: List[int]):
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU()]
            block += [EncodecConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)]
        self.block = nn.ModuleList(block)

        if config.use_conv_shortcut:
            self.shortcut = EncodecConv1d(config, dim, dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, hidden_states):
        residual = hidden_states
        for layer in self.block:
            hidden_states = layer(hidden_states)

        return self.shortcut(residual) + hidden_states


class EncodecEncoder(nn.Module):
    """SEANet encoder as used by EnCodec."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        model = [EncodecConv1d(config, config.audio_channels, config.num_filters, config.kernel_size)]
        scaling = 1

        # Downsample to raw audio scale
        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [EncodecResnetBlock(config, current_scale, [config.dilation_growth_rate**j, 1])]
            # Add downsampling layers
            model += [nn.ELU()]
            model += [EncodecConv1d(config, current_scale, current_scale * 2, kernel_size=ratio * 2, stride=ratio)]
            scaling *= 2

        model += [EncodecLSTM(config, scaling * config.num_filters)]
        model += [nn.ELU()]
        model += [EncodecConv1d(config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size)]

        self.layers = nn.ModuleList(model)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecDecoder(nn.Module):
    """SEANet decoder as used by EnCodec."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model = [EncodecConv1d(config, config.hidden_size, scaling * config.num_filters, config.kernel_size)]

        model += [EncodecLSTM(config, scaling * config.num_filters)]

        # Upsample to raw audio scale
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            # Add upsampling layers
            model += [nn.ELU()]
            model += [
                EncodecConvTranspose1d(config, current_scale, current_scale // 2, kernel_size=ratio * 2, stride=ratio)
            ]
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [EncodecResnetBlock(config, current_scale // 2, (config.dilation_growth_rate**j, 1))]
            scaling //= 2

        # Add final layers
        model += [nn.ELU()]
        model += [EncodecConv1d(config, config.num_filters, config.audio_channels, config.last_kernel_size)]
        self.layers = nn.ModuleList(model)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        embed = torch.zeros(config.codebook_size, config.codebook_dim)

        self.codebook_size = config.codebook_size

        self.register_buffer("inited", torch.Tensor([True]))
        self.register_buffer("cluster_size", torch.zeros(config.codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    def quantize(self, hidden_states):
        embed = self.embed.t()
        scaled_states = hidden_states.pow(2).sum(1, keepdim=True)
        dist = -(scaled_states - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def encode(self, hidden_states):
        shape = hidden_states.shape
        # pre-process
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        # quantize
        embed_ind = self.quantize(hidden_states)
        # post-process
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    def decode(self, embed_ind):
        quantize = nn.functional.embedding(embed_ind, self.embed)
        return quantize


class EncodecVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config: EncodecConfig):
        super().__init__()
        self.codebook = EncodecEuclideanCodebook(config)

    def encode(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        quantize = self.codebook.decode(embed_ind)
        quantize = quantize.permute(0, 2, 1)
        return quantize


class EncodecResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.num_quantizers = config.num_quantizers
        self.layers = nn.ModuleList([EncodecVectorQuantization(config) for _ in range(config.num_quantizers)])

    def get_num_quantizers_for_bandwidth(self, bandwidth: Optional[float] = None) -> int:
        """Return num_quantizers based on specified target bandwidth."""
        bw_per_q = math.log2(self.codebook_size) * self.frame_rate
        num_quantizers = self.num_quantizers
        if bandwidth is not None and bandwidth > 0.0:
            num_quantizers = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return num_quantizers

    def encode(self, embeddings: torch.Tensor, bandwidth: Optional[float] = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given bandwidth. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """
        num_quantizers = self.get_num_quantizers_for_bandwidth(bandwidth)
        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        quantized_out = torch.tensor(0.0, device=codes.device)
        for i, indices in enumerate(codes):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


class EncodecPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EncodecConfig
    base_model_prefix = "encodec"
    main_input_name = "input_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
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
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)


ENCODEC_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`EncodecConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


ENCODEC_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
            Raw audio input converted to Float and padded to the approriate length in order to be encoded using chunks
            of length self.chunk_length and a stride of `config.chunk_stride`.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
            Mask to avoid computing scaling factors on padding token indices (can we avoid computing conv on these+).
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            <Tip warning={true}>

             `padding_mask` should always be passed, unless the input was truncated or not padded. This is because in
             order to process tensors effectively, the input audio should be padded so that `input_length % stride =
             step` with `step = chunk_length-stride`. This ensures that all chunks are of the same shape

            </Tip>

        bandwidth (`float`, *optional*):
            The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible
            bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented as
            `bandwidth == 6.0`
        audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
            Scaling factor for each `audio_codes` input.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The EnCodec neural audio codec model.",
    ENCODEC_START_DOCSTRING,
)
class EncodecModel(EncodecPreTrainedModel):
    def __init__(self, config: EncodecConfig):
        super().__init__(config)
        self.config = config

        self.encoder = EncodecEncoder(config)
        self.decoder = EncodecDecoder(config)

        self.quantizer = EncodecResidualVectorQuantizer(config)

        self.bits_per_codebook = int(math.log2(self.config.codebook_size))
        if 2**self.bits_per_codebook != self.config.codebook_size:
            raise ValueError("The codebook_size must be a power of 2.")

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _encode_frame(
        self, input_values: torch.Tensor, bandwidth: float, padding_mask: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes the given input using the underlying VQVAE. If `config.normalize` is set to `True` the input is first
        normalized. The padding mask is required to compute the correct scale.
        """
        length = input_values.shape[-1]
        duration = length / self.config.sampling_rate

        if self.config.chunk_length_s is not None and duration > 1e-5 + self.config.chunk_length_s:
            raise RuntimeError(f"Duration of frame ({duration}) is longer than chunk {self.config.chunk_length_s}")

        scale = None
        if self.config.normalize:
            # if the padding is non zero
            input_values = input_values * padding_mask
            mono = torch.sum(input_values, 1, keepdim=True) / input_values.shape[1]
            scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
            input_values = input_values / scale

        embeddings = self.encoder(input_values)
        codes = self.quantizer.encode(embeddings, bandwidth)
        codes = codes.transpose(0, 1)
        return codes, scale

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor = None,
        bandwidth: Optional[float] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], EncodecEncoderOutput]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            bandwidth (`float`, *optional*):
                The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible
                bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented
                as bandwidth == 6.0

        Returns:
            A list of frames containing the discrete encoded codes for the input audio waveform, along with rescaling
            factors for each chunk when `normalize` is True. Each frames is a tuple `(codebook, scale)`, with
            `codebook` of shape `[batch_size, num_codebooks, frames]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[0]
        if bandwidth not in self.config.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. "
                f"Select one of {self.config.target_bandwidths}."
            )

        _, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

        chunk_length = self.config.chunk_length
        if chunk_length is None:
            chunk_length = input_length
            stride = input_length
        else:
            stride = self.config.chunk_stride

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        encoded_frames = []
        scales = []

        step = chunk_length - stride
        if (input_length % stride) - step != 0:
            raise ValueError(
                "The input length is not properly padded for batched chunked decoding. Make sure to pad the input correctly."
            )

        for offset in range(0, input_length - step, stride):
            mask = padding_mask[..., offset : offset + chunk_length].bool()
            frame = input_values[:, :, offset : offset + chunk_length]
            encoded_frame, scale = self._encode_frame(frame, bandwidth, mask)
            encoded_frames.append(encoded_frame)
            scales.append(scale)

        encoded_frames = torch.stack(encoded_frames)

        if not return_dict:
            return (encoded_frames, scales)

        return EncodecEncoderOutput(encoded_frames, scales)

    @staticmethod
    def _linear_overlap_add(frames: List[torch.Tensor], stride: int):
        # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
        # e.g., more than 2 frames per position.
        # The core idea is to use a weight function that is a triangle,
        # with a maximum value at the middle of the chunk.
        # We use this weighting when summing the frames, and divide by the sum of weights
        # for each positions at the end. Thus:
        #   - if a frame is the only one to cover a position, the weighting is a no-op.
        #   - if 2 frames cover a position:
        #          ...  ...
        #         /   \/   \
        #        /    /\    \
        #            S  T       , i.e. S offset of second frame starts, T end of first frame.
        # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
        # After the final normalization, the weight of the second frame at position `t` is
        # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
        #
        #   - if more than 2 frames overlap at a given point, we hope that by induction
        #      something sensible happens.
        if len(frames) == 0:
            raise ValueError("`frames` cannot be an empty list.")

        device = frames[0].device
        dtype = frames[0].dtype
        shape = frames[0].shape[:-1]
        total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

        frame_length = frames[0].shape[-1]
        time_vec = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1:-1]
        weight = 0.5 - (time_vec - 0.5).abs()

        sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
        out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
        offset: int = 0

        for frame in frames:
            frame_length = frame.shape[-1]
            out[..., offset : offset + frame_length] += weight[:frame_length] * frame
            sum_weight[offset : offset + frame_length] += weight[:frame_length]
            offset += stride

        if sum_weight.min() == 0:
            raise ValueError(f"`sum_weight` minimum element must be bigger than zero: {sum_weight}`")

        return out / sum_weight

    def _decode_frame(self, codes: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        embeddings = self.quantizer.decode(codes)
        outputs = self.decoder(embeddings)
        if scale is not None:
            outputs = outputs * scale.view(-1, 1, 1)
        return outputs

    def decode(
        self,
        audio_codes: torch.Tensor,
        audio_scales: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecDecoderOutput]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
                Scaling factor for each `audio_codes` input.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        chunk_length = self.config.chunk_length
        if chunk_length is None:
            if len(audio_codes) != 1:
                raise ValueError(f"Expected one frame, got {len(audio_codes)}")
            audio_values = self._decode_frame(audio_codes[0], audio_scales[0])
        else:
            decoded_frames = []

            for frame, scale in zip(audio_codes, audio_scales):
                frames = self._decode_frame(frame, scale)
                decoded_frames.append(frames)

            audio_values = self._linear_overlap_add(decoded_frames, self.config.chunk_stride or 1)

        # truncate based on padding mask
        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if not return_dict:
            return (audio_values,)
        return EncodecDecoderOutput(audio_values)

    @add_start_docstrings_to_model_forward(ENCODEC_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=EncodecOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        bandwidth: Optional[float] = None,
        audio_codes: Optional[torch.Tensor] = None,
        audio_scales: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, EncodecModel

        >>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model_id = "facebook/encodec_24khz"
        >>> model = EncodecModel.from_pretrained(model_id)
        >>> processor = AutoProcessor.from_pretrained(model_id)

        >>> inputs = processor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        if audio_codes is not None and audio_scales is None:
            raise ValueError("You specified `audio_codes` but did not specify the `audio_scales`")

        if audio_scales is not None and audio_codes is None:
            raise ValueError("You specified `audio_scales` but did not specify the `audio_codes`")

        if audio_scales is None and audio_codes is None:
            audio_codes, audio_scales = self.encode(input_values, padding_mask, bandwidth, False)

        audio_values = self.decode(audio_codes, audio_scales, padding_mask, return_dict=return_dict)[0]
        if not return_dict:
            return (audio_codes, audio_values)

        return EncodecOutput(audio_codes=audio_codes, audio_values=audio_values)


__all__ = ["EncodecModel", "EncodecPreTrainedModel"]

# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""PyTorch UnivNetModel model."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...modeling_utils import ModelOutput, PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_univnet import UnivNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "UnivNetConfig"

_CHECKPOINT_FOR_DOC = "dg845/univnet-dev"


@dataclass
class UnivNetModelOutput(ModelOutput):
    """
    Output class for the [`UnivNetModel`], which includes the generated audio waveforms and the original unpadded
    lengths of those waveforms (so that the padding can be removed by [`UnivNetModel.batch_decode`]).

    Args:
        waveforms (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Batched 1D (mono-channel) output audio waveforms.
        waveform_lengths (`torch.FloatTensor` of shape `(batch_size,)`):
            The batched length in samples of each unpadded waveform in `waveforms`.
    """

    waveforms: torch.FloatTensor = None
    waveform_lengths: torch.FloatTensor = None


class UnivNetKernelPredictorResidualBlock(nn.Module):
    """
    Implementation of the residual block for the kernel predictor network inside each location variable convolution
    block (LVCBlock).

    Parameters:
        config: (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
    """

    def __init__(
        self,
        config: UnivNetConfig,
    ):
        super().__init__()
        self.channels = config.model_in_channels
        self.kernel_size = config.kernel_predictor_conv_size
        self.dropout_prob = config.kernel_predictor_dropout
        self.leaky_relu_slope = config.leaky_relu_slope

        padding = (self.kernel_size - 1) // 2

        self.dropout = nn.Dropout(self.dropout_prob)
        self.conv1 = nn.Conv1d(self.channels, self.channels, self.kernel_size, padding=padding, bias=True)
        self.conv2 = nn.Conv1d(self.channels, self.channels, self.kernel_size, padding=padding, bias=True)

    def forward(self, hidden_states: torch.FloatTensor):
        # hidden_states should have shape (batch_size, channels, seq_length)
        residual = hidden_states
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.conv2(hidden_states)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        return hidden_states + residual

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv1)
        weight_norm(self.conv2)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv1)
        nn.utils.remove_weight_norm(self.conv2)


class UnivNetKernelPredictor(nn.Module):
    """
    Implementation of the kernel predictor network which supplies the kernel and bias for the location variable
    convolutional layers (LVCs) in each UnivNet LVCBlock.

    Based on the KernelPredictor implementation in
    [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/lvcnet.py#L7).

    Parameters:
        config: (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
        conv_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the location variable convolutional layer kernels (convolutional weight tensor).
        conv_layers (`int`, *optional*, defaults to 4):
            The number of location variable convolutional layers to output kernels and biases for.
    """

    def __init__(
        self,
        config: UnivNetConfig,
        conv_kernel_size: int = 3,
        conv_layers: int = 4,
    ):
        super().__init__()

        self.conv_in_channels = config.model_hidden_channels
        self.conv_out_channels = 2 * config.model_hidden_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        self.kernel_channels = (
            self.conv_in_channels * self.conv_out_channels * self.conv_kernel_size * self.conv_layers
        )
        self.bias_channels = self.conv_out_channels * self.conv_layers

        self.resnet_in_channels = config.num_mel_bins
        self.resnet_hidden_channels = config.kernel_predictor_hidden_channels
        self.resnet_kernel_size = config.kernel_predictor_conv_size
        self.num_blocks = config.kernel_predictor_num_blocks

        self.leaky_relu_slope = config.leaky_relu_slope

        padding = (self.resnet_kernel_size - 1) // 2

        self.input_conv = nn.Conv1d(self.resnet_in_channels, self.resnet_hidden_channels, 5, padding=2, bias=True)

        self.resblocks = nn.ModuleList([UnivNetKernelPredictorResidualBlock(config) for _ in range(self.num_blocks)])

        self.kernel_conv = nn.Conv1d(
            self.resnet_hidden_channels, self.kernel_channels, self.resnet_kernel_size, padding=padding, bias=True
        )
        self.bias_conv = nn.Conv1d(
            self.resnet_hidden_channels, self.bias_channels, self.resnet_kernel_size, padding=padding, bias=True
        )

    def forward(self, spectrogram: torch.FloatTensor):
        """
        Maps a conditioning log-mel spectrogram to a tensor of convolutional kernels and biases, for use in location
        variable convolutional layers. Note that the input spectrogram should have shape (batch_size, input_channels,
        seq_length).

        Args:
            spectrogram (`torch.FloatTensor` of shape `(batch_size, input_channels, seq_length)`):
                Tensor containing the log-mel spectrograms.

        Returns:
            Tuple[`torch.FloatTensor, `torch.FloatTensor`]: tuple of tensors where the first element is the tensor of
            location variable convolution kernels of shape `(batch_size, self.conv_layers, self.conv_in_channels,
            self.conv_out_channels, self.conv_kernel_size, seq_length)` and the second element is the tensor of
            location variable convolution biases of shape `(batch_size, self.conv_layers. self.conv_out_channels,
            seq_length)`.
        """
        batch_size, _, seq_length = spectrogram.shape

        hidden_states = self.input_conv(spectrogram)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)

        for resblock in self.resblocks:
            hidden_states = resblock(hidden_states)

        kernel_hidden_states = self.kernel_conv(hidden_states)
        bias_hidden_states = self.bias_conv(hidden_states)

        # Reshape kernels and biases to appropriate shape
        kernels = kernel_hidden_states.view(
            batch_size,
            self.conv_layers,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            seq_length,
        ).contiguous()
        biases = bias_hidden_states.view(
            batch_size,
            self.conv_layers,
            self.conv_out_channels,
            seq_length,
        ).contiguous()

        return kernels, biases

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.input_conv)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        weight_norm(self.kernel_conv)
        weight_norm(self.bias_conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input_conv)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.kernel_conv)
        nn.utils.remove_weight_norm(self.bias_conv)


class UnivNetLvcResidualBlock(nn.Module):
    """
    Implementation of the location variable convolution (LVC) residual block for the UnivNet residual network.

    Parameters:
        config: (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
        kernel_size (`int`):
            The kernel size for the dilated 1D convolutional layer.
        dilation (`int`):
            The dilation for the dilated 1D convolutional layer.
    """

    def __init__(
        self,
        config: UnivNetConfig,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        self.hidden_channels = config.model_hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.leaky_relu_slope = config.leaky_relu_slope

        padding = self.dilation * (self.kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            padding=padding,
            dilation=self.dilation,
        )

    def forward(self, hidden_states, kernel, bias, hop_size=256):
        residual = hidden_states
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.location_variable_convolution(hidden_states, kernel, bias, hop_size=hop_size)
        # Gated activation unit
        hidden_states = torch.sigmoid(hidden_states[:, : self.hidden_channels, :]) * torch.tanh(
            hidden_states[:, self.hidden_channels :, :]
        )
        # Skip connection
        hidden_states = residual + hidden_states

        return hidden_states

    # Based on https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/lvcnet.py#L171
    def location_variable_convolution(
        self,
        hidden_states: torch.FloatTensor,
        kernel: torch.FloatTensor,
        bias: torch.FloatTensor,
        dilation: int = 1,
        hop_size: int = 256,
    ):
        """
        Performs location-variable convolution operation on the input sequence (hidden_states) using the local
        convolution kernel. This was introduced in [LVCNet: Efficient Condition-Dependent Modeling Network for Waveform
        Generation](https://arxiv.org/abs/2102.10815) by Zhen Zheng, Jianzong Wang, Ning Cheng, and Jing Xiao.

        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, in_channels, in_length)`):
                The input sequence of shape (batch, in_channels, in_length).
            kernel (`torch.FloatTensor` of shape `(batch_size, in_channels, out_channels, kernel_size, kernel_length)`):
                The local convolution kernel of shape (batch, in_channels, out_channels, kernel_size, kernel_length).
            bias (`torch.FloatTensor` of shape `(batch_size, out_channels, kernel_length)`):
                The bias for the local convolution of shape (batch, out_channels, kernel_length).
            dilation (`int`, *optional*, defaults to 1):
                The dilation of convolution.
            hop_size (`int`, *optional*, defaults to 256):
                The hop_size of the conditioning sequence.
        Returns:
            `torch.FloatTensor`: the output sequence after performing local convolution with shape (batch_size,
            out_channels, in_length).
        """
        batch, _, in_length = hidden_states.shape
        batch, _, out_channels, kernel_size, kernel_length = kernel.shape
        if in_length != (kernel_length * hop_size):
            raise ValueError(
                f"Dim 2 of `hidden_states` should be {kernel_length * hop_size}) but got {in_length}. Please check"
                " `hidden_states` or `kernel` and `hop_size` to make sure they are correct."
            )

        padding = dilation * int((kernel_size - 1) / 2)

        # (batch, in_channels, in_length + 2*padding)
        hidden_states = nn.functional.pad(hidden_states, (padding, padding), "constant", 0)
        # (batch, in_channels, kernel_length, hop_size + 2*padding)
        hidden_states = hidden_states.unfold(2, hop_size + 2 * padding, hop_size)

        if hop_size < dilation:
            hidden_states = nn.functional.pad(hidden_states, (0, dilation), "constant", 0)
        # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        hidden_states = hidden_states.unfold(3, dilation, dilation)
        hidden_states = hidden_states[:, :, :, :, :hop_size]
        # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        hidden_states = hidden_states.transpose(3, 4)
        # (batch, in_channels, kernel_length, dilation, _, kernel_size)
        hidden_states = hidden_states.unfold(4, kernel_size, 1)

        # Apply local convolution kernel to hidden_states.
        output_hidden_states = torch.einsum("bildsk,biokl->bolsd", hidden_states, kernel)

        output_hidden_states = output_hidden_states.to(memory_format=torch.channels_last_3d)
        bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
        output_hidden_states = output_hidden_states + bias
        output_hidden_states = output_hidden_states.contiguous().view(batch, out_channels, -1)

        return output_hidden_states

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)


class UnivNetLvcBlock(nn.Module):
    """
    Implementation of the location variable convolution (LVC) residual block of the UnivNet residual block. Includes a
    `UnivNetKernelPredictor` inside to predict the kernels and biases of the LVC layers.

    Based on LVCBlock in
    [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/lvcnet.py#L98)

    Parameters:
        config (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
        layer_id (`int`):
            An integer corresponding to the index of the current LVC resnet block layer. This should be between 0 and
            `len(config.resblock_stride_sizes) - 1)` inclusive.
        lvc_hop_size (`int`, *optional*, defaults to 256):
            The hop size for the location variable convolutional layers.
    """

    def __init__(
        self,
        config: UnivNetConfig,
        layer_id: int,
        lvc_hop_size: int = 256,
    ):
        super().__init__()
        self.hidden_channels = config.model_hidden_channels
        self.kernel_size = config.resblock_kernel_sizes[layer_id]
        self.stride = config.resblock_stride_sizes[layer_id]
        self.dilations = config.resblock_dilation_sizes[layer_id]
        self.cond_hop_length = lvc_hop_size
        self.leaky_relu_slope = config.leaky_relu_slope
        self.num_blocks = len(self.dilations)

        self.convt_pre = nn.ConvTranspose1d(
            self.hidden_channels,
            self.hidden_channels,
            2 * self.stride,
            stride=self.stride,
            padding=self.stride // 2 + self.stride % 2,
            output_padding=self.stride % 2,
        )

        self.kernel_predictor = UnivNetKernelPredictor(config, self.kernel_size, self.num_blocks)

        self.resblocks = nn.ModuleList(
            [UnivNetLvcResidualBlock(config, self.kernel_size, self.dilations[i]) for i in range(self.num_blocks)]
        )

    def forward(self, hidden_states: torch.FloatTensor, spectrogram: torch.FloatTensor):
        # hidden_states: (batch_size, hidden_channels, seq_length)
        # spectrogram: (batch_size, cond_channels, cond_length)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.convt_pre(hidden_states)

        kernels, biases = self.kernel_predictor(spectrogram)

        for i, resblock in enumerate(self.resblocks):
            kernel = kernels[:, i, :, :, :, :]
            bias = biases[:, i, :, :]
            hidden_states = resblock(hidden_states, kernel, bias, hop_size=self.cond_hop_length)

        return hidden_states

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.convt_pre)
        self.kernel_predictor.apply_weight_norm()
        for layer in self.resblocks:
            layer.apply_weight_norm()

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.convt_pre)
        self.kernel_predictor.remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()


UNIVNET_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`UnivNetConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


UNIVNET_INPUTS_DOCSTRING = r"""
    Converts a noise waveform and a conditioning spectrogram to a speech waveform. Passing a batch of log-mel
    spectrograms returns a batch of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a
    single, un-batched speech waveform.

    Args:
        input_features (`torch.FloatTensor`):
            Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
            config.num_mel_channels)`, or un-batched and of shape `(sequence_length, config.num_mel_channels)`.
        noise_sequence (`torch.FloatTensor`, *optional*):
            Tensor containing a noise sequence of standard Gaussian noise. Can be batched and of shape `(batch_size,
            sequence_length, config.model_in_channels)`, or un-batched and of shape (sequence_length,
            config.model_in_channels)`. If not supplied, will be randomly generated.
        padding_mask (`torch.BoolTensor`, *optional*):
            Mask indicating which parts of each sequence are padded. Mask values are selected in `[0, 1]`:

            - 1 for tokens that are **not masked**
            - 0 for tokens that are **masked**

            The mask can be batched and of shape `(batch_size, sequence_length)` or un-batched and of shape
            `(sequence_length,)`.
        generator (`torch.Generator`, *optional*):
            A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
            deterministic.
        return_dict:
            Whether to return a [`~utils.ModelOutput`] subclass instead of a plain tuple.
"""


@add_start_docstrings(
    """UnivNet GAN vocoder.""",
    UNIVNET_START_DOCSTRING,
)
class UnivNetModel(PreTrainedModel):
    config_class = UnivNetConfig
    main_input_name = "input_features"

    def __init__(self, config: UnivNetConfig):
        super().__init__(config)

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.leaky_relu_slope = config.leaky_relu_slope

        self.conv_pre = nn.Conv1d(
            config.model_in_channels,
            config.model_hidden_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

        # Initialize location-variable convolution ResNet Blocks.
        num_layers = len(config.resblock_stride_sizes)
        hop_length = 1
        hop_lengths = []
        for stride in config.resblock_stride_sizes:
            hop_length = hop_length * stride
            hop_lengths.append(hop_length)

        self.resblocks = nn.ModuleList(
            [
                UnivNetLvcBlock(
                    config,
                    layer_id=i,
                    lvc_hop_size=hop_lengths[i],
                )
                for i in range(num_layers)
            ]
        )

        self.conv_post = nn.Conv1d(config.model_hidden_channels, 1, 7, padding=3, padding_mode="reflect")

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(UNIVNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=UnivNetModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: torch.FloatTensor,
        noise_sequence: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], UnivNetModelOutput]:
        r"""
        Returns:

        Example:

         ```python
         >>> from transformers import UnivNetFeatureExtractor, UnivNetModel
         >>> from datasets import load_dataset, Audio

         >>> model = UnivNetModel.from_pretrained("dg845/univnet-dev")
         >>> feature_extractor = UnivNetFeatureExtractor.from_pretrained("dg845/univnet-dev")

         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> # Resample the audio to the feature extractor's sampling rate.
         >>> ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
         >>> inputs = feature_extractor(
         ...     ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
         ... )
         >>> audio = model(**inputs).waveforms
         >>> list(audio.shape)
         [1, 140288]
         ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Resolve batch sizes for noise_sequence and spectrogram
        spectrogram_batched = input_features.dim() == 3
        if not spectrogram_batched:
            input_features = input_features.unsqueeze(0)
        spectrogram_batch_size, spectrogram_length, _ = input_features.shape

        if noise_sequence is not None:
            noise_sequence_batched = noise_sequence.dim() == 3
            if not noise_sequence_batched:
                noise_sequence = noise_sequence.unsqueeze(0)
        else:
            # Randomly generate noise_sequence
            noise_sequence_shape = (spectrogram_batch_size, spectrogram_length, self.config.model_in_channels)
            noise_sequence = torch.randn(
                noise_sequence_shape, generator=generator, dtype=input_features.dtype, device=input_features.device
            )
        noise_sequence_batch_size = noise_sequence.shape[0]

        if spectrogram_batch_size > 1 and noise_sequence_batch_size == 1:
            # Repeat noise_sequence spectrogram_batch_size times
            noise_sequence = noise_sequence.repeat(spectrogram_batch_size, 1, 1)
        elif noise_sequence_batch_size > 1 and spectrogram_batch_size == 1:
            # Repeat spectrogram noise_sequence_batch_size times
            input_features = input_features.repeat(noise_sequence_batch_size, 1, 1)

        if noise_sequence_batch_size != spectrogram_batch_size:
            raise ValueError(
                f"The batch size of `noise_sequence` is {noise_sequence_batch_size} and the batch size of"
                f" `input_features` is {spectrogram_batch_size}, but the two are expected to be equal."
            )

        if padding_mask is not None:
            if padding_mask.dim() == 1:
                padding_mask = padding_mask.unsqueeze(0)
            padding_mask_batch_size = padding_mask.shape[0]
            if padding_mask_batch_size != spectrogram_batch_size:
                raise ValueError(
                    f"The batch size of `padding_mask` is {padding_mask_batch_size} and the batch size of"
                    f" `input_features` is {spectrogram_batch_size}, but the two are expected to be equal."
                )

        # Change shapes to have channels before sequence lengths
        hidden_states = noise_sequence.transpose(2, 1)
        input_features = input_features.transpose(2, 1)

        hidden_states = self.conv_pre(hidden_states)

        for resblock in self.resblocks:
            hidden_states = resblock(hidden_states, input_features)

        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        # Remove sequence length dimension since this collapses to 1
        # NOTE: keep waveforms batched even if there's only one
        waveform = hidden_states.squeeze(1)

        # Get sequence lengths for UnivNetFeatureExtractor.batch_decode.
        waveform_lengths = None
        if padding_mask is not None:
            # Padding is always contiguous and added on the right
            waveform_lengths = torch.sum(padding_mask, dim=1)

        if not return_dict:
            outputs = (waveform, waveform_lengths)
            return outputs

        return UnivNetModelOutput(
            waveforms=waveform,
            waveform_lengths=waveform_lengths,
        )

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def apply_weight_norm(self):
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv_pre)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        weight_norm(self.conv_post)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)


__all__ = ["UnivNetModel"]

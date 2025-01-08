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
"""UnivNetModel model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class UnivNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UnivNetModel`]. It is used to instantiate a
    UnivNet vocoder model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the UnivNet
    [dg845/univnet-dev](https://huggingface.co/dg845/univnet-dev) architecture, which corresponds to the 'c32'
    architecture in [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/master/config/default_c32.yaml).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_channels (`int`, *optional*, defaults to 64):
            The number of input channels for the UnivNet residual network. This should correspond to
            `noise_sequence.shape[1]` and the value used in the [`UnivNetFeatureExtractor`] class.
        model_hidden_channels (`int`, *optional*, defaults to 32):
            The number of hidden channels of each residual block in the UnivNet residual network.
        num_mel_bins (`int`, *optional*, defaults to 100):
            The number of frequency bins in the conditioning log-mel spectrogram. This should correspond to the value
            used in the [`UnivNetFeatureExtractor`] class.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 3, 3]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the UnivNet residual
            network. The length of `resblock_kernel_sizes` defines the number of resnet blocks and should match that of
            `resblock_stride_sizes` and `resblock_dilation_sizes`.
        resblock_stride_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 4]`):
            A tuple of integers defining the stride sizes of the 1D convolutional layers in the UnivNet residual
            network. The length of `resblock_stride_sizes` should match that of `resblock_kernel_sizes` and
            `resblock_dilation_sizes`.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 9, 27], [1, 3, 9, 27], [1, 3, 9, 27]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            UnivNet residual network. The length of `resblock_dilation_sizes` should match that of
            `resblock_kernel_sizes` and `resblock_stride_sizes`. The length of each nested list in
            `resblock_dilation_sizes` defines the number of convolutional layers per resnet block.
        kernel_predictor_num_blocks (`int`, *optional*, defaults to 3):
            The number of residual blocks in the kernel predictor network, which calculates the kernel and bias for
            each location variable convolution layer in the UnivNet residual network.
        kernel_predictor_hidden_channels (`int`, *optional*, defaults to 64):
            The number of hidden channels for each residual block in the kernel predictor network.
        kernel_predictor_conv_size (`int`, *optional*, defaults to 3):
            The kernel size of each 1D convolutional layer in the kernel predictor network.
        kernel_predictor_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for each residual block in the kernel predictor network.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        leaky_relu_slope (`float`, *optional*, defaults to 0.2):
            The angle of the negative slope used by the leaky ReLU activation.

    Example:

    ```python
    >>> from transformers import UnivNetModel, UnivNetConfig

    >>> # Initializing a Tortoise TTS style configuration
    >>> configuration = UnivNetConfig()

    >>> # Initializing a model (with random weights) from the Tortoise TTS style configuration
    >>> model = UnivNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "univnet"

    def __init__(
        self,
        model_in_channels=64,
        model_hidden_channels=32,
        num_mel_bins=100,
        resblock_kernel_sizes=[3, 3, 3],
        resblock_stride_sizes=[8, 8, 4],
        resblock_dilation_sizes=[[1, 3, 9, 27], [1, 3, 9, 27], [1, 3, 9, 27]],
        kernel_predictor_num_blocks=3,
        kernel_predictor_hidden_channels=64,
        kernel_predictor_conv_size=3,
        kernel_predictor_dropout=0.0,
        initializer_range=0.01,
        leaky_relu_slope=0.2,
        **kwargs,
    ):
        if not (len(resblock_kernel_sizes) == len(resblock_stride_sizes) == len(resblock_dilation_sizes)):
            raise ValueError(
                "`resblock_kernel_sizes`, `resblock_stride_sizes`, and `resblock_dilation_sizes` must all have the"
                " same length (which will be the number of resnet blocks in the model)."
            )

        self.model_in_channels = model_in_channels
        self.model_hidden_channels = model_hidden_channels
        self.num_mel_bins = num_mel_bins
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_stride_sizes = resblock_stride_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.kernel_predictor_num_blocks = kernel_predictor_num_blocks
        self.kernel_predictor_hidden_channels = kernel_predictor_hidden_channels
        self.kernel_predictor_conv_size = kernel_predictor_conv_size
        self.kernel_predictor_dropout = kernel_predictor_dropout
        self.initializer_range = initializer_range
        self.leaky_relu_slope = leaky_relu_slope
        super().__init__(**kwargs)


__all__ = ["UnivNetConfig"]

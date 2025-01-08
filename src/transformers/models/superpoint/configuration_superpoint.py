# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class SuperPointConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SuperPointForKeypointDetection`]. It is used to instantiate a
    SuperPoint model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SuperPoint
    [magic-leap-community/superpoint](https://huggingface.co/magic-leap-community/superpoint) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_hidden_sizes (`List`, *optional*, defaults to `[64, 64, 128, 128]`):
            The number of channels in each convolutional layer in the encoder.
        decoder_hidden_size (`int`, *optional*, defaults to 256): The hidden size of the decoder.
        keypoint_decoder_dim (`int`, *optional*, defaults to 65): The output dimension of the keypoint decoder.
        descriptor_decoder_dim (`int`, *optional*, defaults to 256): The output dimension of the descriptor decoder.
        keypoint_threshold (`float`, *optional*, defaults to 0.005):
            The threshold to use for extracting keypoints.
        max_keypoints (`int`, *optional*, defaults to -1):
            The maximum number of keypoints to extract. If `-1`, will extract all keypoints.
        nms_radius (`int`, *optional*, defaults to 4):
            The radius for non-maximum suppression.
        border_removal_distance (`int`, *optional*, defaults to 4):
            The distance from the border to remove keypoints.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:
    ```python
    >>> from transformers import SuperPointConfig, SuperPointForKeypointDetection

    >>> # Initializing a SuperPoint superpoint style configuration
    >>> configuration = SuperPointConfig()
    >>> # Initializing a model from the superpoint style configuration
    >>> model = SuperPointForKeypointDetection(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "superpoint"

    def __init__(
        self,
        encoder_hidden_sizes: List[int] = [64, 64, 128, 128],
        decoder_hidden_size: int = 256,
        keypoint_decoder_dim: int = 65,
        descriptor_decoder_dim: int = 256,
        keypoint_threshold: float = 0.005,
        max_keypoints: int = -1,
        nms_radius: int = 4,
        border_removal_distance: int = 4,
        initializer_range=0.02,
        **kwargs,
    ):
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_size = decoder_hidden_size
        self.keypoint_decoder_dim = keypoint_decoder_dim
        self.descriptor_decoder_dim = descriptor_decoder_dim
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.border_removal_distance = border_removal_distance
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


__all__ = ["SuperPointConfig"]

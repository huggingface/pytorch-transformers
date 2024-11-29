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

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class LightGlueConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LightGlueModel`]. It is used to instantiate a
    LightGlue model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LightGlue
    [stevenbucaille/superglue_indoor](https://huggingface.co/stevenbucaille/superglue_indoor) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        keypoint_detector_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SuperPointConfig`):
            The config object or dictionary of the keypoint detector.
        descriptor_dim (`int`, *optional*, defaults to 256): The dimension of the descriptors.
        num_layers (`int`, *optional*, defaults to 9): The number of self and cross attention layers.
        num_heads (`int`, *optional*, defaults to 4): The number of heads in the multi-head attention.
        depth_confidence (`float`, *optional*, defaults to 0.95): The confidence threshold used to perform early stopping
        width_confidence (`float`, *optional*, defaults to 0.99): The confidence threshold used to prune points
        filter_threshold (`float`, *optional*, defaults to 0.1):  The confidence theshold used to filter matches
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.


    Examples:
        ```python
        >>> from transformers import LightGlueConfig, LightGlueModel
        >>> # Initializing a LightGlue style configuration
        >>> configuration = LightGlueConfig()
        >>> # Initializing a model from the LightGlue style configuration
        >>> model = LightGlueModel(configuration)
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "lightglue"

    def __init__(
        self,
        keypoint_detector_config=None,
        descriptor_dim: int = 256,
        num_layers: int = 9,
        num_heads: int = 4,
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        filter_threshold: float = 0.1,
        initializer_range=0.02,
        **kwargs,
    ):
        if descriptor_dim % num_heads != 0:
            raise ValueError("descriptor_dim % num_heads is different from zero")

        # TODO Add `add_scale_and_orientation` if keypoint detectors like SIFT or DogHardNet are implemented
        self.descriptor_dim = descriptor_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence
        self.filter_threshold = filter_threshold
        self.initializer_range = initializer_range

        if isinstance(keypoint_detector_config, dict):
            keypoint_detector_config["model_type"] = (
                keypoint_detector_config["model_type"] if "model_type" in keypoint_detector_config else "superpoint"
            )
            keypoint_detector_config = CONFIG_MAPPING[keypoint_detector_config["model_type"]](
                **keypoint_detector_config
            )
        if keypoint_detector_config is None:
            keypoint_detector_config = CONFIG_MAPPING["superpoint"]()

        self.keypoint_detector_config = keypoint_detector_config

        super().__init__(**kwargs)

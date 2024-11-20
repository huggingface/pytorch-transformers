# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Configuration for Backbone models"""

from typing import Any, Dict

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class TimmWrapperConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration for a timm backbone [`TimmWrapper`].

    It is used to instantiate a timm backbone model according to the specified arguments, defining the model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:
    ```python
    >>> from transformers import TimmWrapperConfig, TimmWrapper

    >>> # Initializing a timm backbone
    >>> configuration = TimmWrapperConfig("resnet50")

    >>> # Initializing a model from the configuration
    >>> model = TimmWrapper(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "timm_wrapper"

    def __init__(self, initializer_range: float = 0.02, **kwargs):
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        # timm config stores `num_classes` attribute in root of config and in "pretrained_cfg" dict of config.
        # We are removing this attributes in order to have native `transformers` num_labels attribute in config
        # and not to have duplicated attributes

        num_labels_in_kwargs = kwargs.pop("num_labels", None)
        num_labels_in_dict = config_dict.pop("num_classes", None)

        # passed num_labels has priority over num_classes in config_dict
        kwargs["num_labels"] = num_labels_in_kwargs or num_labels_in_dict

        # pop num_classes from "pretrained_cfg",
        # it is not necessary no have it, only root one is used in timm
        if "pretrained_cfg" in config_dict and "num_classes" in config_dict["pretrained_cfg"]:
            config_dict["pretrained_cfg"].pop("num_classes", None)

        return super().from_dict(config_dict, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["num_classes"] = self.num_labels
        return output
# coding=utf-8
# Copyright The HuggingFace Inc. team. All rights reserved.
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
"""VisionTextDualEncoder model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
from ..chinese_clip.configuration_chinese_clip import ChineseCLIPVisionConfig
from ..clip.configuration_clip import CLIPVisionConfig
from ..siglip.configuration_siglip import SiglipVisionConfig


logger = logging.get_logger(__name__)

VISION_MODEL_CONFIGS = {
    "clip_vision_model": CLIPVisionConfig,
    "chinese_clip_vision_model": ChineseCLIPVisionConfig,
    "siglip_vision_model": SiglipVisionConfig,
}


class VisionTextDualEncoderConfig(PretrainedConfig):
    r"""
    [`VisionTextDualEncoderConfig`] is the configuration class to store the configuration of a
    [`VisionTextDualEncoderModel`]. It is used to instantiate [`VisionTextDualEncoderModel`] model according to the
    specified arguments, defining the text model and vision model configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Examples:

    ```python
    >>> from transformers import ViTConfig, BertConfig, VisionTextDualEncoderConfig, VisionTextDualEncoderModel

    >>> # Initializing a BERT and ViT configuration
    >>> config_vision = ViTConfig()
    >>> config_text = BertConfig()

    >>> config = VisionTextDualEncoderConfig.from_vision_text_configs(config_vision, config_text, projection_dim=512)

    >>> # Initializing a BERT and ViT model (with random weights)
    >>> model = VisionTextDualEncoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_vision = model.config.vision_config
    >>> config_text = model.config.text_config

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("vit-bert")

    >>> # loading model and config from pretrained folder
    >>> vision_text_config = VisionTextDualEncoderConfig.from_pretrained("vit-bert")
    >>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert", config=vision_text_config)
    ```"""

    model_type = "vision-text-dual-encoder"
    sub_configs = {"vision_config": AutoConfig, "text_config": AutoConfig}
    is_composition = True

    def __init__(self, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        super().__init__(**kwargs)

        if "vision_config" not in kwargs:
            raise ValueError("`vision_config` can not be `None`.")

        if "text_config" not in kwargs:
            raise ValueError("`text_config` can not be `None`.")

        vision_config = kwargs.pop("vision_config")
        text_config = kwargs.pop("text_config")

        vision_model_type = vision_config.pop("model_type")
        text_model_type = text_config.pop("model_type")

        vision_config_class = VISION_MODEL_CONFIGS.get(vision_model_type)
        if vision_config_class is not None:
            self.vision_config = vision_config_class(**vision_config)
        else:
            self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config)
            if hasattr(self.vision_config, "vision_config"):
                self.vision_config = self.vision_config.vision_config

        self.text_config = AutoConfig.for_model(text_model_type, **text_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    @classmethod
    def from_vision_text_configs(cls, vision_config: PretrainedConfig, text_config: PretrainedConfig, **kwargs):
        r"""
        Instantiate a [`VisionTextDualEncoderConfig`] (or a derived class) from text model configuration and vision
        model configuration.

        Returns:
            [`VisionTextDualEncoderConfig`]: An instance of a configuration object
        """

        return cls(vision_config=vision_config.to_dict(), text_config=text_config.to_dict(), **kwargs)


__all__ = ["VisionTextDualEncoderConfig"]

# coding=utf-8
# Copyright 2024 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
"""VideoLlava model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class VideoLlavaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VideoLlavaForConditionalGeneration`]. It is used to instantiate an
    VideoLlava model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the like LanguageBind/Video-LLaVA-7B-hf.

    e.g. [LanguageBind/Video-LLaVA-7B-hf](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`VideoLlavaVisionConfig`, *optional*):
            Custom vision config or dict. Defaults to `CLIPVisionConfig` if not indicated.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
            Defaults to `LlamaConfig` if not indicated.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        video_token_index (`int`, *optional*, defaults to 32001):
            The video token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the CLIP backbone.
            Can be either "full" to select all features or "default" to select features without `CLS`.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        image_seq_length (`int`, *optional*, defaults to 256):
            Sequence length of one image embedding.
        video_seq_length (`int`, *optional*, defaults to 2056):
            Sequence length of one video embedding.
        multimodal_projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the multimodal projector.

    Example:

    ```python
    >>> from transformers import VideoLlavaForConditionalGeneration, VideoLlavaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a VideoLlava video_llava-1.5-7b style configuration
    >>> configuration = VideoLlavaConfig(vision_config, text_config)

    >>> # Initializing a model from the video_llava-1.5-7b style configuration
    >>> model = VideoLlavaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "video_llava"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        video_token_index=32001,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_seq_length=256,
        video_seq_length=2056,
        multimodal_projector_bias=True,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.image_seq_length = image_seq_length
        self.video_seq_length = video_seq_length
        self.multimodal_projector_bias = multimodal_projector_bias

        self.vision_config = vision_config

        if isinstance(self.vision_config, dict):
            if "model_type" not in vision_config:
                vision_config["model_type"] = "clip_vision_model"
                logger.warning("Key=`model_type` not found in vision config, setting it to `clip_vision_model`")
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=224,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        if isinstance(text_config, dict):
            if "model_type" not in text_config:
                text_config["model_type"] = "llama"
                logger.warning("Key=`model_type` not found in text config, setting it to `llama`")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config
        super().__init__(**kwargs)


__all__ = ["VideoLlavaConfig"]

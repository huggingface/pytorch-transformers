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
""" Palma model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

PALMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Molbap/model7": "https://huggingface.co/Molbap/model7/resolve/main/config.json",
}



class PalmaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PalmaForConditionalGeneration`]. It is used to instantiate an
    Palma model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Palma-9B.

    e.g. [palma-hf/palma-9b](https://huggingface.co/palma-hf/palma-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`PalmaVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the CLIP backbone.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Palma model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~PalmaForConditionalGeneration`]

    Example:

    ```python
    >>> from transformers import PalmaForConditionalGeneration, PalmaConfig, SiglipVisionConfig, GemmaConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a Gemma config
    >>> text_config = GemmaConfig()

    >>> # Initializing a Palma palma-1.5-7b style configuration
    >>> configuration = GemmaConfig(vision_config, text_config)

    >>> # Initializing a model from the palma-1.5-7b style configuration
    >>> model = PalmaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "palma"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=257152, # put dummy token index at end of vocabulary
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        intermediate_size=16384,
        # FIXME how do we pass vision/text specific config keys here?
        # Why is this setup in init, attributes derived are not used, and configs 
        # are then called with other hardcoded arguments?
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.vision_config = vision_config

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "siglip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=4096,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=257152,
                projection_dim=2048,
            )
        self.vocab_size = self.vocab_size

        self.text_config = text_config

        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "gemma"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            self.vocab_size = self.text_config.vocab_size
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["gemma"](
                hidden_size=2048,
                num_hidden_layers=18, # similar to gemma-2b
                intermediate_size=16384,
                num_attention_heads=8,
                num_key_value_heads=1,
            )

        super().__init__(**kwargs)

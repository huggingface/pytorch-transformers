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
""" PaLIGemmamodel configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

PALIGEMMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Molbap/model7": "https://huggingface.co/Molbap/model7/resolve/main/config.json",
}


class PaLIGemmaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PaLIGemmaForConditionalGeneration`]. It is used to instantiate an
    PaLIGemmamodel according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PaLIGemma-2B.

    e.g. [paligemma-hf/paligemma-2b](https://huggingface.co/paligemma-hf/paligemma-2b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`PaLIGemmaVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 257152):
            The image token index to encode the image prompt.
        vocab_size (`int`, *optional*, defaults to 257152):
            Vocabulary size of the PaLIGemmamodel. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~PaLIGemmaForConditionalGeneration`]
        projection_dim (`int`, *optional*, defaults to 2048):
            Dimension of the multimodal projection space.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden layer of the Language model.

    Example:

    ```python
    >>> from transformers import PaLIGemmaForConditionalGeneration, PaLIGemmaConfig, SiglipVisionConfig, GemmaConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a Gemma config
    >>> text_config = GemmaConfig()

    >>> # Initializing a PaLIGemma paligemma-1.5-7b style configuration
    >>> configuration = GemmaConfig(vision_config, text_config)

    >>> # Initializing a model from the paligemma-1.5-7b style configuration
    >>> model = PaLIGemmaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "paligemma"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=257152,  # put dummy token index at end of vocabulary
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False

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
                vision_use_head=False,
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
                num_hidden_layers=18,  # similar to gemma-2b
                intermediate_size=16384,
                num_attention_heads=8,
                num_key_value_heads=1,
                is_encoder_decoder=False,
            )
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2

        super().__init__(**kwargs)

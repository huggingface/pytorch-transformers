# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""ColPali model configuration"""

import logging
from copy import deepcopy

from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.getLogger(__name__)


class ColPaliConfig(PretrainedConfig):
    r"""
    Configuration class to store the configuration of a [`ColPaliForRetrieval`]. It is used to instantiate an instance
    of `ColPaliForRetrieval` according to the specified arguments, defining the model architecture following the methodology
    from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.

    Creating a configuration with the default settings will result in a configuration where the VLM backbone is set to the
    default PaliGemma configuration, i.e the one from [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2).

    The ColPali config is very similar to [`PaligemmaConfig`], but with an extra attribute defining the embedding dimension.

    Note that contrarily to what the class name suggests (actually the name refers to the ColPali **methodology**), you can
    use a different VLM backbone model than PaliGemma by passing the corresponding VLM configuration to the class constructor.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vlm_config (`PretrainedConfig`, *optional*):
            Configuration of the VLM backbone model.
        text_config (`PretrainedConfig`, *optional*):
            Configuration of the text backbone model. Overrides the `text_config` attribute of the `vlm_config` if provided.
        embedding_dim (`int`, *optional*, defaults to 128):
            Dimension of the multi-vector embeddings produced by the model.

    Example:

    ```python
    from transformers.models.colpali import ColPaliConfig, ColPaliForRetrieval

    config = ColPaliConfig()
    model = ColPaliForRetrieval(config)
    ```
    """

    model_type = "colpali"
    sub_configs = {"vlm_config": PretrainedConfig, "text_config": AutoConfig}

    def __init__(
        self,
        vlm_config=None,
        text_config=None,
        embedding_dim: int = 128,
        **kwargs,
    ):
        if vlm_config is None:
            vlm_config = CONFIG_MAPPING["paligemma"]()
            logger.info(
                "`vlm_config` is `None`. Initializing `vlm_config` with the `PaliGemmaConfig` with default values."
            )
        elif isinstance(vlm_config, dict):
            vlm_config = deepcopy(vlm_config)
            if "model_type" not in vlm_config:
                raise KeyError(
                    "The `model_type` key is missing in the `vlm_config` dictionary. Please provide the model type."
                )
            elif vlm_config["model_type"] not in CONFIG_MAPPING:
                raise ValueError(
                    f"The model type `{vlm_config['model_type']}` is not supported. Please provide a valid model type."
                )
            vlm_config = CONFIG_MAPPING[vlm_config["model_type"]](**vlm_config)
        elif isinstance(vlm_config, PretrainedConfig):
            vlm_config = vlm_config
        else:
            raise TypeError(
                f"Invalid type for `vlm_config`. Expected `PretrainedConfig`, `dict`, or `None`, but got {type(vlm_config)}."
            )

        self.vlm_config = vlm_config
        self.text_config = text_config = text_config if text_config is not None else vlm_config.text_config
        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "gemma"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)

        self.embedding_dim = embedding_dim

        super().__init__(**kwargs)


__all__ = ["ColPaliConfig"]

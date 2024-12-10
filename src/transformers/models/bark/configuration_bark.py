# coding=utf-8
# Copyright 2023 The Suno AI Authors and The HuggingFace Inc. team. All rights reserved.
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
"""BARK model configuration"""

from typing import Dict

from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


BARK_SUBMODELCONFIG_START_DOCSTRING = """
    This is the configuration class to store the configuration of a [`{model}`]. It is used to instantiate the model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Bark [suno/bark](https://huggingface.co/suno/bark)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        block_size (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        input_vocab_size (`int`, *optional*, defaults to 10_048):
            Vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`{model}`]. Defaults to 10_048 but should be carefully thought with
            regards to the chosen sub-model.
        output_vocab_size (`int`, *optional*, defaults to 10_048):
            Output vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented
            by the: `output_ids` when passing forward a [`{model}`]. Defaults to 10_048 but should be carefully thought
            with regards to the chosen sub-model.
        num_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the given sub-model.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer architecture.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the architecture.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the linear layers and layer norm layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
"""


class BarkSubModelConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "vocab_size": "input_vocab_size",
        "window_size": "block_size",
    }

    def __init__(
        self,
        block_size=1024,
        input_vocab_size=10_048,
        output_vocab_size=10_048,
        num_layers=12,
        num_heads=12,
        hidden_size=768,
        dropout=0.0,
        bias=True,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        initializer_range=0.02,
        use_cache=True,
        **kwargs,
    ):
        self.block_size = block_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bias = bias
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


@add_start_docstrings(
    BARK_SUBMODELCONFIG_START_DOCSTRING.format(config="BarkSemanticConfig", model="BarkSemanticModel"),
    """
    Example:

    ```python
    >>> from transformers import BarkSemanticConfig, BarkSemanticModel

    >>> # Initializing a Bark sub-module style configuration
    >>> configuration = BarkSemanticConfig()

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = BarkSemanticModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```""",
)
class BarkSemanticConfig(BarkSubModelConfig):
    model_type = "semantic"
    base_config_key = "semantic_config"


@add_start_docstrings(
    BARK_SUBMODELCONFIG_START_DOCSTRING.format(config="BarkCoarseConfig", model="BarkCoarseModel"),
    """
    Example:

    ```python
    >>> from transformers import BarkCoarseConfig, BarkCoarseModel

    >>> # Initializing a Bark sub-module style configuration
    >>> configuration = BarkCoarseConfig()

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = BarkCoarseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```""",
)
class BarkCoarseConfig(BarkSubModelConfig):
    model_type = "coarse_acoustics"
    base_config_key = "coarse_acoustics_config"


@add_start_docstrings(
    BARK_SUBMODELCONFIG_START_DOCSTRING.format(config="BarkFineConfig", model="BarkFineModel"),
    """
        n_codes_total (`int`, *optional*, defaults to 8):
            The total number of audio codebooks predicted. Used in the fine acoustics sub-model.
        n_codes_given (`int`, *optional*, defaults to 1):
            The number of audio codebooks predicted in the coarse acoustics sub-model. Used in the acoustics
            sub-models.
    Example:

    ```python
    >>> from transformers import BarkFineConfig, BarkFineModel

    >>> # Initializing a Bark sub-module style configuration
    >>> configuration = BarkFineConfig()

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = BarkFineModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```""",
)
class BarkFineConfig(BarkSubModelConfig):
    model_type = "fine_acoustics"
    base_config_key = "fine_acoustics_config"

    def __init__(self, tie_word_embeddings=True, n_codes_total=8, n_codes_given=1, **kwargs):
        self.n_codes_total = n_codes_total
        self.n_codes_given = n_codes_given

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class BarkConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`BarkModel`]. It is used to instantiate a Bark
    model according to the specified sub-models configurations, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the Bark
    [suno/bark](https://huggingface.co/suno/bark) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
    semantic_config ([`BarkSemanticConfig`], *optional*):
        Configuration of the underlying semantic sub-model.
    coarse_acoustics_config ([`BarkCoarseConfig`], *optional*):
        Configuration of the underlying coarse acoustics sub-model.
    fine_acoustics_config ([`BarkFineConfig`], *optional*):
        Configuration of the underlying fine acoustics sub-model.
    codec_config ([`AutoConfig`], *optional*):
        Configuration of the underlying codec sub-model.

    Example:

    ```python
    >>> from transformers import (
    ...     BarkSemanticConfig,
    ...     BarkCoarseConfig,
    ...     BarkFineConfig,
    ...     BarkModel,
    ...     BarkConfig,
    ...     AutoConfig,
    ... )

    >>> # Initializing Bark sub-modules configurations.
    >>> semantic_config = BarkSemanticConfig()
    >>> coarse_acoustics_config = BarkCoarseConfig()
    >>> fine_acoustics_config = BarkFineConfig()
    >>> codec_config = AutoConfig.from_pretrained("facebook/encodec_24khz")


    >>> # Initializing a Bark module style configuration
    >>> configuration = BarkConfig.from_sub_model_configs(
    ...     semantic_config, coarse_acoustics_config, fine_acoustics_config, codec_config
    ... )

    >>> # Initializing a model (with random weights)
    >>> model = BarkModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "bark"
    sub_configs = {
        "semantic_config": BarkSemanticConfig,
        "coarse_acoustics_config": BarkCoarseConfig,
        "fine_acoustics_config": BarkFineConfig,
        "codec_config": AutoConfig,
    }

    def __init__(
        self,
        semantic_config: Dict = None,
        coarse_acoustics_config: Dict = None,
        fine_acoustics_config: Dict = None,
        codec_config: Dict = None,
        initializer_range=0.02,
        **kwargs,
    ):
        if semantic_config is None:
            semantic_config = {}
            logger.info("semantic_config is None. initializing the semantic model with default values.")

        if coarse_acoustics_config is None:
            coarse_acoustics_config = {}
            logger.info("coarse_acoustics_config is None. initializing the coarse model with default values.")

        if fine_acoustics_config is None:
            fine_acoustics_config = {}
            logger.info("fine_acoustics_config is None. initializing the fine model with default values.")

        if codec_config is None:
            codec_config = {}
            logger.info("codec_config is None. initializing the codec model with default values.")

        self.semantic_config = BarkSemanticConfig(**semantic_config)
        self.coarse_acoustics_config = BarkCoarseConfig(**coarse_acoustics_config)
        self.fine_acoustics_config = BarkFineConfig(**fine_acoustics_config)
        codec_model_type = codec_config["model_type"] if "model_type" in codec_config else "encodec"
        self.codec_config = CONFIG_MAPPING[codec_model_type](**codec_config)

        self.initializer_range = initializer_range

        super().__init__(**kwargs)

    @classmethod
    def from_sub_model_configs(
        cls,
        semantic_config: BarkSemanticConfig,
        coarse_acoustics_config: BarkCoarseConfig,
        fine_acoustics_config: BarkFineConfig,
        codec_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`BarkConfig`] (or a derived class) from bark sub-models configuration.

        Returns:
            [`BarkConfig`]: An instance of a configuration object
        """
        return cls(
            semantic_config=semantic_config.to_dict(),
            coarse_acoustics_config=coarse_acoustics_config.to_dict(),
            fine_acoustics_config=fine_acoustics_config.to_dict(),
            codec_config=codec_config.to_dict(),
            **kwargs,
        )


__all__ = ["BarkCoarseConfig", "BarkConfig", "BarkFineConfig", "BarkSemanticConfig"]

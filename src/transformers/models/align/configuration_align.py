# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" ALIGN model configuration"""

import copy
import os
from typing import TYPE_CHECKING, List, Union


if TYPE_CHECKING:
    pass

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "kakaobrain/align-base": "https://huggingface.co/kakaobrain/align-base/resolve/main/config.json",
}


class ALIGNTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ALIGNTextModel`]. It is used to instantiate a
    ALIGN text encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text encoder of the ALIGN
    [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values here are
    copied from BERT.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Example:

    ```python
    >>> from transformers import ALIGNTextConfig, ALIGNTextModel

    >>> # Initializing a ALIGNTextConfig with kakaobrain/align-base style configuration
    >>> configuration = ALIGNTextConfig()

    >>> # Initializing a ALIGNTextModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = ALIGNTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "align_text_model"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.pad_token_id = pad_token_id

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from ALIGNConfig
        if config_dict.get("model_type") == "align":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ALIGNVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ALIGNVisionModel`]. It is used to instantiate a
    ALIGN vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the ALIGN
    [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture. The default values are copied
    from EfficientNet (efficientnet-b7)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 600):
            The input image size.
        width_coefficient (`float`, *optional*, defaults to 2.0):
            Scaling coefficient for network width at each stage.
        depth_coefficient (`float`, *optional*, defaults to 3.1):
            Scaling coefficient for network depth at each stage.
        depth_divisor `int`, *optional*, defaults to 8):
            A unit of network width.
        kernel_sizes (`List[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`):
            List of kernel sizes to be used in each block.
        in_channels (`List[int]`, *optional*, defaults to `[32, 16, 24, 40, 80, 112, 192]`):
            List of input channel sizes to be used in each block for convolutional layers.
        out_channels (`List[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`):
            List of output channel sizes to be used in each block for convolutional layers.
        depthwise_padding (`List[int]`, *optional*, defaults to `[]`):
            List of block indices with square padding.
        strides: (`List[int]`, *optional*, defaults to `[1, 2, 2, 2, 1, 2, 1]`):
            List of stride sizes to be used in each block for convolutional layers.
        num_block_repeats (`List[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`):
            List of the number of times each block is to repeated.
        expand_ratios (`List[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`):
            List of scaling coefficient of each block.
        squeeze_expansion_ratio (`float`, *optional*, defaults to 0.25):
            Squeeze expansion ratio.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
            `"selu", `"gelu_new"`, `"silu"` and `"mish"` are supported.
        hiddem_dim (`int`, *optional*, defaults to 1280):
            The hidden dimension of the layer before the classification head.
        pooling_type (`str` or `function`, *optional*, defaults to `"mean"`):
            Type of final pooling to be applied before the dense classification head. Available options are [`"mean"`,
            `"max"`]
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        batch_norm_eps (`float`, *optional*, defaults to 1e-3):
            The epsilon used by the batch normalization layers.
        batch_norm_momentum (`float`, *optional*, defaults to 0.99):
            The momentum used by the batch normalization layers.
        dropout_rate (`float`, *optional*, defaults to 0.5):
            The dropout rate to be applied before final classifier layer.
        drop_connect_rate (`float`, *optional*, defaults to 0.2):
            The drop rate for skip connections.

    Example:

    ```python
    >>> from transformers import ALIGNVisionConfig, ALIGNVisionModel

    >>> # Initializing a ALIGNVisionConfig with kakaobrain/align-base style configuration
    >>> configuration = ALIGNVisionConfig()

    >>> # Initializing a ALIGNVisionModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = ALIGNVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "align_vision_model"

    def __init__(
        self,
        num_channels: int = 3,
        image_size: int = 600,
        width_coefficient: float = 2.0,
        depth_coefficient: float = 3.1,
        depth_divisor: int = 8,
        kernel_sizes: List[int] = [3, 3, 5, 3, 5, 5, 3],
        in_channels: List[int] = [32, 16, 24, 40, 80, 112, 192],
        out_channels: List[int] = [16, 24, 40, 80, 112, 192, 320],
        depthwise_padding: List[int] = [],
        strides: List[int] = [1, 2, 2, 2, 1, 2, 1],
        num_block_repeats: List[int] = [1, 2, 2, 3, 3, 4, 1],
        expand_ratios: List[int] = [1, 6, 6, 6, 6, 6, 6],
        squeeze_expansion_ratio: float = 0.25,
        hidden_act: str = "swish",
        hidden_dim: int = 2560,
        pooling_type: str = "mean",
        initializer_range: float = 0.02,
        batch_norm_eps: float = 0.001,
        batch_norm_momentum: float = 0.99,
        dropout_rate: float = 0.5,
        drop_connect_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.image_size = image_size
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.depth_divisor = depth_divisor
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_padding = depthwise_padding
        self.strides = strides
        self.num_block_repeats = num_block_repeats
        self.expand_ratios = expand_ratios
        self.squeeze_expansion_ratio = squeeze_expansion_ratio
        self.hidden_act = hidden_act
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.initializer_range = initializer_range
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.num_hidden_layers = sum(num_block_repeats) * 4

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from ALIGNConfig
        if config_dict.get("model_type") == "align":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ALIGNConfig(PretrainedConfig):
    r"""
    [`ALIGNConfig`] is the configuration class to store the configuration of a [`ALIGNModel`]. It is used to
    instantiate a ALIGN model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ALIGN
    [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ALIGNTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ALIGNVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original ALIGN implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ALIGNConfig, ALIGNModel

    >>> # Initializing a ALIGNConfig with kakaobrain/align-base style configuration
    >>> configuration = ALIGNConfig()

    >>> # Initializing a ALIGNModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = ALIGNModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ALIGNConfig from a ALIGNTextConfig and a ALIGNVisionConfig
    >>> from transformers import ALIGNTextConfig, ALIGNVisionConfig

    >>> # Initializing a ALIGNText and ALIGNVision configuration
    >>> config_text = ALIGNTextConfig()
    >>> config_vision = ALIGNVisionConfig()

    >>> config = ALIGNConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "align"
    is_composition = True

    def __init__(
        self, text_config=None, vision_config=None, projection_dim=640, logit_scale_init_value=2.6592, **kwargs
    ):
        super().__init__(**kwargs)

        # If `_config_dict` exist, we use them for the backward compatibility.
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        if text_config_dict is not None:
            text_config = text_config_dict
        if vision_config_dict is not None:
            vision_config = vision_config_dict

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the ALIGNTextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the ALIGNVisionConfig with default values.")

        self.text_config = ALIGNTextConfig(**text_config)
        self.vision_config = ALIGNVisionConfig(**vision_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: ALIGNTextConfig, vision_config: ALIGNVisionConfig, **kwargs):
        r"""
        Instantiate a [`ALIGNConfig`] (or a derived class) from align text model configuration and align vision model
        configuration.

        Returns:
            [`ALIGNConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output

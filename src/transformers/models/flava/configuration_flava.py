# coding=utf-8
# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
"""FLAVA model configurations"""

from typing import Any, Dict

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class FlavaImageConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FlavaImageModel`]. It is used to instantiate an
    FLAVA model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
    [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        mask_token (`bool`, *optional*, defaults to `True`):
            Whether to use a mask token or not. Used in MIM (Masked Image Modeling) loss for FLAVA.
        vocab_size (`int`, *optional*, defaults to 8192):
            Vocabulary size of the [`FlavaImageCodebook`] used in conjunction with [`FlavaImageModel`] for MIM (Masked
            Image Modeling) loss for FLAVA.

    Example:

    ```python
    >>> from transformers import FlavaImageConfig, FlavaImageModel

    >>> # Initializing a FlavaImageModel with  style configuration
    >>> configuration = FlavaImageConfig()

    >>> # Initializing a FlavaImageModel model (with random weights) from the style configuration
    >>> model = FlavaImageModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flava_image_model"
    base_config_key = "image_config"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: int = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        qkv_bias: bool = True,
        mask_token: bool = True,
        vocab_size: int = 8192,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.mask_token = mask_token
        self.vocab_size = vocab_size


class FlavaTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FlavaTextModel`]. It is used to instantiate an
    FLAVA model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
    [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`FlavaTextModel`].
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`FlavaTextModel`]. Note that even though
            text encoder allows `token_type_ids`'s value as 2, for text-only pretraining and fine-tuning, only 1 is
            used similar to RoBERTa.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048). For VL, max_length passed to model is 77.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.

    Example:

    ```python
    >>> from transformers import FlavaTextConfig, FlavaTextModel

    >>> # Initializing a FlavaTextModel with  style configuration
    >>> configuration = FlavaTextConfig()

    >>> # Initializing a FlavaTextModel model (with random weights) from the style configuration
    >>> model = FlavaTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flava_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size: int = 30522,
        type_vocab_size: int = 2,
        max_position_embeddings: int = 512,
        position_embedding_type: str = "absolute",
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        qkv_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.pad_token_id = pad_token_id


class FlavaMultimodalConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FlavaMultimodalModel`]. It is used to instantiate
    an FLAVA model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
    [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        use_cls_token (`bool`, *optional*, defaults to `True`):
            Whether to use an extra CLS token for multimodal settings. Usually needed by the FLAVA model.


    Example:

    ```python
    >>> from transformers import FlavaMultimodalConfig, FlavaMultimodalModel

    >>> # Initializing a FlavaMultimodalModel with  style configuration
    >>> configuration = FlavaMultimodalConfig()

    >>> # Initializing a FlavaMultimodalModel model (with random weights) from the style configuration
    >>> model = FlavaMultimodalModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flava_multimodal_model"
    base_config_key = "multimodal_config"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: int = "gelu",
        hidden_dropout_prob: int = 0.0,
        attention_probs_dropout_prob: int = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        qkv_bias: bool = True,
        use_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_cls_token = use_cls_token


class FlavaImageCodebookConfig(PretrainedConfig):
    model_type = "flava_image_codebook"
    base_config_key = "image_codebook_config"

    r"""
    [`FlavaImageCodebookConfig`] is the configuration class to store the configuration of a [`FlavaImageCodebook`]. It
    is used to instantiate an FLAVA model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA
    [facebook/flava-image-codebook](https://huggingface.co/facebook/flava-image-codebook) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_groups (`int`, *optional*, defaults to 4):
            Number of groups to be created. This parameter as of now doesn't affect the model and is used for some
            internal calculation and estimations.
        input_channels (`int`, *optional*, defaults to 3):
            Number of channels in the image to be passed.
        num_blocks_per_group (`int`, *optional*, defaults to 2):
            Number of conv-based blocks per group.
        hidden_size (`int`, *optional*, defaults to 256):
            Size of hidden dim for the blocks.
        vocab_size (`int`, *optional*, defaults to 8192):
            Size of the output vocabulary for the codebook.
        freeze (`bool`, defaults to `True`):
            Whether to freeze the weights of the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import FlavaImageCodebookConfig, FlavaImageCodebook

    >>> # Initializing a FlavaImageCodebook with style configuration
    >>> configuration = FlavaImageCodebookConfig()

    >>> # Initializing a FlavaImageCodebook model (with random weights) from the style configuration
    >>> model = FlavaImageCodebook(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    def __init__(
        self,
        num_groups: int = 4,
        input_channels: int = 3,
        num_blocks_per_group: int = 2,
        hidden_size: int = 256,
        vocab_size: int = 8192,
        freeze: int = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_groups = num_groups
        self.input_channels = input_channels
        self.num_blocks_per_group = num_blocks_per_group
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.freeze = freeze
        self.initializer_range = initializer_range


class FlavaConfig(PretrainedConfig):
    r"""
    [`FlavaConfig`] is the configuration class to store the configuration of a [`FlavaModel`]. It is used to
    instantiate FLAVA model according to the specified arguments, defining the text model, image model, image codebook
    and multimodal model configs. Instantiating a configuration with the defaults will yield a similar configuration to
    that of the FLAVA [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`FlavaTextConfig`].
        image_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`FlavaImageConfig`].
        multimodal_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`FlavaMultimodalConfig`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and image projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original FLAVA/CLIP
            implementation.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        ce_ignore_index (`int`, *optional*, defaults to -100):
            Cross entropy index to ignore.
        mim_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to MIM (Masked Image Modeling) unimodal loss
        mlm_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to MLM (Masked Language Modeling) unimodal loss
        global_contrastive_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to global contrastive cross-alignment loss.
        itm_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to image-text matching multimodal loss.
        mmm_image_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to MMM loss's image part.
        mmm_text_weight (`float`, *optional*, defaults to 1.0):
            Weight to be assigned to MMM loss's text part.
        global_backprop_contrastive (`bool`, *optional*, defaults to `True`):
            Whether to use global backpropgation through all workers in contrastive loss.
        skip_unmasked_multimodal_encoder (`bool`, *optional*, defaults to `True`):
            Whether to skip running unmasked multimodal encoder whose outputs are not used by FLAVA losses.
        return_loss (`bool`, *optional*, defaults to `True`):
            Whether to return loss or not

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import FlavaConfig, FlavaModel, FlavaForPreTraining

    >>> # Initializing a FlavaConfig with style configuration
    >>> configuration = FlavaConfig()

    >>> # Initializing a FlavaModel and FlavaForPreTraining model (with random weights) from the style configuration
    >>> model = FlavaModel(configuration)
    >>> model_pre = FlavaForPreTraining(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> configuration_pre = model_pre.config
    ```
    """

    model_type = "flava"
    sub_configs = {
        "text_config": FlavaTextConfig,
        "image_config": FlavaImageConfig,
        "multimodal_config": FlavaMultimodalConfig,
        "image_codebook_config": FlavaImageCodebookConfig,
    }

    def __init__(
        self,
        image_config: Dict[str, Any] = None,
        text_config: Dict[str, Any] = None,
        multimodal_config: Dict[str, Any] = None,
        image_codebook_config: Dict[str, Any] = None,
        hidden_size: int = 768,
        layer_norm_eps: float = 1e-12,
        projection_dim: int = 768,
        init_codebook: bool = True,
        logit_scale_init_value: float = 2.6592,
        initializer_range: float = 0.02,
        ce_ignore_index: int = -100,
        mim_weight: float = 1.0,
        mlm_weight: float = 1.0,
        global_contrastive_weight: float = 1.0,
        itm_weight: float = 1.0,
        mmm_image_weight: float = 1.0,
        mmm_text_weight: float = 1.0,
        global_backprop_contrastive: bool = True,
        skip_unmasked_multimodal_encoder: bool = True,
        return_loss: bool = True,
        **kwargs,
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        text_config_dict = kwargs.pop("text_config_dict", None)
        image_config_dict = kwargs.pop("image_config_dict", None)
        multimodal_config_dict = kwargs.pop("multimodal_config_dict", None)
        image_codebook_config_dict = kwargs.pop("image_codebook_config_dict", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = FlavaTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key not in ["transformers_version"]:
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `FlavaTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if image_config_dict is not None:
            if image_config is None:
                image_config = {}

            # This is the complete result when using `image_config_dict`.
            _image_config_dict = FlavaImageConfig(**image_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _image_config_dict:
                _image_config_dict["id2label"] = {
                    str(key): value for key, value in _image_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_image_config_dict` and `image_config` but being different.
            for key, value in _image_config_dict.items():
                if key in image_config and value != image_config[key] and key not in ["transformers_version"]:
                    # If specified in `image_config_dict`
                    if key in image_config_dict:
                        message = (
                            f"`{key}` is found in both `image_config_dict` and `image_config` but with different "
                            f'values. The value `image_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`image_config_dict` is provided which will be used to initialize `FlavaImageConfig`. "
                            f'The value `image_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `image_config` with the ones in `_image_config_dict`.
            image_config.update(_image_config_dict)

        if multimodal_config_dict is not None:
            if multimodal_config is None:
                multimodal_config = {}

            # This is the complete result when using `multimodal_config_dict`.
            _multimodal_config_dict = FlavaMultimodalConfig(**multimodal_config_dict).to_dict()

            # Give a warning if the values exist in both `_multimodal_config_dict` and `multimodal_config` but being
            # different.
            for key, value in _multimodal_config_dict.items():
                if (
                    key in multimodal_config
                    and value != multimodal_config[key]
                    and key not in ["transformers_version"]
                ):
                    # If specified in `multimodal_config_dict`
                    if key in multimodal_config_dict:
                        message = (
                            f"`{key}` is found in both `multimodal_config_dict` and `multimodal_config` but with "
                            f'different values. The value `multimodal_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`multimodal_config_dict` is provided which will be used to initialize "
                            f'`FlavaMultimodalConfig`. The value `multimodal_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `multimodal_config` with the ones in `_multimodal_config_dict`.
            multimodal_config.update(_multimodal_config_dict)

        if image_codebook_config_dict is not None:
            if image_codebook_config is None:
                image_codebook_config = {}

            # This is the complete result when using `image_codebook_config_dict`.
            _image_codebook_config_dict = FlavaImageCodebookConfig(**image_codebook_config_dict).to_dict()

            # Give a warning if the values exist in both `_image_codebook_config_dict` and `image_codebook_config` but
            # being different.
            for key, value in _image_codebook_config_dict.items():
                if (
                    key in image_codebook_config
                    and value != image_codebook_config[key]
                    and key not in ["transformers_version"]
                ):
                    # If specified in `image_codebook_config_dict`
                    if key in image_codebook_config_dict:
                        message = (
                            f"`{key}` is found in both `image_codebook_config_dict` and `image_codebook_config` but "
                            f'with different values. The value `image_codebook_config_dict["{key}"]` will be used '
                            "instead."
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`image_codebook_config_dict` is provided which will be used to initialize "
                            f'`FlavaImageCodebookConfig`. The value `image_codebook_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `image_codebook_config` with the ones in `_image_codebook_config_dict`.
            image_codebook_config.update(_image_codebook_config_dict)

        if image_config is None:
            image_config = {}
            logger.info("`image_config` is `None`. initializing the `FlavaImageConfig` with default values.")

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `FlavaTextConfig` with default values.")

        if multimodal_config is None:
            multimodal_config = {}
            logger.info("`multimodal_config` is `None`. initializing the `FlavaMultimodalConfig` with default values.")

        if image_codebook_config is None:
            image_codebook_config = {}
            logger.info(
                "`image_codebook_config` is `None`. initializing the `FlavaImageCodebookConfig` with default values."
            )

        self.image_config = FlavaImageConfig(**image_config)
        self.text_config = FlavaTextConfig(**text_config)
        self.multimodal_config = FlavaMultimodalConfig(**multimodal_config)
        self.image_codebook_config = FlavaImageCodebookConfig(**image_codebook_config)
        self.projection_dim = projection_dim
        self.init_codebook = init_codebook

        self.hidden_size = hidden_size
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        self.ce_ignore_index = ce_ignore_index
        self.mim_weight = mim_weight
        self.mlm_weight = mlm_weight
        self.global_contrastive_weight = global_contrastive_weight
        self.itm_weight = itm_weight
        self.mmm_image_weight = mmm_image_weight
        self.mmm_text_weight = mmm_text_weight
        self.global_backprop_contrastive = global_backprop_contrastive
        self.skip_unmasked_multimodal_encoder = skip_unmasked_multimodal_encoder
        self.return_loss = return_loss

    @classmethod
    def from_configs(
        cls,
        image_config: FlavaImageConfig,
        text_config: FlavaTextConfig,
        multimodal_config: FlavaMultimodalConfig,
        image_codebook_config: FlavaImageCodebookConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`FlavaConfig`] (or a derived class) from flava text model configuration, flava image model
        configuration, flava multimodal model and flava codebook model configuration.

        Returns:
            [`FlavaConfig`]: An instance of a configuration object
        """

        return cls(
            image_config=image_config.to_dict(),
            text_config=text_config.to_dict(),
            multimodal_config=multimodal_config.to_dict(),
            image_codebook_config=image_codebook_config.to_dict(),
            **kwargs,
        )


__all__ = ["FlavaConfig", "FlavaImageCodebookConfig", "FlavaImageConfig", "FlavaMultimodalConfig", "FlavaTextConfig"]

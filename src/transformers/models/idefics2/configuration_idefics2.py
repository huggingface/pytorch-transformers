# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Idefics2 model configuration"""

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

IDEFICS2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "HuggingFaceM4": "https://huggingface.co/HuggingFaceM4/resolve/main/config.json",
}


class Idefics2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Idefics2VisionModel`]. It is used to instantiate a
    Idefics2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SigLIP checkpoint
    [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224) used in the Idefics2 model
    [amyeroberts/idefics2](https://huggingface.co/amyeroberts/idefics2).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        intializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for initializing all weight matrices in the model.

    Example:

    ```python
    >>> from transformers import Idefics2VisionConfig, Idefics2VisionModel

    >>> # Initializing a Idefics2VisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = Idefics2VisionConfig()

    >>> # Initializing a Idefics2VisionModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = Idefics2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "idefics2"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from Idefics2Config
        if config_dict.get("model_type") == "idefics2":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class Idefics2PerceiverConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the perceiver block.
        resampler_n_latents (`int`, *optional*, defaults to 64):
            Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
        resampler_depth (`int`, *optional*, defaults to 3):
            Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (<= 3).
        resampler_n_heads (`int`, *optional*, defaults to 16):
            Number of heads in each Transformer block (for multi-headed self-attention).
        resampler_head_dim (`int`, *optional*, defaults to 96):
            Dimensionality of each head projection in the Transformer block.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key-value heads in the perceiver attention block.
        qk_layer_norms_perceiver (`bool`, *optional*, defaults to `True`):
            Whether or not to use qk layer norms in perceiver
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    """

    model_type = "idefics2"

    def __init__(
        self,
        hidden_act="silu",
        resampler_n_latents=64,
        resampler_depth=3,
        resampler_n_heads=16,
        resampler_head_dim=96,
        num_key_value_heads=4,
        qk_layer_norms_perceiver=True,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.hidden_act = hidden_act
        self.resampler_n_latents = resampler_n_latents
        self.resampler_depth = resampler_depth
        self.resampler_n_heads = resampler_n_heads
        self.num_key_value_heads = num_key_value_heads
        self.resampler_head_dim = resampler_head_dim
        self.qk_layer_norms_perceiver = qk_layer_norms_perceiver
        self.attention_dropout = attention_dropout
        if self.num_key_value_heads > self.resampler_n_heads:
            raise ValueError(
                f"num_key_value_heads={self.num_key_value_heads} must be less than or equal to"
                f" resampler_n_heads={self.resampler_n_heads}"
            )
        super().__init__(**kwargs)


class Idefics2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Idefics2Model`]. It is used to instantiate a
    Idefics2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the model of the Idefics2
    [amyeroberts/idefics2](https://huggingface.co/amyeroberts/idefics2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        additional_vocab_size (`int`, *optional*, defaults to 2):
            Additional vocabulary size of the model, typically for the special "<img>" token.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Idefics2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Idefics2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with. Mistral's sliding window attention
            allows sequence of up to 4096*32 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        image_token_id (`int`, *optional*, defaults to 32001):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention window size. If not specified, will default to `4096`.
        qk_layer_norms (`bool`, *optional*, defaults to `True`):
            Whether to add layer norm after q and k
        freeze_text_layers (`bool`, *optional*, defaults to `False`):
            Whether to freeze text layers
        freeze_text_module_exceptions (`bool`, *optional*, defaults to `[]`):
            Exceptions to freezing text layers when `freeze_text_layers` is `True`
        freeze_vision_layers (`bool`, *optional*, defaults to `False`):
            Whether to freeze vision layers
        freeze_vision_module_exceptions (`bool`, *optional*, defaults to `[]`):
            Exceptions to freezing vision layers when `freeze_vision_layers` is `True`
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        vision_config (`IdeficsVisionConfig` or `dict`, *optional*):
            Custom vision config or dict
        perceiver_config (`IdeficsPerceiverConfig` or `dict`, *optional*):
            Custom perceiver config or dict

    Example:
    ```python
    >>> from transformers import Idefics2Model, Idefics2Config
    >>> # Initializing configuration
    >>> configuration = Idefics2Config()
    >>> # Initializing a model from the configuration
    >>> model = Idefics2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "idefics2"
    is_composition = False

    def __init__(
        self,
        additional_vocab_size=2,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 8,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,  # None in the original configuration_mistral, we set it to the unk_token_id
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=32_001,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        qk_layer_norms=True,
        freeze_text_layers=False,
        freeze_text_module_exceptions=[],
        freeze_vision_layers=False,
        freeze_vision_module_exceptions=[],
        attention_dropout=0.0,
        vision_config=None,
        perceiver_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.image_token_id = image_token_id
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        self.qk_layer_norms = qk_layer_norms
        self.freeze_vision_layers = freeze_vision_layers

        self.freeze_text_layers = freeze_text_layers
        self.freeze_text_module_exceptions = freeze_text_module_exceptions
        self.freeze_vision_module_exceptions = freeze_vision_module_exceptions

        self.attention_dropout = attention_dropout

        if perceiver_config is None:
            self.perceiver_config = Idefics2PerceiverConfig()
            logger.info("perciver_config is None, using default perceiver config")
        elif isinstance(perceiver_config, dict):
            self.perceiver_config = Idefics2PerceiverConfig(**perceiver_config)
        elif isinstance(perceiver_config, Idefics2PerceiverConfig):
            self.perceiver_config = perceiver_config

        if vision_config is None:
            self.vision_config = Idefics2VisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = Idefics2VisionConfig(**vision_config)
        elif isinstance(vision_config, Idefics2VisionConfig):
            self.vision_config = vision_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

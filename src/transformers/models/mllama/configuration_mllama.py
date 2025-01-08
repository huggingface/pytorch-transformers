# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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
"""Mllama model configuration"""

from typing import Dict, List, Optional

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class MllamaVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MllamaVisionModel`]. It is used to instantiate an
    Mllama vision model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mllama-11B.

    e.g. [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the encoder layers and the pooler layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_global_layers (`int`, *optional*, defaults to 8):
            Number of global layers in the Transformer encoder.
            Vision model has a second transformer encoder, called global.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        vision_output_dim (`int`, *optional*, defaults to 7680):
            Dimensionality of the vision model output. Includes output of transformer
            encoder with intermediate layers and global transformer encoder.
        image_size (`int`, *optional*, defaults to 448):
            The size (resolution) of each image *tile*.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        max_num_tiles (`int`, *optional*, defaults to 4):
            Maximum number of tiles for image splitting.
        intermediate_layers_indices (`List[int]`, *optional*, defaults to [3, 7, 15, 23, 30]):
            Indices of intermediate layers of transformer encoder from which to extract and output features.
            These output features are concatenated with final hidden state of transformer encoder.
        supported_aspect_ratios (`List[List[int]]`, *optional*):
            List of supported aspect ratios for image splitting. If not specified, the default supported aspect ratios
            are [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]] for `max_num_tiles=4`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import MllamaVisionConfig, MllamaVisionModel

    >>> # Initializing a Llama config
    >>> config = MllamaVisionConfig()

    >>> # Initializing a vision model from the mllama-11b style configuration
    >>> model = MllamaVisionModel(config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mllama_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1280,
        hidden_act: str = "gelu",
        num_hidden_layers: int = 32,
        num_global_layers: int = 8,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        intermediate_size: int = 5120,
        vision_output_dim: int = 7680,
        image_size: int = 448,
        patch_size: int = 14,
        norm_eps: float = 1e-5,
        max_num_tiles: int = 4,
        intermediate_layers_indices: Optional[List[int]] = None,
        supported_aspect_ratios: Optional[List[List[int]]] = None,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if supported_aspect_ratios is None:
            if max_num_tiles != 4:
                raise ValueError("max_num_tiles must be 4 for default supported aspect ratios")
            supported_aspect_ratios = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]

        if intermediate_layers_indices is None:
            intermediate_layers_indices = [3, 7, 15, 23, 30]

        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.vision_output_dim = vision_output_dim
        self.patch_size = patch_size
        self.intermediate_layers_indices = intermediate_layers_indices
        self.num_global_layers = num_global_layers
        self.max_num_tiles = max_num_tiles
        self.norm_eps = norm_eps
        self.attention_heads = num_attention_heads
        self.supported_aspect_ratios = supported_aspect_ratios
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    @property
    def max_aspect_ratio_id(self) -> int:
        return len(self.supported_aspect_ratios)


class MllamaTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MllamaTextModel`]. It is used to instantiate an
    Mllama text model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mllama-11B.

    e.g. [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the Mllama text model. Defines the maximum number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`MllamaTextModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If not
            specified, will default to `num_attention_heads`.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        rope_theta (`float`, *optional*, defaults to `500000.0`):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        cross_attention_layers (`List[int]`, *optional*):
            Indices of the cross attention layers. If not specified, will default to [3, 8, 13, 18, 23, 28, 33, 38].
        dropout (`float`, *optional*, defaults to 0):
            The dropout probability for self- and cross-attention layers.
        bos_token_id (`int`, *optional*, defaults to 128000):
            The id of the beginning of sentence token.
        eos_token_id (`int`, *optional*, defaults to 128001):
            The id of the end of sentence token.
        pad_token_id (`int`, *optional*, defaults to 128004):
            The id of the padding token.

    Example:

    ```python
    >>> from transformers import MllamaTextModel, MllamaTextConfig

    >>> # Initializing a Mllama text config
    >>> config = MllamaTextConfig()

    >>> # Initializing a model from the Mllama text configuration
    >>> model = MllamaTextModel(config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mllama_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 4096,
        hidden_act: str = "silu",
        num_hidden_layers: int = 40,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 14_336,
        rope_theta: float = 500_000,
        rope_scaling: Optional[Dict] = None,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 131_072,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        cross_attention_layers: Optional[List[int]] = None,
        dropout: float = 0,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        pad_token_id: Optional[int] = 128004,
        **kwargs,
    ):
        if cross_attention_layers is None:
            cross_attention_layers = [3, 8, 13, 18, 23, 28, 33, 38]

        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.cross_attention_layers = cross_attention_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MllamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MllamaForConditionalGeneration`]. It is used to instantiate an
    Mllama model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mllama-9B.

    e.g. [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MllamaVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `MllamaTextConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 128256):
            The image token index to encode the image prompt.

    Example:

    ```python
    >>> from transformers import MllamaForConditionalGeneration, MllamaConfig, MllamaVisionConfig, MllamaTextConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = MllamaVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = MllamaTextConfig()

    >>> # Initializing a mllama-11b style configuration
    >>> configuration = MllamaConfig(vision_config, text_config)

    >>> # Initializing a model from the mllama-11b style configuration
    >>> model = MllamaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mllama"
    sub_configs = {"text_config": MllamaTextConfig, "vision_config": MllamaVisionConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=128256,
        **kwargs,
    ):
        if vision_config is None:
            self.vision_config = MllamaVisionConfig()
            logger.info("vision_config is None, using default mllama vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = MllamaVisionConfig(**vision_config)
        elif isinstance(vision_config, MllamaVisionConfig):
            self.vision_config = vision_config

        self.image_token_index = image_token_index

        if text_config is None:
            self.text_config = MllamaTextConfig()
            logger.info("text_config is None, using default mllama text config")
        elif isinstance(text_config, dict):
            self.text_config = MllamaTextConfig(**text_config)
        elif isinstance(text_config, MllamaTextConfig):
            self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["MllamaConfig"]

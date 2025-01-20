# coding=utf-8
# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""KOSMOS-2.5 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Kosmos2_5TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2_5TextModel`]. It is used to instantiate a
    KOSMOS-2.5 text decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text decoder of the KOSMOS-2.5
    [microsoft/kosmos-2.5](https://huggingface.co/microsoft/kosmos-2.5) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 108481):
            Vocabulary size of the Kosmos2_5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Kosmos2_5Model`].
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        embed_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the layers and the pooler layer.
        layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        ffn_dim (`int`, *optional*, defaults to 8192):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `True`):
            Scale embeddings by diving by sqrt(embed_dim).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    ```"""

    model_type = "kosmos_2_5_text_model"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "embed_dim",
        "num_hidden_layers": "layers",
    }

    def __init__(
        self,
        vocab_size=108481,
        max_position_embeddings=4096,
        embed_dim=1536,
        layers=24,
        ffn_dim=6144,
        attention_heads=16,
        activation_function="gelu",
        dropout=0.1,
        attention_dropout=0,
        activation_dropout=0.0,
        layerdrop=0.0,
        layer_norm_eps=1e-5,
        init_std=0.02,
        scale_embedding=True,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = embed_dim
        self.layers = layers
        self.ffn_dim = ffn_dim
        self.attention_heads = attention_heads
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.init_std = init_std
        self.scale_embedding = scale_embedding
        self.use_cache = use_cache


class Kosmos2_5VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2_5VisionModel`]. It is used to
    instantiate a KOSMOS-2.5 vision encoder according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of the vision encoder of the KOSMOS-2.5
    [microsoft/kosmos-2.5](https://huggingface.co/microsoft/kosmos-2.5) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        patch_embed_hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the input patch_embedding layer in the Transformer encoder.
        d_ff (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        d_kv (`int`, *optional*, defaults to 64):
            Dimensionality of the key, query, value projections per attention head.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        dense_act_fn (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        seq_len (`int`, *optional*, defaults to 4096):
            Maximum sequence length (here number of patches) supported by the model.
    Example:

    ```python
    >>> from transformers import Kosmos2_5VisionConfig, Kosmos2_5VisionModel

    >>> # Initializing a Kosmos2_5VisionConfig with microsoft/kosmos-2.5 style configuration
    >>> configuration = Kosmos2_5VisionConfig()

    >>> # Initializing a Kosmos2_5VisionModel (with random weights) from the microsoft/kosmos-2.5 style configuration
    >>> model = Kosmos2_5VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "kosmos_2_5_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1536,
        patch_embed_hidden_size=768,
        d_ff=3968,
        d_kv=64,
        num_hidden_layers=18,
        num_attention_heads=24,
        dense_act_fn="gelu_new",
        layer_norm_eps=1e-6,
        dropout_rate=0.0,
        attention_dropout=0.0,
        seq_len=4096,
        initializer_factor=1.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.patch_embed_hidden_size = patch_embed_hidden_size
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.dense_act_fn = dense_act_fn
        self.seq_len = seq_len
        self.d_kv = d_kv
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range


class Kosmos2_5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Kosmos2_5Model`]. It is used to instantiate a
    KOSMOS-2.5 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the KOSMOS-2.5
    [microsoft/kosmos-2.5](https://huggingface.co/microsoft/kosmos-2.5) architecture.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2_5TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2_5VisionConfig`].
        latent_query_num (`int`, *optional*, defaults to 2048):
            The number of latent query tokens that represent the image features used in the text decoder component.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

    model_type = "kosmos-2.5"
    sub_configs = {"text_config": Kosmos2_5TextConfig, "vision_config": Kosmos2_5VisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        latent_query_num=2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the Kosmos2_5TextConfig with default values.")
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the Kosmos2_5VisionConfig with default values.")

        self.text_config = Kosmos2_5TextConfig(**text_config)
        self.vision_config = Kosmos2_5VisionConfig(**vision_config)

        self.latent_query_num = latent_query_num

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: Kosmos2_5TextConfig,
        vision_config: Kosmos2_5VisionConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`Kosmos2_5Config`] (or a derived class) from Kosmos2_5 text model configuration and Kosmos2_5
        vision model configuration.

        Returns:
            [`Kosmos2_5Config`]: An instance of a configuration object
        """

        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )


__all__ = ["Kosmos2_5Config"]

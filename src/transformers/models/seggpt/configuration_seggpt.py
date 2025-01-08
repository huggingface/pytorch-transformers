# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""SegGpt model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class SegGptConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SegGptModel`]. It is used to instantiate a SegGPT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SegGPT
    [BAAI/seggpt-vit-large](https://huggingface.co/BAAI/seggpt-vit-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        image_size (`List[int]`, *optional*, defaults to `[896, 448]`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        mlp_dim (`int`, *optional*):
            The dimensionality of the MLP layer in the Transformer encoder. If unset, defaults to
            `hidden_size` * 4.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The drop path rate for the dropout layers.
        pretrain_image_size (`int`, *optional*, defaults to 224):
            The pretrained size of the absolute position embeddings.
        decoder_hidden_size (`int`, *optional*, defaults to 64):
            Hidden size for decoder.
        use_relative_position_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to use relative position embeddings in the attention layers.
        merge_index (`int`, *optional*, defaults to 2):
            The index of the encoder layer to merge the embeddings.
        intermediate_hidden_state_indices (`List[int]`, *optional*, defaults to `[5, 11, 17, 23]`):
            The indices of the encoder layers which we store as features for the decoder.
        beta (`float`, *optional*, defaults to 0.01):
            Regularization factor for SegGptLoss (smooth-l1 loss).

    Example:

    ```python
    >>> from transformers import SegGptConfig, SegGptModel

    >>> # Initializing a SegGPT seggpt-vit-large style configuration
    >>> configuration = SegGptConfig()

    >>> # Initializing a model (with random weights) from the seggpt-vit-large style configuration
    >>> model = SegGptModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "seggpt"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=[896, 448],
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        mlp_dim=None,
        drop_path_rate=0.1,
        pretrain_image_size=224,
        decoder_hidden_size=64,
        use_relative_position_embeddings=True,
        merge_index=2,
        intermediate_hidden_state_indices=[5, 11, 17, 23],
        beta=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if merge_index > min(intermediate_hidden_state_indices):
            raise ValueError(
                f"Merge index must be less than the minimum encoder output index, but got {merge_index=} and {intermediate_hidden_state_indices=}"
            )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.pretrain_image_size = pretrain_image_size
        self.decoder_hidden_size = decoder_hidden_size
        self.use_relative_position_embeddings = use_relative_position_embeddings
        self.merge_index = merge_index
        self.intermediate_hidden_state_indices = intermediate_hidden_state_indices
        self.beta = beta
        self.mlp_dim = int(hidden_size * 4) if mlp_dim is None else mlp_dim


__all__ = ["SegGptConfig"]

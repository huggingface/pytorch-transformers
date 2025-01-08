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
"""MGP-STR model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MgpstrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`MgpstrModel`]. It is used to instantiate an
    MGP-STR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MGP-STR
    [alibaba-damo/mgp-str-base](https://huggingface.co/alibaba-damo/mgp-str-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`List[int]`, *optional*, defaults to `[32, 128]`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 4):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        max_token_length (`int`, *optional*, defaults to 27):
            The max number of output tokens.
        num_character_labels (`int`, *optional*, defaults to 38):
            The number of classes for character head .
        num_bpe_labels (`int`, *optional*, defaults to 50257):
            The number of classes for bpe head .
        num_wordpiece_labels (`int`, *optional*, defaults to 30522):
            The number of classes for wordpiece head .
        hidden_size (`int`, *optional*, defaults to 768):
            The embedding dimension.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of mlp hidden dim to embedding dim.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        distilled (`bool`, *optional*, defaults to `False`):
            Model includes a distillation token and head as in DeiT models.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        drop_rate (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder.
        attn_drop_rate (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The stochastic depth rate.
        output_a3_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not the model should returns A^3 module attentions.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import MgpstrConfig, MgpstrForSceneTextRecognition

    >>> # Initializing a Mgpstr mgp-str-base style configuration
    >>> configuration = MgpstrConfig()

    >>> # Initializing a model (with random weights) from the mgp-str-base style configuration
    >>> model = MgpstrForSceneTextRecognition(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mgp-str"

    def __init__(
        self,
        image_size=[32, 128],
        patch_size=4,
        num_channels=3,
        max_token_length=27,
        num_character_labels=38,
        num_bpe_labels=50257,
        num_wordpiece_labels=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        distilled=False,
        layer_norm_eps=1e-5,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        output_a3_attentions=False,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.max_token_length = max_token_length
        self.num_character_labels = num_character_labels
        self.num_bpe_labels = num_bpe_labels
        self.num_wordpiece_labels = num_wordpiece_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.distilled = distilled
        self.layer_norm_eps = layer_norm_eps
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.output_a3_attentions = output_a3_attentions
        self.initializer_range = initializer_range


__all__ = ["MgpstrConfig"]

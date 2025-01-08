# coding=utf-8
# Copyright Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""LayoutLMv2 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import is_detectron2_available, logging


logger = logging.get_logger(__name__)


# soft dependency
if is_detectron2_available():
    import detectron2


class LayoutLMv2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LayoutLMv2Model`]. It is used to instantiate an
    LayoutLMv2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LayoutLMv2
    [microsoft/layoutlmv2-base-uncased](https://huggingface.co/microsoft/layoutlmv2-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the LayoutLMv2 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`LayoutLMv2Model`] or [`TFLayoutLMv2Model`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`LayoutLMv2Model`] or
            [`TFLayoutLMv2Model`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum value that the 2D position embedding might ever be used with. Typically set this to something
            large just in case (e.g., 1024).
        max_rel_pos (`int`, *optional*, defaults to 128):
            The maximum number of relative positions to be used in the self-attention mechanism.
        rel_pos_bins (`int`, *optional*, defaults to 32):
            The number of relative position bins to be used in the self-attention mechanism.
        fast_qkv (`bool`, *optional*, defaults to `True`):
            Whether or not to use a single matrix for the queries, keys, values in the self-attention layers.
        max_rel_2d_pos (`int`, *optional*, defaults to 256):
            The maximum number of relative 2D positions in the self-attention mechanism.
        rel_2d_pos_bins (`int`, *optional*, defaults to 64):
            The number of 2D relative position bins in the self-attention mechanism.
        image_feature_pool_shape (`List[int]`, *optional*, defaults to [7, 7, 256]):
            The shape of the average-pooled feature map.
        coordinate_size (`int`, *optional*, defaults to 128):
            Dimension of the coordinate embeddings.
        shape_size (`int`, *optional*, defaults to 128):
            Dimension of the width and height embeddings.
        has_relative_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use a relative attention bias in the self-attention mechanism.
        has_spatial_attention_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use a spatial attention bias in the self-attention mechanism.
        has_visual_segment_embedding (`bool`, *optional*, defaults to `False`):
            Whether or not to add visual segment embeddings.
        detectron2_config_args (`dict`, *optional*):
            Dictionary containing the configuration arguments of the Detectron2 visual backbone. Refer to [this
            file](https://github.com/microsoft/unilm/blob/master/layoutlmft/layoutlmft/models/layoutlmv2/detectron2_config.py)
            for details regarding default values.

    Example:

    ```python
    >>> from transformers import LayoutLMv2Config, LayoutLMv2Model

    >>> # Initializing a LayoutLMv2 microsoft/layoutlmv2-base-uncased style configuration
    >>> configuration = LayoutLMv2Config()

    >>> # Initializing a model (with random weights) from the microsoft/layoutlmv2-base-uncased style configuration
    >>> model = LayoutLMv2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "layoutlmv2"

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
        max_2d_position_embeddings=1024,
        max_rel_pos=128,
        rel_pos_bins=32,
        fast_qkv=True,
        max_rel_2d_pos=256,
        rel_2d_pos_bins=64,
        convert_sync_batchnorm=True,
        image_feature_pool_shape=[7, 7, 256],
        coordinate_size=128,
        shape_size=128,
        has_relative_attention_bias=True,
        has_spatial_attention_bias=True,
        has_visual_segment_embedding=False,
        detectron2_config_args=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.max_rel_pos = max_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.fast_qkv = fast_qkv
        self.max_rel_2d_pos = max_rel_2d_pos
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.convert_sync_batchnorm = convert_sync_batchnorm
        self.image_feature_pool_shape = image_feature_pool_shape
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.has_visual_segment_embedding = has_visual_segment_embedding
        self.detectron2_config_args = (
            detectron2_config_args if detectron2_config_args is not None else self.get_default_detectron2_config()
        )

    @classmethod
    def get_default_detectron2_config(cls):
        return {
            "MODEL.MASK_ON": True,
            "MODEL.PIXEL_STD": [57.375, 57.120, 58.395],
            "MODEL.BACKBONE.NAME": "build_resnet_fpn_backbone",
            "MODEL.FPN.IN_FEATURES": ["res2", "res3", "res4", "res5"],
            "MODEL.ANCHOR_GENERATOR.SIZES": [[32], [64], [128], [256], [512]],
            "MODEL.RPN.IN_FEATURES": ["p2", "p3", "p4", "p5", "p6"],
            "MODEL.RPN.PRE_NMS_TOPK_TRAIN": 2000,
            "MODEL.RPN.PRE_NMS_TOPK_TEST": 1000,
            "MODEL.RPN.POST_NMS_TOPK_TRAIN": 1000,
            "MODEL.POST_NMS_TOPK_TEST": 1000,
            "MODEL.ROI_HEADS.NAME": "StandardROIHeads",
            "MODEL.ROI_HEADS.NUM_CLASSES": 5,
            "MODEL.ROI_HEADS.IN_FEATURES": ["p2", "p3", "p4", "p5"],
            "MODEL.ROI_BOX_HEAD.NAME": "FastRCNNConvFCHead",
            "MODEL.ROI_BOX_HEAD.NUM_FC": 2,
            "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION": 14,
            "MODEL.ROI_MASK_HEAD.NAME": "MaskRCNNConvUpsampleHead",
            "MODEL.ROI_MASK_HEAD.NUM_CONV": 4,
            "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION": 7,
            "MODEL.RESNETS.DEPTH": 101,
            "MODEL.RESNETS.SIZES": [[32], [64], [128], [256], [512]],
            "MODEL.RESNETS.ASPECT_RATIOS": [[0.5, 1.0, 2.0]],
            "MODEL.RESNETS.OUT_FEATURES": ["res2", "res3", "res4", "res5"],
            "MODEL.RESNETS.NUM_GROUPS": 32,
            "MODEL.RESNETS.WIDTH_PER_GROUP": 8,
            "MODEL.RESNETS.STRIDE_IN_1X1": False,
        }

    def get_detectron2_config(self):
        detectron2_config = detectron2.config.get_cfg()
        for k, v in self.detectron2_config_args.items():
            attributes = k.split(".")
            to_set = detectron2_config
            for attribute in attributes[:-1]:
                to_set = getattr(to_set, attribute)
            setattr(to_set, attributes[-1], v)

        return detectron2_config


__all__ = ["LayoutLMv2Config"]

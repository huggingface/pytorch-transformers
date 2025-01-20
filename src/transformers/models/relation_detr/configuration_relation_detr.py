# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Relation DETR model configuration"""

import math
from typing import Literal

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import verify_backbone_config_arguments
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class RelationDetrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RelationDetrModel`]. It is used to instantiate
    a Relation DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Relation DETR
    [xiuqhou/relation-detr](https://huggingface.co/xiuqhou/relation-detr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_timm_backbone (`bool`, *optional*, defaults to `True`):
            Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [`AutoBackbone`]
            API.
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
            case it will default to `ResNetConfig()`.
        backbone_features_format (`str`, *optional*, defaults to `"channels_first"`):
            The format of the features output by the backbone. Can be either `"channels_first"` or `"channels_last"`.
        backbone_post_layer_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization after the backbone. Mainly used for the `FocalNet` backbone to be compatible
            with official implementation.
        num_queries (`int`, *optional*, defaults to 900):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`RelationDetrForObjectDetection`] can detect in a single image.
        hybrid_queries (`int`, *optional*, defaults to 1500):
            Number of hybrid queries, i.e. detection slots. This is the number of auxiliary objects used for faster convergence in training.
        hybrid_assign (`int`, *optional*, defaults to 6):
            Number of hybrid assignments, i.e. the number of auxiliary objects assigned to each object in training.
        num_denoising (`int`, *optional*, defaults to 100):
            Number of denoising samples for each image in training. It is used to generate noisy labels for the speed up training.
        label_noise_ratio (`float`, *optional*, defaults to 0.5):
            The probability of adding noise to the labels in the denoising training.
        box_noise_scale (`float`, *optional*, defaults to 1.0):
            The scale of the noise added to the bounding box coordinates in the denoising training.
        sin_cos_temperature (`float`, *optional*, defaults to 10000):
            The temperature of the sine and cosine positional encodings.
        sin_cos_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the sine and cosine positional encodings.
        sin_cos_scale (`float`, *optional*, defaults to 6.28):
            The scale of the sine and cosine positional encodings.
        sin_cos_offset (`float`, *optional*, defaults to -0.5):
            The offset of the sine and cosine positional encodings.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        d_relation (`int`, *optional*, defaults to 16):
            Dimension of the sin-cos embedding of position relation.
        rel_temperature (`float`, *optional*, defaults to 10000):
            The temperature of the relation positional encodings.
        rel_scale (`float`, *optional*, defaults to 100):
            The scale of the relation positional encodings.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1.0):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        init_bias_prior_prob (`float`, *optional*):
            The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
            If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.
        backbone (`str`, *optional*, defaults to `"resnet50"`):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use pretrained weights for the backbone.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        num_feature_levels (`int`, *optional*, defaults to 4):
            The number of input feature levels.
        encoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the encoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        class_cost (`float`, *optional*, defaults to 2):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        class_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the classification loss in the object detection loss.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        focal_gamma (`float`, *optional*, defaults to 2.0):
            Gamma parameter in the focal loss.
        two_stage_binary_cls (`bool`, *optional*, defaults to `False`):
            Whether to use binary classification for the first-stage loss in two-stage settings.
        disable_custom_kernels (`bool`, *optional*, defaults to `False`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.

    Examples:

    ```python
    >>> from transformers import RelationDetrConfig, RelationDetrModel

    >>> # Initializing a Relation DETR xiuqhou/relation-detr style configuration
    >>> configuration = RelationDetrConfig()

    >>> # Initializing a model (with random weights) from the xiuqhou/relation-detr style configuration
    >>> model = RelationDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "relation_detr"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        use_timm_backbone=True,
        backbone_config=None,
        backbone_features_format: Literal["channels_first", "channels_last"] = "channels_first",
        backbone_post_layer_norm=False,
        num_queries=900,
        hybrid_queries=1500,
        hybrid_assign=6,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        sin_cos_temperature=10000,
        sin_cos_normalize=True,
        sin_cos_scale=2 * math.pi,
        sin_cos_offset=-0.5,
        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        activation_function="relu",
        d_model=256,
        layer_norm_eps=1e-5,
        d_relation=16,
        rel_temperature=10000,
        rel_scale=100,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        init_xavier_std=1.0,
        init_bias_prior_prob=None,
        backbone="resnet50",
        use_pretrained_backbone=True,
        backbone_kwargs=None,
        num_feature_levels=4,
        encoder_n_points=4,
        decoder_n_points=4,
        class_cost=2,
        bbox_cost=5,
        giou_cost=2,
        class_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        focal_alpha=0.25,
        focal_gamma=2.0,
        two_stage_binary_cls=False,
        disable_custom_kernels=False,
        is_encoder_decoder=True,
        **kwargs,
    ):
        if isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone=backbone,
            backbone_config=backbone_config,
            backbone_kwargs=backbone_kwargs,
        )

        if backbone_features_format not in ["channels_first", "channels_last"]:
            raise ValueError(
                f"`backbone_features_format` should be either 'channels_first' or 'channels_last', "
                f"but got {backbone_features_format}."
            )

        self.use_timm_backbone = use_timm_backbone
        self.backbone_config = backbone_config
        self.backbone_features_format = backbone_features_format
        self.backbone_post_layer_norm = backbone_post_layer_norm
        self.sin_cos_temperature = sin_cos_temperature
        self.sin_cos_normalize = sin_cos_normalize
        self.sin_cos_scale = sin_cos_scale
        self.sin_cos_offset = sin_cos_offset
        self.num_queries = num_queries
        self.hybrid_queries = hybrid_queries
        self.hybrid_assign = hybrid_assign
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.activation_function = activation_function
        self.d_model = d_model
        self.layer_norm_eps = layer_norm_eps
        self.d_relation = d_relation
        self.rel_temperature = rel_temperature
        self.rel_scale = rel_scale
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.init_bias_prior_prob = init_bias_prior_prob
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.backbone_kwargs = backbone_kwargs
        # deformable attributes
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.class_loss_coefficient = class_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.two_stage_binary_cls = two_stage_binary_cls
        self.disable_custom_kernels = disable_custom_kernels
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model


__all__ = ["RelationDetrConfig"]

# coding=utf-8
# Copyright 2022 Meta Platforms, Inc.and The HuggingFace Inc. team. All rights reserved.
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
"""Mask2Former model configuration"""

from typing import Dict, List, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import verify_backbone_config_arguments
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class Mask2FormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Mask2FormerModel`]. It is used to instantiate a
    Mask2Former model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Mask2Former
    [facebook/mask2former-swin-small-coco-instance](https://huggingface.co/facebook/mask2former-swin-small-coco-instance)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Currently, Mask2Former only supports the [Swin Transformer](swin) as backbone.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `SwinConfig()`):
            The configuration of the backbone model. If unset, the configuration corresponding to
            `swin-base-patch4-window12-384` will be used.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, `False`):
            Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
            library.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        feature_size (`int`, *optional*, defaults to 256):
            The features (channels) of the resulting feature maps.
        mask_feature_size (`int`, *optional*, defaults to 256):
            The masks' features size, this value will also be used to specify the Feature Pyramid Network features'
            size.
        hidden_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the encoder layers.
        encoder_feedforward_dim (`int`, *optional*, defaults to 1024):
            Dimension of feedforward network for deformable detr encoder used as part of pixel decoder.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of layers in the deformable detr encoder used as part of pixel decoder.
        decoder_layers (`int`, *optional*, defaults to 10):
            Number of layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Feature dimension in feedforward network for transformer decoder.
        pre_norm (`bool`, *optional*, defaults to `False`):
            Whether to use pre-LayerNorm or not for transformer decoder.
        enforce_input_projection (`bool`, *optional*, defaults to `False`):
            Whether to add an input projection 1x1 convolution even if the input channels and hidden dim are identical
            in the Transformer decoder.
        common_stride (`int`, *optional*, defaults to 4):
            Parameter used for determining number of FPN levels used as part of pixel decoder.
        ignore_value (`int`, *optional*, defaults to 255):
            Category id to be ignored during training.
        num_queries (`int`, *optional*, defaults to 100):
            Number of queries for the decoder.
        no_object_weight (`int`, *optional*, defaults to 0.1):
            The weight to apply to the null (no object) class.
        class_weight (`int`, *optional*, defaults to 2.0):
            The weight for the cross entropy loss.
        mask_weight (`int`, *optional*, defaults to 5.0):
            The weight for the mask loss.
        dice_weight (`int`, *optional*, defaults to 5.0):
            The weight for the dice loss.
        train_num_points (`str` or `function`, *optional*, defaults to 12544):
            Number of points used for sampling during loss calculation.
        oversample_ratio (`float`, *optional*, defaults to 3.0):
            Oversampling parameter used for calculating no. of sampled points
        importance_sample_ratio (`float`, *optional*, defaults to 0.75):
            Ratio of points that are sampled via importance sampling.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1.0):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        use_auxiliary_loss (`boolean``, *optional*, defaults to `True`):
            If `True` [`Mask2FormerForUniversalSegmentationOutput`] will contain the auxiliary losses computed using
            the logits from each decoder's stage.
        feature_strides (`List[int]`, *optional*, defaults to `[4, 8, 16, 32]`):
            Feature strides corresponding to features generated from backbone network.
        output_auxiliary_logits (`bool`, *optional*):
            Should the model output its `auxiliary_logits` or not.

    Examples:

    ```python
    >>> from transformers import Mask2FormerConfig, Mask2FormerModel

    >>> # Initializing a Mask2Former facebook/mask2former-swin-small-coco-instance configuration
    >>> configuration = Mask2FormerConfig()

    >>> # Initializing a model (with random weights) from the facebook/mask2former-swin-small-coco-instance style configuration
    >>> model = Mask2FormerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """

    model_type = "mask2former"
    backbones_supported = ["swin"]
    attribute_map = {"hidden_size": "hidden_dim"}

    def __init__(
        self,
        backbone_config: Optional[Dict] = None,
        feature_size: int = 256,
        mask_feature_size: int = 256,
        hidden_dim: int = 256,
        encoder_feedforward_dim: int = 1024,
        activation_function: str = "relu",
        encoder_layers: int = 6,
        decoder_layers: int = 10,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
        dim_feedforward: int = 2048,
        pre_norm: bool = False,
        enforce_input_projection: bool = False,
        common_stride: int = 4,
        ignore_value: int = 255,
        num_queries: int = 100,
        no_object_weight: float = 0.1,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        train_num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        init_std: float = 0.02,
        init_xavier_std: float = 1.0,
        use_auxiliary_loss: bool = True,
        feature_strides: List[int] = [4, 8, 16, 32],
        output_auxiliary_logits: bool = None,
        backbone: Optional[str] = None,
        use_pretrained_backbone: bool = False,
        use_timm_backbone: bool = False,
        backbone_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `Swin` backbone.")
            backbone_config = CONFIG_MAPPING["swin"](
                image_size=224,
                num_channels=3,
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                drop_path_rate=0.3,
                use_absolute_embeddings=False,
                out_features=["stage1", "stage2", "stage3", "stage4"],
            )
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.pop("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone=backbone,
            backbone_config=backbone_config,
            backbone_kwargs=backbone_kwargs,
        )
        # verify that the backbone is supported
        if backbone_config is not None and backbone_config.model_type not in self.backbones_supported:
            logger.warning_once(
                f"Backbone {backbone_config.model_type} is not a supported model and may not be compatible with Mask2Former. "
                f"Supported model types: {','.join(self.backbones_supported)}"
            )

        self.backbone_config = backbone_config
        self.feature_size = feature_size
        self.mask_feature_size = mask_feature_size
        self.hidden_dim = hidden_dim
        self.encoder_feedforward_dim = encoder_feedforward_dim
        self.activation_function = activation_function
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.enforce_input_projection = enforce_input_projection
        self.common_stride = common_stride
        self.ignore_value = ignore_value
        self.num_queries = num_queries
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.use_auxiliary_loss = use_auxiliary_loss
        self.feature_strides = feature_strides
        self.output_auxiliary_logits = output_auxiliary_logits
        self.num_hidden_layers = decoder_layers
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs

        super().__init__(**kwargs)

    @classmethod
    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`Mask2FormerConfig`] (or a derived class) from a pre-trained backbone model configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration.

        Returns:
            [`Mask2FormerConfig`]: An instance of a configuration object
        """
        return cls(
            backbone_config=backbone_config,
            **kwargs,
        )


__all__ = ["Mask2FormerConfig"]

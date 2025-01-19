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
"""DAB-DETR model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import verify_backbone_config_arguments
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class DabDetrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DabDetrModel`]. It is used to instantiate
    a DAB-DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the DAB-DETR
    [IDEA-Research/dab_detr-base](https://huggingface.co/IDEA-Research/dab_detr-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_timm_backbone (`bool`, *optional*, defaults to `True`):
            Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [`AutoBackbone`]
            API.
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
            case it will default to `ResNetConfig()`.
        backbone (`str`, *optional*, defaults to `"resnet50"`):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use pretrained weights for the backbone.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`DabDetrModel`] can detect in a single image. For COCO, we recommend 100 queries.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 2048):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Indicates whether the transformer model architecture is an encoder-decoder or not.
        activation_function (`str` or `function`, *optional*, defaults to `"prelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_size (`int`, *optional*, defaults to 256):
            This parameter is a general dimension parameter, defining dimensions for components such as the encoder layer and projection parameters in the decoder layer, among others.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1.0):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        auxiliary_loss (`bool`, *optional*, defaults to `False`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        dilation (`bool`, *optional*, defaults to `False`):
            Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when `use_timm_backbone` = `True`.
        class_cost (`float`, *optional*, defaults to 2):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        cls_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the classification loss in the object detection loss function.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        temperature_height (`int`, *optional*, defaults to 20):
            Temperature parameter to tune the flatness of positional attention (HEIGHT)
        temperature_width (`int`, *optional*, defaults to 20):
            Temperature parameter to tune the flatness of positional attention (WIDTH)
        query_dim (`int`, *optional*, defaults to 4):
            Query dimension parameter represents the size of the output vector.
        random_refpoints_xy (`bool`, *optional*, defaults to `False`):
            Whether to fix the x and y coordinates of the anchor boxes with random initialization.
        keep_query_pos (`bool`, *optional*, defaults to `False`):
            Whether to concatenate the projected positional embedding from the object query into the original query (key) in every decoder layer.
        num_patterns (`int`, *optional*, defaults to 0):
            Number of pattern embeddings.
        normalize_before (`bool`, *optional*, defaults to `False`):
            Whether we use a normalization layer in the Encoder or not.
        sine_position_embedding_scale (`float`, *optional*, defaults to 'None'):
            Scaling factor applied to the normalized positional encodings.
        initializer_bias_prior_prob (`float`, *optional*):
            The prior probability used by the bias initializer to initialize biases for `enc_score_head` and `class_embed`.
            If `None`, `prior_prob` computed as `prior_prob = 1 / (num_labels + 1)` while initializing model weights.


    Examples:

    ```python
    >>> from transformers import DabDetrConfig, DabDetrModel

    >>> # Initializing a DAB-DETR IDEA-Research/dab_detr-base style configuration
    >>> configuration = DabDetrConfig()

    >>> # Initializing a model (with random weights) from the IDEA-Research/dab_detr-base style configuration
    >>> model = DabDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "dab-detr"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        use_timm_backbone=True,
        backbone_config=None,
        backbone="resnet50",
        use_pretrained_backbone=True,
        backbone_kwargs=None,
        num_queries=300,
        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        is_encoder_decoder=True,
        activation_function="prelu",
        hidden_size=256,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        init_xavier_std=1.0,
        auxiliary_loss=False,
        dilation=False,
        class_cost=2,
        bbox_cost=5,
        giou_cost=2,
        cls_loss_coefficient=2,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        focal_alpha=0.25,
        temperature_height=20,
        temperature_width=20,
        query_dim=4,
        random_refpoints_xy=False,
        keep_query_pos=False,
        num_patterns=0,
        normalize_before=False,
        sine_position_embedding_scale=None,
        initializer_bias_prior_prob=None,
        **kwargs,
    ):
        if query_dim != 4:
            raise ValueError("The query dimensions has to be 4.")

        # We default to values which were previously hard-coded in the model. This enables configurability of the config
        # while keeping the default behavior the same.
        if use_timm_backbone and backbone_kwargs is None:
            backbone_kwargs = {}
            if dilation:
                backbone_kwargs["output_stride"] = 16
            backbone_kwargs["out_indices"] = [1, 2, 3, 4]
            backbone_kwargs["in_chans"] = 3  # num_channels
        # Backwards compatibility
        elif not use_timm_backbone and backbone in (None, "resnet50"):
            if backbone_config is None:
                logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
                backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)
            backbone = None
            # set timm attributes to None
            dilation = None

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone=backbone,
            backbone_config=backbone_config,
            backbone_kwargs=backbone_kwargs,
        )

        self.use_timm_backbone = use_timm_backbone
        self.backbone_config = backbone_config
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.num_hidden_layers = encoder_layers
        self.auxiliary_loss = auxiliary_loss
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.backbone_kwargs = backbone_kwargs
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.cls_loss_coefficient = cls_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.focal_alpha = focal_alpha
        self.query_dim = query_dim
        self.random_refpoints_xy = random_refpoints_xy
        self.keep_query_pos = keep_query_pos
        self.num_patterns = num_patterns
        self.normalize_before = normalize_before
        self.temperature_width = temperature_width
        self.temperature_height = temperature_height
        self.sine_position_embedding_scale = sine_position_embedding_scale
        self.initializer_bias_prior_prob = initializer_bias_prior_prob
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)


__all__ = ["DabDetrConfig"]

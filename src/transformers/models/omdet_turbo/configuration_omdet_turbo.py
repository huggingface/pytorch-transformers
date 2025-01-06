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
"""OmDet-Turbo model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import verify_backbone_config_arguments
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class OmDetTurboConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmDetTurboForObjectDetection`].
    It is used to instantiate a OmDet-Turbo model according to the specified arguments, defining the model architecture
    Instantiating a configuration with the defaults will yield a similar configuration to that of the OmDet-Turbo
    [omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`PretrainedConfig`, *optional*):
            The configuration of the text backbone.
        backbone_config (`PretrainedConfig`, *optional*):
            The configuration of the vision backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use the timm for the vision backbone.
        backbone (`str`, *optional*, defaults to `"swin_tiny_patch4_window7_224"`):
            The name of the pretrained vision backbone to use. If `use_pretrained_backbone=False` a randomly initialized
            backbone with the same architecture `backbone` is used.
        backbone_kwargs (`dict`, *optional*):
            Additional kwargs for the vision backbone.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use a pretrained vision backbone.
        apply_layernorm_after_vision_backbone (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization on the feature maps of the vision backbone output.
        image_size (`int`, *optional*, defaults to 640):
            The size (resolution) of each image.
        disable_custom_kernels (`bool`, *optional*, defaults to `False`):
            Whether to disable custom kernels.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for layer normalization.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value for batch normalization.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        text_projection_in_dim (`int`, *optional*, defaults to 512):
            The input dimension for the text projection.
        text_projection_out_dim (`int`, *optional*, defaults to 512):
            The output dimension for the text projection.
        task_encoder_hidden_dim (`int`, *optional*, defaults to 1024):
            The feedforward dimension for the task encoder.
        class_embed_dim (`int`, *optional*, defaults to 512):
            The dimension of the classes embeddings.
        class_distance_type (`str`, *optional*, defaults to `"cosine"`):
            The type of of distance to compare predicted classes to projected classes embeddings.
            Can be `"cosine"` or `"dot"`.
        num_queries (`int`, *optional*, defaults to 900):
            The number of queries.
        csp_activation (`str`, *optional*, defaults to `"silu"`):
            The activation function of the Cross Stage Partial (CSP) networks of the encoder.
        conv_norm_activation (`str`, *optional*, defaults to `"gelu"`):
            The activation function of the ConvNormLayer layers of the encoder.
        encoder_feedforward_activation (`str`, *optional*, defaults to `"relu"`):
            The activation function for the feedforward network of the encoder.
        encoder_feedforward_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate following the activation of the encoder feedforward network.
        encoder_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate of the encoder multi-head attention module.
        hidden_expansion (`int`, *optional*, defaults to 1):
            The hidden expansion of the CSP networks in the encoder.
        vision_features_channels (`tuple(int)`, *optional*, defaults to `[256, 256, 256]`):
            The projected vision features channels used as inputs for the decoder.
        encoder_hidden_dim (`int`, *optional*, defaults to 256):
            The hidden dimension of the encoder.
        encoder_in_channels (`List(int)`, *optional*, defaults to `[192, 384, 768]`):
            The input channels for the encoder.
        encoder_projection_indices (`List(int)`, *optional*, defaults to `[2]`):
            The indices of the input features projected by each layers.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            The number of attention heads for the encoder.
        encoder_dim_feedforward (`int`, *optional*, defaults to 2048):
            The feedforward dimension for the encoder.
        encoder_layers (`int`, *optional*, defaults to 1):
            The number of layers in the encoder.
        positional_encoding_temperature (`int`, *optional*, defaults to 10000):
            The positional encoding temperature in the encoder.
        num_feature_levels (`int`, *optional*, defaults to 3):
            The number of feature levels for the multi-scale deformable attention module of the decoder.
        decoder_hidden_dim (`int`, *optional*, defaults to 256):
            The hidden dimension of the decoder.
        decoder_num_heads (`int`, *optional*, defaults to 8):
            The number of heads for the decoder.
        decoder_num_layers (`int`, *optional*, defaults to 6):
            The number of layers for the decoder.
        decoder_activation (`str`, *optional*, defaults to `"relu"`):
            The activation function for the decoder.
        decoder_dim_feedforward (`int`, *optional*, defaults to 2048):
            The feedforward dimension for the decoder.
        decoder_num_points (`int`, *optional*, defaults to 4):
            The number of points sampled in the decoder multi-scale deformable attention module.
        decoder_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate for the decoder.
        eval_size (`Tuple[int, int]`, *optional*):
            Height and width used to computes the effective height and width of the position embeddings after taking
            into account the stride (see RTDetr).
        learn_initial_query (`bool`, *optional*, defaults to `False`):
            Whether to learn the initial query.
        cache_size (`int`, *optional*, defaults to 100):
            The cache size for the classes and prompts caches.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder-decoder model or not.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from the architecture. The values in kwargs will be saved as part of the configuration
            and can be used to control the model outputs.

    Examples:

    ```python
    >>> from transformers import OmDetTurboConfig, OmDetTurboForObjectDetection

    >>> # Initializing a OmDet-Turbo omlab/omdet-turbo-swin-tiny-hf style configuration
    >>> configuration = OmDetTurboConfig()

    >>> # Initializing a model (with random weights) from the omlab/omdet-turbo-swin-tiny-hf style configuration
    >>> model = OmDetTurboForObjectDetection(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "omdet-turbo"
    attribute_map = {
        "encoder_hidden_dim": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        text_config=None,
        backbone_config=None,
        use_timm_backbone=True,
        backbone="swin_tiny_patch4_window7_224",
        backbone_kwargs=None,
        use_pretrained_backbone=False,
        apply_layernorm_after_vision_backbone=True,
        image_size=640,
        disable_custom_kernels=False,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        init_std=0.02,
        text_projection_in_dim=512,
        text_projection_out_dim=512,
        task_encoder_hidden_dim=1024,
        class_embed_dim=512,
        class_distance_type="cosine",
        num_queries=900,
        csp_activation="silu",
        conv_norm_activation="gelu",
        encoder_feedforward_activation="relu",
        encoder_feedforward_dropout=0.0,
        encoder_dropout=0.0,
        hidden_expansion=1,
        vision_features_channels=[256, 256, 256],
        encoder_hidden_dim=256,
        encoder_in_channels=[192, 384, 768],
        encoder_projection_indices=[2],
        encoder_attention_heads=8,
        encoder_dim_feedforward=2048,
        encoder_layers=1,
        positional_encoding_temperature=10000,
        num_feature_levels=3,
        decoder_hidden_dim=256,
        decoder_num_heads=8,
        decoder_num_layers=6,
        decoder_activation="relu",
        decoder_dim_feedforward=2048,
        decoder_num_points=4,
        decoder_dropout=0.0,
        eval_size=None,
        learn_initial_query=False,
        cache_size=100,
        is_encoder_decoder=True,
        **kwargs,
    ):
        if use_timm_backbone:
            if backbone_config is None:
                backbone_kwargs = {
                    "out_indices": [1, 2, 3],
                    "img_size": image_size,
                    "always_partition": True,
                }
        elif backbone_config is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `swin` vision config.")
            backbone_config = CONFIG_MAPPING["swin"](
                window_size=7,
                image_size=image_size,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                out_indices=[2, 3, 4],
            )
        elif isinstance(backbone_config, dict):
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

        if text_config is None:
            logger.info(
                "`text_config` is `None`. Initializing the config with the default `clip_text_model` text config."
            )
            text_config = CONFIG_MAPPING["clip_text_model"]()
        elif isinstance(text_config, dict):
            text_model_type = text_config.get("model_type")
            text_config = CONFIG_MAPPING[text_model_type](**text_config)

        if class_distance_type not in ["cosine", "dot"]:
            raise ValueError(
                f"Invalid `class_distance_type`. It should be either `cosine` or `dot`, but got {class_distance_type}."
            )

        self.text_config = text_config
        self.backbone_config = backbone_config
        self.use_timm_backbone = use_timm_backbone
        self.backbone = backbone
        self.backbone_kwargs = backbone_kwargs
        self.use_pretrained_backbone = use_pretrained_backbone
        self.apply_layernorm_after_vision_backbone = apply_layernorm_after_vision_backbone
        self.image_size = image_size
        self.disable_custom_kernels = disable_custom_kernels
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps
        self.init_std = init_std
        self.text_projection_in_dim = text_projection_in_dim
        self.text_projection_out_dim = text_projection_out_dim
        self.task_encoder_hidden_dim = task_encoder_hidden_dim
        self.class_embed_dim = class_embed_dim
        self.class_distance_type = class_distance_type
        self.num_queries = num_queries
        self.csp_activation = csp_activation
        self.conv_norm_activation = conv_norm_activation
        self.encoder_feedforward_activation = encoder_feedforward_activation
        self.encoder_feedforward_dropout = encoder_feedforward_dropout
        self.encoder_dropout = encoder_dropout
        self.hidden_expansion = hidden_expansion
        self.vision_features_channels = vision_features_channels
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_in_channels = encoder_in_channels
        self.encoder_projection_indices = encoder_projection_indices
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_dim_feedforward = encoder_dim_feedforward
        self.encoder_layers = encoder_layers
        self.positional_encoding_temperature = positional_encoding_temperature
        self.num_feature_levels = num_feature_levels
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_num_heads = decoder_num_heads
        self.decoder_num_layers = decoder_num_layers
        self.decoder_activation = decoder_activation
        self.decoder_dim_feedforward = decoder_dim_feedforward
        self.decoder_num_points = decoder_num_points
        self.decoder_dropout = decoder_dropout
        self.eval_size = eval_size
        self.learn_initial_query = learn_initial_query
        self.cache_size = cache_size
        self.is_encoder_decoder = is_encoder_decoder

        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)


__all__ = ["OmDetTurboConfig"]

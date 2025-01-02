# coding=utf-8
# Copyright 2022 SHI Labs and The HuggingFace Inc. team. All rights reserved.
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
"""OneFormer model configuration"""

from typing import Dict, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import verify_backbone_config_arguments
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class OneFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OneFormerModel`]. It is used to instantiate a
    OneFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OneFormer
    [shi-labs/oneformer_ade20k_swin_tiny](https://huggingface.co/shi-labs/oneformer_ade20k_swin_tiny) architecture
    trained on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig`, *optional*, defaults to `SwinConfig`):
            The configuration of the backbone model.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `False`):
            Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
            library.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        ignore_value (`int`, *optional*, defaults to 255):
            Values to be ignored in GT label while calculating loss.
        num_queries (`int`, *optional*, defaults to 150):
            Number of object queries.
        no_object_weight (`float`, *optional*, defaults to 0.1):
            Weight for no-object class predictions.
        class_weight (`float`, *optional*, defaults to 2.0):
            Weight for Classification CE loss.
        mask_weight (`float`, *optional*, defaults to 5.0):
            Weight for binary CE loss.
        dice_weight (`float`, *optional*, defaults to 5.0):
            Weight for dice loss.
        contrastive_weight (`float`, *optional*, defaults to 0.5):
            Weight for contrastive loss.
        contrastive_temperature (`float`, *optional*, defaults to 0.07):
            Initial value for scaling the contrastive logits.
        train_num_points (`int`, *optional*, defaults to 12544):
            Number of points to sample while calculating losses on mask predictions.
        oversample_ratio (`float`, *optional*, defaults to 3.0):
            Ratio to decide how many points to oversample.
        importance_sample_ratio (`float`, *optional*, defaults to 0.75):
            Ratio of points that are sampled via importance sampling.
        init_std (`float`, *optional*, defaults to 0.02):
            Standard deviation for normal intialization.
        init_xavier_std (`float`, *optional*, defaults to 1.0):
            Standard deviation for xavier uniform initialization.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon for layer normalization.
        is_training (`bool`, *optional*, defaults to `False`):
            Whether to run in training or inference mode.
        use_auxiliary_loss (`bool`, *optional*, defaults to `True`):
            Whether to calculate loss using intermediate predictions from transformer decoder.
        output_auxiliary_logits (`bool`, *optional*, defaults to `True`):
            Whether to return intermediate predictions from transformer decoder.
        strides (`list`, *optional*, defaults to `[4, 8, 16, 32]`):
            List containing the strides for feature maps in the encoder.
        task_seq_len (`int`, *optional*, defaults to 77):
            Sequence length for tokenizing text list input.
        text_encoder_width (`int`, *optional*, defaults to 256):
            Hidden size for text encoder.
        text_encoder_context_length (`int`, *optional*, defaults to 77):
            Input sequence length for text encoder.
        text_encoder_num_layers (`int`, *optional*, defaults to 6):
            Number of layers for transformer in text encoder.
        text_encoder_vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size for tokenizer.
        text_encoder_proj_layers (`int`, *optional*, defaults to 2):
            Number of layers in MLP for project text queries.
        text_encoder_n_ctx (`int`, *optional*, defaults to 16):
            Number of learnable text context queries.
        conv_dim (`int`, *optional*, defaults to 256):
            Feature map dimension to map outputs from the backbone.
        mask_dim (`int`, *optional*, defaults to 256):
            Dimension for feature maps in pixel decoder.
        hidden_dim (`int`, *optional*, defaults to 256):
            Dimension for hidden states in transformer decoder.
        encoder_feedforward_dim (`int`, *optional*, defaults to 1024):
            Dimension for FFN layer in pixel decoder.
        norm (`str`, *optional*, defaults to `"GN"`):
            Type of normalization.
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of layers in pixel decoder.
        decoder_layers (`int`, *optional*, defaults to 10):
            Number of layers in transformer decoder.
        use_task_norm (`bool`, *optional*, defaults to `True`):
            Whether to normalize the task token.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in transformer layers in the pixel and transformer decoders.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability for pixel and transformer decoders.
        dim_feedforward (`int`, *optional*, defaults to 2048):
            Dimension for FFN layer in transformer decoder.
        pre_norm (`bool`, *optional*, defaults to `False`):
            Whether to normalize hidden states before attention layers in transformer decoder.
        enforce_input_proj (`bool`, *optional*, defaults to `False`):
            Whether to project hidden states in transformer decoder.
        query_dec_layers (`int`, *optional*, defaults to 2):
            Number of layers in query transformer.
        common_stride (`int`, *optional*, defaults to 4):
            Common stride used for features in pixel decoder.

    Examples:
    ```python
    >>> from transformers import OneFormerConfig, OneFormerModel

    >>> # Initializing a OneFormer shi-labs/oneformer_ade20k_swin_tiny configuration
    >>> configuration = OneFormerConfig()
    >>> # Initializing a model (with random weights) from the shi-labs/oneformer_ade20k_swin_tiny style configuration
    >>> model = OneFormerModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "oneformer"
    attribute_map = {"hidden_size": "hidden_dim"}

    def __init__(
        self,
        backbone_config: Optional[Dict] = None,
        backbone: Optional[str] = None,
        use_pretrained_backbone: bool = False,
        use_timm_backbone: bool = False,
        backbone_kwargs: Optional[Dict] = None,
        ignore_value: int = 255,
        num_queries: int = 150,
        no_object_weight: int = 0.1,
        class_weight: float = 2.0,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        contrastive_weight: float = 0.5,
        contrastive_temperature: float = 0.07,
        train_num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        init_std: float = 0.02,
        init_xavier_std: float = 1.0,
        layer_norm_eps: float = 1e-05,
        is_training: bool = False,
        use_auxiliary_loss: bool = True,
        output_auxiliary_logits: bool = True,
        strides: Optional[list] = [4, 8, 16, 32],
        task_seq_len: int = 77,
        text_encoder_width: int = 256,
        text_encoder_context_length: int = 77,
        text_encoder_num_layers: int = 6,
        text_encoder_vocab_size: int = 49408,
        text_encoder_proj_layers: int = 2,
        text_encoder_n_ctx: int = 16,
        conv_dim: int = 256,
        mask_dim: int = 256,
        hidden_dim: int = 256,
        encoder_feedforward_dim: int = 1024,
        norm: str = "GN",
        encoder_layers: int = 6,
        decoder_layers: int = 10,
        use_task_norm: bool = True,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        pre_norm: bool = False,
        enforce_input_proj: bool = False,
        query_dec_layers: int = 2,
        common_stride: int = 4,
        **kwargs,
    ):
        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is unset. Initializing the config with the default `Swin` backbone.")
            backbone_config = CONFIG_MAPPING["swin"](
                image_size=224,
                num_channels=3,
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                drop_path_rate=0.3,
                use_absolute_embeddings=False,
                out_features=["stage1", "stage2", "stage3", "stage4"],
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

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs
        self.ignore_value = ignore_value
        self.num_queries = num_queries
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.layer_norm_eps = layer_norm_eps
        self.is_training = is_training
        self.use_auxiliary_loss = use_auxiliary_loss
        self.output_auxiliary_logits = output_auxiliary_logits
        self.strides = strides
        self.task_seq_len = task_seq_len
        self.text_encoder_width = text_encoder_width
        self.text_encoder_context_length = text_encoder_context_length
        self.text_encoder_num_layers = text_encoder_num_layers
        self.text_encoder_vocab_size = text_encoder_vocab_size
        self.text_encoder_proj_layers = text_encoder_proj_layers
        self.text_encoder_n_ctx = text_encoder_n_ctx
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim
        self.encoder_feedforward_dim = encoder_feedforward_dim
        self.norm = norm
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.use_task_norm = use_task_norm
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.enforce_input_proj = enforce_input_proj
        self.query_dec_layers = query_dec_layers
        self.common_stride = common_stride
        self.num_hidden_layers = decoder_layers

        super().__init__(**kwargs)

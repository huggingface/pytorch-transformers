# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""UperNet model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import verify_backbone_config_arguments
from ..auto.configuration_auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class UperNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`UperNetForSemanticSegmentation`]. It is used to
    instantiate an UperNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the UperNet
    [openmmlab/upernet-convnext-tiny](https://huggingface.co/openmmlab/upernet-convnext-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `ResNetConfig()`):
            The configuration of the backbone model.
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
        hidden_size (`int`, *optional*, defaults to 512):
            The number of hidden units in the convolutional layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        pool_scales (`Tuple[int]`, *optional*, defaults to `[1, 2, 3, 6]`):
            Pooling scales used in Pooling Pyramid Module applied on the last feature map.
        use_auxiliary_head (`bool`, *optional*, defaults to `True`):
            Whether to use an auxiliary head during training.
        auxiliary_loss_weight (`float`, *optional*, defaults to 0.4):
            Weight of the cross-entropy loss of the auxiliary head.
        auxiliary_channels (`int`, *optional*, defaults to 256):
            Number of channels to use in the auxiliary head.
        auxiliary_num_convs (`int`, *optional*, defaults to 1):
            Number of convolutional layers to use in the auxiliary head.
        auxiliary_concat_input (`bool`, *optional*, defaults to `False`):
            Whether to concatenate the output of the auxiliary head with the input before the classification layer.
        loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function.

    Examples:

    ```python
    >>> from transformers import UperNetConfig, UperNetForSemanticSegmentation

    >>> # Initializing a configuration
    >>> configuration = UperNetConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = UperNetForSemanticSegmentation(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "upernet"

    def __init__(
        self,
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        backbone_kwargs=None,
        hidden_size=512,
        initializer_range=0.02,
        pool_scales=[1, 2, 3, 6],
        use_auxiliary_head=True,
        auxiliary_loss_weight=0.4,
        auxiliary_in_channels=384,
        auxiliary_channels=256,
        auxiliary_num_convs=1,
        auxiliary_concat_input=False,
        loss_ignore_index=255,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
            backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage1", "stage2", "stage3", "stage4"])
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
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.pool_scales = pool_scales
        self.use_auxiliary_head = use_auxiliary_head
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.auxiliary_in_channels = auxiliary_in_channels
        self.auxiliary_channels = auxiliary_channels
        self.auxiliary_num_convs = auxiliary_num_convs
        self.auxiliary_concat_input = auxiliary_concat_input
        self.loss_ignore_index = loss_ignore_index


__all__ = ["UperNetConfig"]

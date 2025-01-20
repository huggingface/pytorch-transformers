# coding=utf-8
# Copyright The HuggingFace Inc. team. All rights reserved.
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
""" Fast model configuration"""
from transformers import CONFIG_MAPPING, PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

FAST_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "fast_base_tt_800_finetune_ic17mlt": (
        "https://huggingface.co/Raghavan/fast_base_tt_800_finetune_ic17mlt/raw/main/config.json"
    ),
}


class FastConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastForSceneTextRecognition`]. It is used to
    instantiate a FastForSceneTextRecognition model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastForSceneTextRecognition.
    [Raghavan/fast_base_tt_800_finetune_ic17mlt](https://huggingface.co/Raghavan/fast_base_tt_800_finetune_ic17mlt)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_timm_backbone (`bool`, *optional*, defaults to `True`):
            Whether or not to use the `timm` library for the backbone. If set to `False`, will use the [`AutoBackbone`]
            API.
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
            case it will default to `ResNetConfig()`.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        neck_in_channels (`List[int]`, *optional*, defaults to `[64, 128, 256, 512]`):
            Denotes the in channels of FASTRepConvLayer in neck module.
        neck_out_channels (`List[int]`, *optional*, defaults to `[128, 128, 128, 128]`):
            Denotes the out channels of FASTRepConvLayer in neck module. Should be of same length of `neck_in_channels`
        neck_kernel_size (`List[int]`, *optional*, defaults to `[[3, 3], [3, 3], [3, 3], [3, 3]]`):
            Denotes the kernel_size of FASTRepConvLayer in neck module. Should be of same length of `neck_in_channels`
        neck_stride (`List[int]`, *optional*, defaults to `[1, 1, 1, 1]`):
            Denotes the neck_stride of FASTRepConvLayer in neck module. Should be of same length of `neck_in_channels`
        head_pooling_size (`int`, *optional*, defaults to 9):
            Denotes the pooling size of head layer
        head_dropout_ratio (`int`, *optional*, defaults to 0):
            Denotes the dropout ratio used in dropout layer of head layer..
        head_conv_in_channels (`int`, *optional*, defaults to 512):
            Denotes the in channels of first conv layer in head layer.
        head_conv_out_channels (`int`, *optional*, defaults to 128):
            Denotes the out channels of first conv layer in head layer.
        head_conv_kernel_size (`List[int]`, *optional*, defaults to `[3, 3]`):
            Denotes the conv kernel size of first conv layer in head layer.
        head_conv_stride (`int`, *optional*, defaults to 1):
            Denotes the conv stride of first conv layer in head layer.
        head_final_kernel_size (`int`, *optional*, defaults to 1):
            Denotes the conv kernel size of final conv layer in head layer.
        head_final_stride (`int`, *optional*, defaults to 1):
            Denotes the conv stride of final conv layer in head layer.
        head_final_bias (`bool`, *optional*, defaults to `False`):
            Denotes the conv bais of final conv layer in head layer.
        head_final_in_channels (`int`, *optional*, defaults to 128):
            Denotes the in channels of final conv layer in head layer.
        head_final_out_channels (`int`, *optional*, defaults to 5):
            Denotes the out channels of final conv layer in head layer.
        backbone (`str`, *optional*, defaults to `"resnet50"`):
            Name of convolutional backbone to use in case `use_timm_backbone` = `True`. Supports any convolutional
            backbone from the timm package. For a list of all available models, see [this
            page](https://rwightman.github.io/pytorch-image-models/#load-a-pretrained-model).
        use_pretrained_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use pretrained weights for the backbone. Only supported when `use_timm_backbone` = `True`.
        dilation (`bool`, *optional*, defaults to `False`):
            Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when
            `use_timm_backbone` = `True`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.

    Examples:

    ```python
    >>> from transformers import FastConfig, FastForSceneTextRecognition

    >>> # Initializing a Fast Config
    >>> configuration = FastConfig()

    >>> # Initializing a model (with random weights)
    >>> model = FastForSceneTextRecognition(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    r"""
    [Raghavan/fast_base_tt_800_finetune_ic17mlt](https://huggingface.co/Raghavan/fast_base_tt_800_finetune_ic17mlt)
    """

    def __init__(
        self,
        use_timm_backbone=True,
        backbone_config=None,
        num_channels=3,
        neck_in_channels=[64, 128, 256, 512],
        neck_out_channels=[128, 128, 128, 128],
        neck_kernel_size=[[3, 3], [3, 3], [3, 3], [3, 3]],
        neck_stride=[1, 1, 1, 1],
        head_pooling_size=9,
        head_dropout_ratio=0,
        head_conv_in_channels=512,
        head_conv_out_channels=128,
        head_conv_kernel_size=[3, 3],
        head_conv_stride=1,
        head_final_kernel_size=1,
        head_final_stride=1,
        head_final_bias=False,
        head_final_in_channels=128,
        head_final_out_channels=5,
        backbone="resnet50",
        use_pretrained_backbone=True,
        dilation=False,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if backbone_config is not None and use_timm_backbone:
            raise ValueError("You can't specify both `backbone_config` and `use_timm_backbone`.")

        if not use_timm_backbone:
            if backbone_config is None:
                logger.info(
                    "`backbone_config` is `None`. Initializing the config with the default `TextNet` backbone."
                )
                backbone_config = CONFIG_MAPPING["textnet"](out_features=["stage1", "stage2", "stage3", "stage4"])
                
            elif isinstance(backbone_config, dict):
                backbone_model_type = backbone_config.get("model_type")
                config_class = CONFIG_MAPPING[backbone_model_type]
                backbone_config = config_class.from_dict(backbone_config)
            # set timm attributes to None
            dilation, backbone, use_pretrained_backbone = None, None, None

        self.use_timm_backbone = use_timm_backbone
        self.backbone_config = backbone_config
        self.num_channels = num_channels

        self.neck_in_channels = neck_in_channels
        self.neck_out_channels = neck_out_channels
        self.neck_kernel_size = neck_kernel_size
        self.neck_stride = neck_stride

        self.head_pooling_size = head_pooling_size
        self.head_dropout_ratio = head_dropout_ratio

        self.head_conv_in_channels = head_conv_in_channels
        self.head_conv_out_channels = head_conv_out_channels
        self.head_conv_kernel_size = head_conv_kernel_size
        self.head_conv_stride = head_conv_stride

        self.head_final_kernel_size = head_final_kernel_size
        self.head_final_stride = head_final_stride
        self.head_final_bias = head_final_bias
        self.head_final_in_channels = head_final_in_channels
        self.head_final_out_channels = head_final_out_channels

        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.dilation = dilation

        self.initializer_range = initializer_range

    @classmethod
    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`FastConfig`] (or a derived class) from a pre-trained backbone model configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration.
        Returns:
            [`FastConfig`]: An instance of a configuration object
        """
        return cls(backbone_config=backbone_config, **kwargs)

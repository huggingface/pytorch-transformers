# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""DepthPro model configuration"""

import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING
from ..dinov2.configuration_dinov2 import Dinov2Config


logger = logging.get_logger(__name__)


class DepthProConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DepthProModel`]. It is used to instantiate a
    DepthPro model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DepthPro
    [apple/DepthPro](https://huggingface.co/apple/DepthPro) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers. Should match hidden_size of backbone.
        fusion_hidden_size (`int`, *optional*, defaults to 256):
            The number of channels before fusion.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        patch_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each patch. This is also the image_size for backbone model.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        intermediate_hook_ids (`List[int]`, *optional*, defaults to `[11, 5]`):
            Indices of the intermediate hidden states from the patch encoder to use for fusion.
        intermediate_feature_dims (`List[int]`, *optional*, defaults to `[256, 256]`):
            Hidden state dimensions during upsampling for each intermediate hidden state in `intermediate_hook_ids`.
        scaled_images_ratios (`List[float]`, *optional*, defaults to `[0.25, 0.5, 1]`):
            Ratios of scaled images to be used by the patch encoder.
        scaled_images_overlap_ratios (`List[float]`, *optional*, defaults to `[0.0, 0.5, 0.25]`):
            Overlap ratios between patches for each scaled image in `scaled_images_ratios`.
        scaled_images_feature_dims (`List[int]`, *optional*, defaults to `[1024, 1024, 512]`):
            Hidden state dimensions during upsampling for each scaled image in `scaled_images_ratios`.
        merge_padding_value (`int`, *optional*, defaults to 3):
            When merging smaller patches back to the image size, overlapping sections of this size are removed.
        use_batch_norm_in_fusion_residual (`bool`, *optional*, defaults to `False`):
            Whether to use batch normalization in the pre-activate residual units of the fusion blocks.
        use_bias_in_fusion_residual (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the pre-activate residual units of the fusion blocks.
        use_fov_model (`bool`, *optional*, defaults to `False`):
            Whether to use `DepthProFOVModel` to generate the field of view.
        num_fov_head_layers (`int`, *optional*, defaults to 2):
            Number of convolution layers in the head of `DepthProFOVModel`.
        backbone_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*):
            The configuration of the backbone model, which is loaded using the [`AutoModel`] API.
            By default, Dinov2 model is used as backbone.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoModel when loading from the backbone_config.

    Example:

    ```python
    >>> from transformers import DepthProConfig, DepthProModel

    >>> # Initializing a DepthPro apple/DepthPro style configuration
    >>> configuration = DepthProConfig()

    >>> # Initializing a model (with random weights) from the apple/DepthPro style configuration
    >>> model = DepthProModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "depth_pro"

    def __init__(
        self,
        hidden_size=1024,
        fusion_hidden_size=256,
        num_hidden_layers=24,
        num_attention_heads=16,
        initializer_range=0.02,
        patch_size=384,
        num_channels=3,
        intermediate_hook_ids=[11, 5],
        intermediate_feature_dims=[256, 256],
        scaled_images_ratios=[0.25, 0.5, 1],
        scaled_images_overlap_ratios=[0.0, 0.5, 0.25],
        scaled_images_feature_dims=[1024, 1024, 512],
        merge_padding_value=3,
        use_batch_norm_in_fusion_residual=False,
        use_bias_in_fusion_residual=True,
        use_fov_model=False,
        num_fov_head_layers=2,
        backbone_config=None,
        backbone_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # scaled_images_ratios is sorted
        if scaled_images_ratios != sorted(scaled_images_ratios):
            raise ValueError(
                f"Values in scaled_images_ratios={scaled_images_ratios} " "should be sorted from low to high"
            )

        # scaled_images_ratios, scaled_images_overlap_ratios, scaled_images_feature_dims should be consistent
        if not (len(scaled_images_ratios) == len(scaled_images_overlap_ratios) == len(scaled_images_feature_dims)):
            raise ValueError(
                f"len(scaled_images_ratios)={len(scaled_images_ratios)} and "
                f"len(scaled_images_overlap_ratios)={len(scaled_images_overlap_ratios)} and "
                f"len(scaled_images_feature_dims)={len(scaled_images_feature_dims)}, "
                f"should match in config."
            )

        # intermediate_hook_ids, intermediate_feature_dims should be consistent
        if not (len(intermediate_hook_ids) == len(intermediate_feature_dims)):
            raise ValueError(
                f"len(intermediate_hook_ids)={len(intermediate_hook_ids)} and "
                f"len(intermediate_feature_dims)={len(intermediate_feature_dims)}, "
                f"should match in config."
            )

        # fusion_hidden_size should be consistent with num_fov_head_layers
        if fusion_hidden_size // 2**num_fov_head_layers == 0:
            raise ValueError(
                f"fusion_hidden_size={fusion_hidden_size} should be consistent with num_fov_head_layers={num_fov_head_layers} "
                "i.e fusion_hidden_size // 2**num_fov_head_layers > 0"
            )

        self.hidden_size = hidden_size
        self.fusion_hidden_size = fusion_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.use_batch_norm_in_fusion_residual = use_batch_norm_in_fusion_residual
        self.use_bias_in_fusion_residual = use_bias_in_fusion_residual
        self.use_fov_model = use_fov_model
        self.num_fov_head_layers = num_fov_head_layers
        self.intermediate_hook_ids = intermediate_hook_ids
        self.intermediate_feature_dims = intermediate_feature_dims
        self.scaled_images_ratios = scaled_images_ratios
        self.scaled_images_overlap_ratios = scaled_images_overlap_ratios
        self.scaled_images_feature_dims = scaled_images_feature_dims
        self.merge_padding_value = merge_padding_value

        self.backbone_config = self._create_backbone_config(backbone_config)
        self.backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs

    def _create_backbone_config(self, backbone_config=None):
        """"""
        # for compatibility between DepthPro and the backbone model
        # make sure these parameters of backbone model are same as DepthPro
        matching_config = [
            "hidden_size",
            "num_channels",
            "num_hidden_layers",
            "num_attention_heads",
            "_attn_implementation",
        ]
        matching_config_dict = {k: getattr(self, k) for k in matching_config}

        if backbone_config is None:
            # use Dinov2 config by default
            logger.info("Initializing the config with the default Dinov2 backbone.")
            backbone_config = Dinov2Config(
                patch_size=16,
                image_size=self.patch_size,
                **matching_config_dict,
            )

        elif isinstance(backbone_config, dict):
            assert backbone_config.get("model_type") is not None
            logger.info(f"Initializing the config with a {backbone_config.get('model_type')} backbone.")
            backbone_config.update(
                {
                    "image_size": self.patch_size,
                    **matching_config_dict,
                }
            )
            config_class = CONFIG_MAPPING[backbone_config.get("model_type")]
            backbone_config = config_class.from_dict(backbone_config)

        elif isinstance(backbone_config, PretrainedConfig):
            backbone_config = backbone_config

        else:
            raise ValueError(
                f"backbone_config must be a dictionary or a `PretrainedConfig`, got {backbone_config.__class__}."
            )

        # validate the config compatibility between DepthPro and the backbone model
        if self.patch_size != backbone_config.image_size:
            # patches of input image are created of size patch_size x patch_size
            # then these patches are given to the backbone model as input images
            raise ValueError(
                f"patch_size={self.patch_size} should be equal to backbone_config.image_size={backbone_config.image_size}."
            )
        for key in matching_config:
            config_value = getattr(self, key)
            backbone_config_value = getattr(backbone_config, key)
            if config_value != backbone_config_value:
                raise ValueError(
                    f"{key}={config_value} should be equal to backbone_config.{key}={backbone_config_value}."
                )

        return backbone_config

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        if output["backbone_config"] is not None:
            output["backbone_config"] = self.backbone_config.to_dict()

        output["model_type"] = self.__class__.model_type
        return output


__all__ = ["DepthProConfig"]

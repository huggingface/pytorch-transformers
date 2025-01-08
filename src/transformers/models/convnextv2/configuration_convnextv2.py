# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""ConvNeXTV2 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices


logger = logging.get_logger(__name__)


class ConvNextV2Config(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ConvNextV2Model`]. It is used to instantiate an
    ConvNeXTV2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ConvNeXTV2
    [facebook/convnextv2-tiny-1k-224](https://huggingface.co/facebook/convnextv2-tiny-1k-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        patch_size (`int`, *optional*, defaults to 4):
            Patch size to use in the patch embedding layer.
        num_stages (`int`, *optional*, defaults to 4):
            The number of stages in the model.
        hidden_sizes (`List[int]`, *optional*, defaults to `[96, 192, 384, 768]`):
            Dimensionality (hidden size) at each stage.
        depths (`List[int]`, *optional*, defaults to `[3, 3, 9, 3]`):
            Depth (number of blocks) for each stage.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The drop rate for stochastic depth.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.

    Example:
    ```python
    >>> from transformers import ConvNeXTV2Config, ConvNextV2Model

    >>> # Initializing a ConvNeXTV2 convnextv2-tiny-1k-224 style configuration
    >>> configuration = ConvNeXTV2Config()

    >>> # Initializing a model (with random weights) from the convnextv2-tiny-1k-224 style configuration
    >>> model = ConvNextV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "convnextv2"

    def __init__(
        self,
        num_channels=3,
        patch_size=4,
        num_stages=4,
        hidden_sizes=None,
        depths=None,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        drop_path_rate=0.0,
        image_size=224,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_stages = num_stages
        self.hidden_sizes = [96, 192, 384, 768] if hidden_sizes is None else hidden_sizes
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )


__all__ = ["ConvNextV2Config"]

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
"""BiT model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices


logger = logging.get_logger(__name__)


class BitConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BitModel`]. It is used to instantiate an BiT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BiT
    [google/bit-50](https://huggingface.co/google/bit-50) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embedding_size (`int`, *optional*, defaults to 64):
            Dimensionality (hidden size) for the embedding layer.
        hidden_sizes (`List[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`):
            Dimensionality (hidden size) at each stage.
        depths (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            Depth (number of layers) for each stage.
        layer_type (`str`, *optional*, defaults to `"preactivation"`):
            The layer to use, it can be either `"preactivation"` or `"bottleneck"`.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
            are supported.
        global_padding (`str`, *optional*):
            Padding strategy to use for the convolutional layers. Can be either `"valid"`, `"same"`, or `None`.
        num_groups (`int`, *optional*, defaults to 32):
            Number of groups used for the `BitGroupNormActivation` layers.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The drop path rate for the stochastic depth.
        embedding_dynamic_padding (`bool`, *optional*, defaults to `False`):
            Whether or not to make use of dynamic padding for the embedding layer.
        output_stride (`int`, *optional*, defaults to 32):
            The output stride of the model.
        width_factor (`int`, *optional*, defaults to 1):
            The width factor for the model.
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
    >>> from transformers import BitConfig, BitModel

    >>> # Initializing a BiT bit-50 style configuration
    >>> configuration = BitConfig()

    >>> # Initializing a model (with random weights) from the bit-50 style configuration
    >>> model = BitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "bit"
    layer_types = ["preactivation", "bottleneck"]
    supported_padding = ["SAME", "VALID"]

    def __init__(
        self,
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048],
        depths=[3, 4, 6, 3],
        layer_type="preactivation",
        hidden_act="relu",
        global_padding=None,
        num_groups=32,
        drop_path_rate=0.0,
        embedding_dynamic_padding=False,
        output_stride=32,
        width_factor=1,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")
        if global_padding is not None:
            if global_padding.upper() in self.supported_padding:
                global_padding = global_padding.upper()
            else:
                raise ValueError(f"Padding strategy {global_padding} not supported")
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.global_padding = global_padding
        self.num_groups = num_groups
        self.drop_path_rate = drop_path_rate
        self.embedding_dynamic_padding = embedding_dynamic_padding
        self.output_stride = output_stride
        self.width_factor = width_factor

        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )


__all__ = ["BitConfig"]

# coding=utf-8
# Copyright 2024 the Fast authors and HuggingFace Inc. team.  All rights reserved.
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
"""TextNet model configuration"""

from transformers import PretrainedConfig
from transformers.utils import logging
from transformers.utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices


logger = logging.get_logger(__name__)


class TextNetConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TextNextModel`]. It is used to instantiate a
    TextNext model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [czczup/textnet-base](https://huggingface.co/czczup/textnet-base). Configuration objects inherit from
    [`PretrainedConfig`] and can be used to control the model outputs.Read the documentation from [`PretrainedConfig`]
    for more information.

    Args:
        stem_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the initial convolution layer.
        stem_stride (`int`, *optional*, defaults to 2):
            The stride for the initial convolution layer.
        stem_num_channels (`int`, *optional*, defaults to 3):
            The num of channels in input for the initial convolution layer.
        stem_out_channels (`int`, *optional*, defaults to 64):
            The num of channels in out for the initial convolution layer.
        stem_act_func (`str`, *optional*, defaults to `"relu"`):
            The activation function for the initial convolution layer.
        image_size (`Tuple[int, int]`, *optional*, defaults to `[640, 640]`):
            The size (resolution) of each image.
        conv_layer_kernel_sizes (`List[List[List[int]]]`, *optional*):
            A list of stage-wise kernel sizes. If `None`, defaults to:
            `[[[3, 3], [3, 3], [3, 3]], [[3, 3], [1, 3], [3, 3], [3, 1]], [[3, 3], [3, 3], [3, 1], [1, 3]], [[3, 3], [3, 1], [1, 3], [3, 3]]]`.
        conv_layer_strides (`List[List[int]]`, *optional*):
            A list of stage-wise strides. If `None`, defaults to:
            `[[1, 2, 1], [2, 1, 1, 1], [2, 1, 1, 1], [2, 1, 1, 1]]`.
        hidden_sizes (`List[int]`, *optional*, defaults to `[64, 64, 128, 256, 512]`):
            Dimensionality (hidden size) at each stage.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage.

    Examples:

    ```python
    >>> from transformers import TextNetConfig, TextNetBackbone

    >>> # Initializing a TextNetConfig
    >>> configuration = TextNetConfig()

    >>> # Initializing a model (with random weights)
    >>> model = TextNetBackbone(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "textnet"

    def __init__(
        self,
        stem_kernel_size=3,
        stem_stride=2,
        stem_num_channels=3,
        stem_out_channels=64,
        stem_act_func="relu",
        image_size=[640, 640],
        conv_layer_kernel_sizes=None,
        conv_layer_strides=None,
        hidden_sizes=[64, 64, 128, 256, 512],
        batch_norm_eps=1e-5,
        initializer_range=0.02,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if conv_layer_kernel_sizes is None:
            conv_layer_kernel_sizes = [
                [[3, 3], [3, 3], [3, 3]],
                [[3, 3], [1, 3], [3, 3], [3, 1]],
                [[3, 3], [3, 3], [3, 1], [1, 3]],
                [[3, 3], [3, 1], [1, 3], [3, 3]],
            ]
        if conv_layer_strides is None:
            conv_layer_strides = [[1, 2, 1], [2, 1, 1, 1], [2, 1, 1, 1], [2, 1, 1, 1]]

        self.stem_kernel_size = stem_kernel_size
        self.stem_stride = stem_stride
        self.stem_num_channels = stem_num_channels
        self.stem_out_channels = stem_out_channels
        self.stem_act_func = stem_act_func

        self.image_size = image_size
        self.conv_layer_kernel_sizes = conv_layer_kernel_sizes
        self.conv_layer_strides = conv_layer_strides

        self.initializer_range = initializer_range
        self.hidden_sizes = hidden_sizes
        self.batch_norm_eps = batch_norm_eps

        self.depths = [len(layer) for layer in self.conv_layer_kernel_sizes]
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, 5)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )


__all__ = ["TextNetConfig"]

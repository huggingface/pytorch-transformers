# coding=utf-8
# Copyright 2024 S-Lab, Nanyang Technological University and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the S-Lab License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/sczhou/ProPainter/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ProPainter model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class ProPainterConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ProPainterModel`]. It is used to instantiate a ProPainter
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ProPainter [ruffy369/propainter](https://huggingface.co/ruffy369/propainter)
    architecture.

    The original configuration and code can be referred from [here](https://github.com/sczhou/ProPainter)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_local_frames_propainter (`int`, *optional*, defaults to 10):
            The number of local frames used in the ProPainter inpaint_generator network.
        flow_weight_flow_complete_net (`float`, *optional*, defaults to 0.25):
            The weight of the flow loss in the flow completion network.
        hole_weight (`float`, *optional*, defaults to 1.0):
            The weight for the hole loss.
        valid_weight (`float`, *optional*, defaults to 1.0):
            The weight for the valid region loss.
        adversarial_weight (`float`, *optional*, defaults to 0.01):
            The weight of the adversarial loss in the ProPainter inpaint_generator network.
        gan_loss (`str`, *optional*, defaults to `"hinge"`):
            The type of GAN loss to use. Options are `"hinge"`, `"nsgan"`, or `"lsgan"`.
        perceptual_weight (`float`, *optional*, defaults to 0.0):
            The weight of the perceptual loss.
        interp_mode (`str`, *optional*, defaults to `"nearest"`):
            The interpolation mode used for resizing. Options are `"nearest"`, `"bilinear"`, `"bicubic"`.
        ref_stride (`int`, *optional*, defaults to 10):
            The stride for reference frames in the ProPainter inpaint_generator network.
        neighbor_length (`int`, *optional*, defaults to 10):
            The length of neighboring frames considered in the ProPainter inpaint_generator network.
        subvideo_length (`int`, *optional*, defaults to 80):
            The length of sub-videos for training.
        correlation_levels (`int`, *optional*, defaults to 4):
            The number of correlation levels used in the RAFT optical flow model.
        correlation_radius (`int`, *optional*, defaults to 4):
            The radius of the correlation window used in the RAFT optical flow model.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to layers in the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
        raft_iter (`int`, *optional*, defaults to 20):
            The number of iterations for RAFT model updates.
        num_channels (`int`, *optional*, defaults to 128):
            The number of channels in the feature maps.
        hidden_size (`int`, *optional*, defaults to 512):
            The dimensionality of hidden layers.
        kernel_size (`List[int]`, *optional*, defaults to `[7, 7]`):
            The size of the convolution kernels.
        kernel_size_3d (`List[int]`, *optional*, defaults to `[1, 3, 3]`):
            The size of the 3d convolution kernels.
        kernel_size_3d_discriminator (`List[int]`, *optional*, defaults to `[3, 5, 5]`):
            The size of the 3d convolution kernels for discriminator modules used to calculate losses.
        padding_inpaint_generator (`List[int]`, *optional*, defaults to `[3, 3]`):
            The padding size for the convolution kernels in inpaint_generator module.
        padding (`int`, *optional*, defaults to 1):
            The padding size for the convolution kernels.
        conv2d_stride (`List[int]`, *optional*, defaults to `[3, 3]`):
            The stride for the convolution kernels.
        conv3d_stride (`List[int]`, *optional*, defaults to `[1, 1, 1]`):
            The stride for the 3d convolution kernels.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            The number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 4):
            The number of attention heads for each attention layer in the model.
        window_size (`List[int]`, *optional*, defaults to `[5, 9]`):
            The size of the sliding window for attention operations.
        pool_size (`List[int]`, *optional*, defaults to `[4, 4]`):
            The size of the pooling layers in the model.
        use_discriminator (`bool`, *optional*, defaults to `True`):
            Whether to enable discriminator.
        in_channels (`List[int]`, *optional*, defaults to `[64, 64, 96]`):
            The number of input channels at different levels of the model.
        channels (`List[int]`, *optional*, defaults to `[64, 96, 128]`):
            The number of channels at different levels of the model.
        multi_level_conv_stride (`List[int]`, *optional*, defaults to `[1, 2, 2]`):
            The stride values for the convolution layers at different levels of the model.
        norm_fn (`List[str]`, *optional*, defaults to `['batch', 'group', 'instance', 'none']`):
            The type of normalization to use in the model. Available options are:
            - `"batch"`: Use Batch Normalization.
            - `"group"`: Use Group Normalization.
            - `"instance"`: Use Instance Normalization.
            - `"none"`: No normalization will be applied.
        patch_size (`int`, *optional*, defaults to 3):
            The kernel size of the 2D convolution layer.
        negative_slope_default (`float`, *optional*, defaults to 0.2):
            Controls the slope for negative inputs in LeakyReLU. This is  the oneused at most of the places in different module classes
        negative_slope_1 (`float`, *optional*, defaults to 0.1):
            Controls the slope for negative inputs in LeakyReLU. Used in few certain modules.
        negative_slope_2 (`float`, *optional*, defaults to 0.01):
            Controls the slope for negative inputs in LeakyReLU. Used in few certain modules.
        group (`List[int]`, *optional*, defaults to `[1, 2, 4, 8, 1]`):
            Specifies the number of groups for feature aggregation at different layers in the ProPainterEncoder.
        kernel_size_3d_downsample (`List[int]`, *optional*, defaults to `[1, 5, 5]`):
            Kernel size for 3D downsampling layers along depth, height, and width.
        intermediate_dilation_padding (`List[Tuple[int, int, int]]`, *optional*, defaults to `[(0, 3, 3), (0, 2, 2), (0, 1, 1)]`):
            Padding values for intermediate dilation layers (depth, height, width).
        padding_downsample (`List[int]`, *optional*, defaults to `[0, 2, 2]`):
            Padding for downsampling layers along depth, height, and width.
        padding_mode (`str`, *optional*, defaults to `"replicate"`):
            Padding mode for convolution layers (default: "replicate").
        intermediate_dilation_levels (`List[Tuple[int, int, int]]`, *optional*, defaults to `[(1, 3, 3), (1, 2, 2), (1, 1, 1)]`):
            Dilation rates for intermediate layers (depth, height, width).
        num_channels_img_prop_module (`int`, *optional*, defaults to 3):
            The number of channels for image propagation module in ProPainterBidirectionalPropagationInPaint module.
        deform_groups (`int`, *optional*, defaults to `16`):
            Specifies the number of deformable group partitions in the deformable convolution layer.

    Example:

    ```python
    >>> from transformers import ProPainterConfig, ProPainterModel

    >>> # Initializing a ProPainter style configuration
    >>> configuration = ProPainterConfig()

    >>> # Initializing a model (with random weights) from the propainter style configuration
    >>> model = ProPainterModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "propainter"

    def __init__(
        self,
        num_local_frames_propainter=10,
        flow_weight_flow_complete_net=0.25,
        hole_weight=1.0,
        valid_weight=1.0,
        adversarial_weight=0.01,
        gan_loss="hinge",
        perceptual_weight=0.0,
        interp_mode="nearest",
        ref_stride=10,
        neighbor_length=10,
        subvideo_length=80,
        correlation_levels=4,
        correlation_radius=4,
        dropout=0.0,
        initializer_range=0.02,
        raft_iter=20,
        num_channels=128,
        hidden_size=512,
        kernel_size=[7, 7],
        kernel_size_3d=[1, 3, 3],
        kernel_size_3d_discriminator=[3, 5, 5],
        padding_inpaint_generator=[3, 3],
        padding=1,
        conv2d_stride=[3, 3],
        conv3d_stride=[1, 1, 1],
        num_hidden_layers=8,
        num_attention_heads=4,
        window_size=[5, 9],
        pool_size=[4, 4],
        use_discriminator=True,
        in_channels=[64, 64, 96],
        channels=[64, 96, 128],
        multi_level_conv_stride=[1, 2, 2],
        norm_fn=["batch", "group", "instance", "none"],
        patch_size=3,
        negative_slope_default=0.2,
        negative_slope_1=0.1,
        negative_slope_2=0.01,
        group=[1, 2, 4, 8, 1],
        kernel_size_3d_downsample=[1, 5, 5],
        intermediate_dilation_padding=[(0, 3, 3), (0, 2, 2), (0, 1, 1)],
        padding_downsample=[0, 2, 2],
        padding_mode="replicate",
        intermediate_dilation_levels=[(1, 3, 3), (1, 2, 2), (1, 1, 1)],
        num_channels_img_prop_module=3,
        deform_groups=16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_local_frames_propainter = num_local_frames_propainter
        self.flow_weight_flow_complete_net = flow_weight_flow_complete_net
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.adversarial_weight = adversarial_weight
        self.gan_loss = gan_loss
        self.perceptual_weight = perceptual_weight
        self.interp_mode = interp_mode
        self.ref_stride = ref_stride
        self.neighbor_length = neighbor_length
        self.subvideo_length = subvideo_length
        self.correlation_levels = correlation_levels
        self.correlation_radius = correlation_radius
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.raft_iter = raft_iter
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.kernel_size_3d = kernel_size_3d
        self.kernel_size_3d_discriminator = kernel_size_3d_discriminator
        self.padding_inpaint_generator = padding_inpaint_generator
        self.padding = padding
        self.conv2d_stride = conv2d_stride
        self.conv3d_stride = conv3d_stride
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.pool_size = pool_size
        self.use_discriminator = use_discriminator
        self.in_channels = in_channels
        self.channels = channels
        self.multi_level_conv_stride = multi_level_conv_stride
        self.norm_fn = norm_fn
        self.patch_size = patch_size
        self.negative_slope_default = negative_slope_default
        self.negative_slope_1 = negative_slope_1
        self.negative_slope_2 = negative_slope_2
        self.group = group
        self.kernel_size_3d_downsample = kernel_size_3d_downsample
        self.intermediate_dilation_padding = intermediate_dilation_padding
        self.padding_downsample = padding_downsample
        self.padding_mode = padding_mode
        self.intermediate_dilation_levels = intermediate_dilation_levels
        self.num_channels_img_prop_module = num_channels_img_prop_module
        self.deform_groups = deform_groups

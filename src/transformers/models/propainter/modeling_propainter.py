# coding=utf-8
# Copyright 2024 S-Lab, Nanyang Technological University, The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ProPainter model."""

import itertools
import math
from collections import namedtuple
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from torch import nn
from torch.nn import L1Loss
from torch.nn.functional import normalize
from torch.nn.modules.utils import _pair
from torchvision import models as tv

from ...modeling_outputs import (
    BaseModelOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import TORCH_INIT_FUNCTIONS, PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_propainter import ProPainterConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ProPainterConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "ruffy369/propainter"
_EXPECTED_OUTPUT_SHAPE = ["batch_size", 80, 240, 432, 3]


# Adapted from original code at https://github.com/sczhou/ProPainter
class ProPainterResidualBlock(nn.Module):
    def __init__(
        self,
        config: ProPainterConfig,
        in_channels: int,
        channels: int,
        norm_fn: str = "group",
        stride: int = 1,
    ):
        super().__init__()

        self.config = config

        self.conv1 = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=config.patch_size,
            padding=config.padding,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=config.patch_size, padding=config.padding)
        self.relu = nn.ReLU(inplace=True)

        num_groups = channels // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(channels)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(channels)
            self.norm2 = nn.InstanceNorm2d(channels)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(channels)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, hidden_states):
        residual = hidden_states
        residual = self.relu(self.norm1(self.conv1(residual)))

        residual = self.relu(self.norm2(self.conv2(residual)))

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.relu(hidden_states + residual)

        return hidden_states


class ProPainterBasicEncoder(nn.Module):
    def __init__(self, config: ProPainterConfig, output_dim: int = 128, norm_fn: str = "batch"):
        super().__init__()

        self.config = config

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=config.num_hidden_layers, num_channels=config.in_channels[0])

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(config.in_channels[0])

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(config.in_channels[0])

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()

        else:
            raise ValueError(f"Unsupported normalization function: {norm_fn}")

        self.conv1 = nn.Conv2d(
            3,
            config.in_channels[0],
            kernel_size=config.kernel_size[0],
            stride=config.multi_level_conv_stride[1],
            padding=3,
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.resblocks = [
            [
                ProPainterResidualBlock(config, in_channel, num_channels, norm_fn, stride),
                ProPainterResidualBlock(config, num_channels, num_channels, norm_fn, stride=1),
            ]
            for in_channel, num_channels, stride in zip(
                config.in_channels, config.channels, config.multi_level_conv_stride
            )
        ]
        # using itertools makes flattening a little faster :)
        self.resblocks = nn.ModuleList(list(itertools.chain.from_iterable(self.resblocks)))

        # output convolution
        self.conv2 = nn.Conv2d(config.num_channels, output_dim, kernel_size=1)

        self.dropout = None
        if self.config.dropout > 0:
            self.dropout = nn.Dropout2d(p=self.config.dropout)

    def forward(self, image):
        is_iterable = isinstance(image, (tuple, list))
        if is_iterable:
            batch_dim = image[0].shape[0]
            image = torch.cat(image, dim=0)

        hidden_states = self.conv1(image)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.relu1(hidden_states)

        for resblock in self.resblocks:
            hidden_states = resblock(hidden_states)

        hidden_states = self.conv2(hidden_states)

        if self.training and self.dropout is not None:
            hidden_states = self.dropout(hidden_states)

        if is_iterable:
            hidden_states = torch.split(hidden_states, [batch_dim, batch_dim], dim=0)

        return hidden_states


class ProPainterBasicMotionEncoder(nn.Module):
    def __init__(self, config: ProPainterConfig):
        super().__init__()
        self.config = config

        correlation_planes = config.correlation_levels * (2 * config.correlation_radius + 1) ** 2
        self.conv_corr1 = nn.Conv2d(correlation_planes, config.num_channels * 2, 1, padding=0)
        self.conv_corr2 = nn.Conv2d(config.num_channels * 2, 192, config.patch_size, padding=config.padding)
        self.conv_flow1 = nn.Conv2d(2, config.num_channels, config.kernel_size[0], padding=3)
        self.conv_flow2 = nn.Conv2d(
            config.num_channels,
            config.in_channels[0],
            config.patch_size,
            padding=config.padding,
        )
        self.conv = nn.Conv2d(
            config.in_channels[0] + 192,
            config.num_channels - 2,
            config.patch_size,
            padding=config.padding,
        )

    def forward(self, optical_flow, correlation):
        hidden_states_correlation = F.relu(self.conv_corr1(correlation))
        hidden_states_correlation = F.relu(self.conv_corr2(hidden_states_correlation))
        hidden_states_flow = F.relu(self.conv_flow1(optical_flow))
        hidden_states_flow = F.relu(self.conv_flow2(hidden_states_flow))

        hidden_states = torch.cat([hidden_states_correlation, hidden_states_flow], dim=1)
        hidden_states = F.relu(self.conv(hidden_states))
        hidden_states = torch.cat([hidden_states, optical_flow], dim=1)

        return hidden_states


class ProPainterSepConvGRU(nn.Module):
    def __init__(
        self,
        config: ProPainterConfig,
        hidden_dim: int = 128,
        input_dim: int = 192 + 128,
    ):
        super().__init__()
        self.config = config

        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, hidden_states, motion_features):
        hidden_states_motion_features = torch.cat([hidden_states, motion_features], dim=1)
        z = torch.sigmoid(self.convz1(hidden_states_motion_features))
        r = torch.sigmoid(self.convr1(hidden_states_motion_features))
        q = torch.tanh(self.convq1(torch.cat([r * hidden_states, motion_features], dim=1)))
        hidden_states = (1 - z) * hidden_states + z * q
        hidden_states_motion_features = torch.cat([hidden_states, motion_features], dim=1)
        z = torch.sigmoid(self.convz2(hidden_states_motion_features))
        r = torch.sigmoid(self.convr2(hidden_states_motion_features))
        q = torch.tanh(self.convq2(torch.cat([r * hidden_states, motion_features], dim=1)))
        hidden_states = (1 - z) * hidden_states + z * q

        return hidden_states


class ProPainterFlowHead(nn.Module):
    def __init__(self, config: ProPainterConfig, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, config.patch_size, padding=config.padding)
        self.conv2 = nn.Conv2d(hidden_dim, 2, config.patch_size, padding=config.padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, hidden_states):
        hidden_states = self.relu(self.conv1(hidden_states))
        hidden_states = self.conv2(hidden_states)

        return hidden_states


class ProPainterBasicUpdateBlock(nn.Module):
    def __init__(self, config: ProPainterConfig, hidden_dim: int = 128, input_dim: int = 128):
        super().__init__()
        self.config = config
        self.encoder = ProPainterBasicMotionEncoder(config)
        self.gru = ProPainterSepConvGRU(config, hidden_dim=hidden_dim, input_dim=input_dim + hidden_dim)
        self.flow_head = ProPainterFlowHead(config, input_dim=hidden_dim, hidden_dim=config.num_channels * 2)

        self.mask = nn.Sequential(
            nn.Conv2d(
                config.num_channels,
                config.num_channels * 2,
                config.patch_size,
                padding=config.padding,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.num_channels * 2, config.in_channels[0] * 9, 1, padding=0),
        )

    def forward(self, network, input, correlation, optical_flow):
        motion_features = self.encoder(optical_flow, correlation)
        input = torch.cat([input, motion_features], dim=1)

        network = self.gru(network, input)
        delta_flow = self.flow_head(network)
        # scale mask to balance gradients
        mask = 0.25 * self.mask(network)
        return network, mask, delta_flow


def coords_grid(batch_size, height, width):
    coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch_size, 1, 1, 1)


def sample_point(img, coords):
    """Wrapper for grid_sample, uses pixel coordinates"""
    height, width = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (width - 1) - 1
    ygrid = 2 * ygrid / (height - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    return img


class ProPainterCorrBlock:
    def __init__(
        self,
        config: ProPainterConfig,
        feature_map_1: torch.tensor,
        feature_map_2: torch.tensor,
        num_levels: int = 4,
        radius: int = 4,
    ):
        self.config = config
        self.num_levels = num_levels
        self.radius = radius
        self.correlation_pyramid = []

        # all pairs correlation
        correlation = ProPainterCorrBlock.correlation(feature_map_1, feature_map_2)

        batch_size, height_1, width_1, dimension, height_2, width_2 = correlation.shape
        correlation = correlation.reshape(batch_size * height_1 * width_1, dimension, height_2, width_2)

        self.correlation_pyramid.append(correlation)
        for _ in range(self.num_levels - 1):
            correlation = F.avg_pool2d(correlation, 2, stride=config.multi_level_conv_stride[1])
            self.correlation_pyramid.append(correlation)

    def __call__(self, coords):
        radius = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch_size, height_1, width_1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            correlation = self.correlation_pyramid[i]
            delta_x = torch.linspace(-radius, radius, 2 * radius + 1)
            delta_y = torch.linspace(-radius, radius, 2 * radius + 1)
            delta = torch.stack(torch.meshgrid(delta_y, delta_x), axis=-1).to(coords.device)
            centroid_lvl = coords.reshape(batch_size * height_1 * width_1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * radius + 1, 2 * radius + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            correlation = sample_point(correlation, coords_lvl)
            correlation = correlation.view(batch_size, height_1, width_1, -1)
            out_pyramid.append(correlation)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def correlation(feature_map_1, feature_map_2):
        batch_size, dimension, height, width = feature_map_1.shape
        feature_map_1 = feature_map_1.view(batch_size, dimension, height * width)
        feature_map_2 = feature_map_2.view(batch_size, dimension, height * width)
        correlation = torch.matmul(feature_map_1.transpose(1, 2), feature_map_2)
        correlation = correlation.view(batch_size, height, width, 1, height, width)
        return correlation / torch.sqrt(torch.tensor(dimension).float())


class ProPainterRaftOpticalFlow(nn.Module):
    def __init__(self, config: ProPainterConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.num_channels
        self.context_dim = config.num_channels

        self.feature_network = ProPainterBasicEncoder(
            config,
            output_dim=self.hidden_dim + self.context_dim,
            norm_fn=config.norm_fn[2],
        )  # norm_fn: "instance"
        self.context_network = ProPainterBasicEncoder(
            config,
            output_dim=self.hidden_dim + self.context_dim,
            norm_fn=config.norm_fn[0],  # norm_fn: "batch"
        )
        self.update_block = ProPainterBasicUpdateBlock(config, hidden_dim=self.hidden_dim)

    def initialize_flow(self, image):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, height, width = image.shape
        coords0 = coords_grid(N, height // 8, width // 8).to(image.device)
        coords1 = coords_grid(N, height // 8, width // 8).to(image.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [height/8, width/8, 2] -> [height, width, 2] using convex combination"""
        N, _, height, width = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, height, width)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=self.config.padding)
        up_flow = up_flow.view(N, 2, 9, 1, 1, height, width)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * height, 8 * width)

    def _forward(self, image1, image2, iters=12, flow_init=None):
        """Estimate optical flow between pair of frames"""

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        feature_map_1, feature_map_2 = self.feature_network([image1, image2])

        feature_map_1 = feature_map_1.float()
        feature_map_2 = feature_map_2.float()

        correlation_fn = ProPainterCorrBlock(
            self.config, feature_map_1, feature_map_2, radius=self.config.correlation_radius
        )

        context_network_out = self.context_network(image1)
        network, input = torch.split(context_network_out, [self.hidden_dim, self.context_dim], dim=1)
        network = torch.tanh(network)
        input = torch.relu(input)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        for _ in range(iters):
            coords1 = coords1.detach()
            correlation = correlation_fn(coords1)  # index correlation volume

            optical_flow = coords1 - coords0
            network, up_mask, delta_flow = self.update_block(network, input, correlation, optical_flow)

            coords1 = coords1 + delta_flow

            if up_mask is None:
                new_size = (
                    8 * (coords1 - coords0).shape[2],
                    8 * (coords1 - coords0).shape[3],
                )
                flow_up = 8 * F.interpolate(
                    (coords1 - coords0),
                    size=new_size,
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

        return coords1 - coords0, flow_up

    def forward(self, ground_truth_local_frames, iters=20):
        batch_size, temporal_length, num_channels, height, width = ground_truth_local_frames.size()

        ground_truth_local_frames_1 = ground_truth_local_frames[:, :-1, :, :, :].reshape(
            -1, num_channels, height, width
        )
        ground_truth_local_frames_2 = ground_truth_local_frames[:, 1:, :, :, :].reshape(
            -1, num_channels, height, width
        )
        _, ground_truth_flows_forward = self._forward(ground_truth_local_frames_1, ground_truth_local_frames_2, iters)
        _, ground_truth_flows_backward = self._forward(ground_truth_local_frames_2, ground_truth_local_frames_1, iters)

        ground_truth_flows_forward = ground_truth_flows_forward.view(batch_size, temporal_length - 1, 2, height, width)
        ground_truth_flows_backward = ground_truth_flows_backward.view(
            batch_size, temporal_length - 1, 2, height, width
        )

        return ground_truth_flows_forward, ground_truth_flows_backward


class ProPainterP3DBlock(nn.Module):
    def __init__(
        self,
        config: ProPainterConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_residual: bool = False,
        bias=True,
    ):
        super().__init__()
        self.config = config
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, config.patch_size, config.patch_size),
                stride=(1, stride, stride),
                padding=(0, config.padding, config.padding),
                bias=bias,
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(2, 0, 0),
                dilation=(2, 1, 1),
                bias=bias,
            )
        )
        self.use_residual = use_residual

    def forward(self, hidden_state):
        features1 = self.conv1(hidden_state)
        features2 = self.conv2(features1)
        if self.use_residual:
            hidden_state = hidden_state + features2
        else:
            hidden_state = features2
        return hidden_state


class ProPainterEdgeDetection(nn.Module):
    def __init__(
        self,
        config: ProPainterConfig,
        in_channel: int = 2,
        out_channel: int = 1,
        intermediate_channel: int = 16,
    ):
        super().__init__()

        self.config = config
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channel,
                intermediate_channel,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
        )

        self.intermediate_layer_1 = nn.Sequential(
            nn.Conv2d(
                intermediate_channel,
                intermediate_channel,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
        )

        self.intermediate_layer_2 = nn.Sequential(
            nn.Conv2d(
                intermediate_channel,
                intermediate_channel,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            )
        )

        self.relu = nn.LeakyReLU(config.negative_slope_2, inplace=True)

        self.out_layer = nn.Conv2d(intermediate_channel, out_channel, 1, config.multi_level_conv_stride[0], 0)

    def forward(self, flow):
        flow = self.projection(flow)
        edge = self.intermediate_layer_1(flow)
        edge = self.intermediate_layer_2(edge)
        edge = self.relu(flow + edge)
        edge = self.out_layer(edge)
        edge = torch.sigmoid(edge)

        return edge


class ProPainterBidirectionalPropagationFlowComplete(nn.Module):
    def __init__(self, config: ProPainterConfig):
        super().__init__()
        self.config = config

        modules = ["backward_", "forward_"]
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()

        for i, module in enumerate(modules):
            self.deform_align[module] = ProPainterSecondOrderDeformableAlignment(
                config,
                2 * config.num_channels,
                config.num_channels,
                config.patch_size,
                padding=config.padding,
                deform_groups=16,
            )

            self.backbone[module] = nn.Sequential(
                nn.Conv2d(
                    (2 + i) * config.num_channels,
                    config.num_channels,
                    config.patch_size,
                    config.multi_level_conv_stride[0],
                    config.padding,
                ),
                nn.LeakyReLU(negative_slope=config.negative_slope_1, inplace=True),
                nn.Conv2d(
                    config.num_channels,
                    config.num_channels,
                    config.patch_size,
                    config.multi_level_conv_stride[0],
                    config.padding,
                ),
            )

        self.fusion = nn.Conv2d(
            2 * config.num_channels,
            config.num_channels,
            config.multi_level_conv_stride[0],
            config.padding,
            0,
        )

    def forward(self, hidden_state):
        """
        hidden_state shape : [batch_size, timesteps, num_channels, height, width]
        return [batch_size, timesteps, num_channels, height, width]
        """

        batch_size, timesteps, _, height, width = hidden_state.shape
        features = {}
        features["spatial"] = [hidden_state[:, i, :, :, :] for i in range(0, timesteps)]

        for module_name in ["backward_", "forward_"]:
            features[module_name] = []

            frame_indices = range(0, timesteps)
            mapping_idx = list(range(0, len(features["spatial"])))
            mapping_idx += mapping_idx[::-1]

            if "backward" in module_name:
                frame_indices = frame_indices[::-1]

            feature_propagation = hidden_state.new_zeros(batch_size, self.config.num_channels, height, width)
            for frame_count, frame_id in enumerate(frame_indices):
                feat_current = features["spatial"][mapping_idx[frame_id]]
                if frame_count > 0:
                    first_order_condition_features = feature_propagation

                    second_order_propagated_features = torch.zeros_like(feature_propagation)
                    second_order_condition_features = torch.zeros_like(first_order_condition_features)
                    if frame_count > 1:
                        second_order_propagated_features = features[module_name][-2]
                        second_order_condition_features = second_order_propagated_features

                    condition_features = torch.cat(
                        [first_order_condition_features, feat_current, second_order_condition_features], dim=1
                    )
                    feature_propagation = torch.cat([feature_propagation, second_order_propagated_features], dim=1)
                    feature_propagation = self.deform_align[module_name](feature_propagation, condition_features)
                feat = (
                    [feat_current]
                    + [features[k][frame_id] for k in features if k not in ["spatial", module_name]]
                    + [feature_propagation]
                )

                feat = torch.cat(feat, dim=1)
                feature_propagation = feature_propagation + self.backbone[module_name](feat)
                features[module_name].append(feature_propagation)
            if "backward" in module_name:
                features[module_name] = features[module_name][::-1]

        outputs = []
        for i in range(0, timesteps):
            align_feats = [features[k].pop(0) for k in features if k != "spatial"]
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        hidden_state = torch.stack(outputs, dim=1) + hidden_state

        return hidden_state


def flow_warp(features, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        features (Tensor): Tensor with size (n, num_channels, height, width).
        flow (Tensor): Tensor with size (n, height, width, 2). The last dimension is
            a two-num_channels, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if features.size()[-2:] != flow.size()[1:3]:
        raise ValueError(
            f"The spatial sizes of input ({features.size()[-2:]}) and " f"flow ({flow.size()[1:3]}) are not the same."
        )
    _, _, height, width = features.size()
    device = flow.device
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height, device=device), torch.arange(0, width, device=device))
    grid = torch.stack((grid_x, grid_y), 2).type_as(features)  # (width, height, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(width - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(height - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        features,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return output


def forward_backward_consistency_check(flow_forward, flow_backward, alpha1=0.01, alpha2=0.5):
    """
    Checks the consistency between forward and backward optical flows.

    Args:
        flow_forward (torch.Tensor): The forward optical flow.
        flow_backward (torch.Tensor): The backward optical flow.
        alpha1 (float, optional): Scaling factor for the occlusion threshold. Default is 0.01.
        alpha2 (float, optional): Constant for the occlusion threshold. Default is 0.5.

    Returns:
        torch.Tensor: A mask indicating regions where the forward and backward flows are consistent.

    The function warps the backward flow to the forward flow space and computes the difference
    between the forward flow and the warped backward flow. It also calculates an occlusion threshold
    using the squared norms of the forward flow and the warped backward flow. The mask identifies
    regions where the flow difference is below this threshold, indicating consistency.
    """

    flow_backward_warped_to_forward = flow_warp(flow_backward, flow_forward.permute(0, 2, 3, 1))
    flow_diff_forward = flow_forward + flow_backward_warped_to_forward

    flow_forward_norm_squared = (
        torch.norm(flow_forward, p=2, dim=1, keepdim=True) ** 2
        + torch.norm(flow_backward_warped_to_forward, p=2, dim=1, keepdim=True) ** 2
    )
    flow_forward_occlusion_threshold = alpha1 * flow_forward_norm_squared + alpha2

    forward_backward_valid_mask = (
        torch.norm(flow_diff_forward, p=2, dim=1, keepdim=True) ** 2 < flow_forward_occlusion_threshold
    ).to(flow_forward)
    return forward_backward_valid_mask


class ProPainterBidirectionalPropagationInPaint(nn.Module):
    def __init__(self, config: ProPainterConfig, num_channels: int, learnable: bool = True):
        super().__init__()
        self.config = config
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.num_channels = num_channels
        self.propagation_list = ["backward_1", "forward_1"]
        self.learnable = learnable

        if self.learnable:
            for _, module in enumerate(self.propagation_list):
                self.deform_align[module] = ProPainterDeformableAlignment(
                    config,
                    num_channels,
                    num_channels,
                )

                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(
                        2 * num_channels + 2,
                        num_channels,
                        config.patch_size,
                        config.multi_level_conv_stride[0],
                        config.padding,
                    ),
                    nn.LeakyReLU(negative_slope=config.negative_slope_default, inplace=True),
                    nn.Conv2d(
                        num_channels,
                        num_channels,
                        config.patch_size,
                        config.multi_level_conv_stride[0],
                        config.padding,
                    ),
                )

            self.fuse = nn.Sequential(
                nn.Conv2d(
                    2 * num_channels + 2,
                    num_channels,
                    config.patch_size,
                    config.multi_level_conv_stride[0],
                    config.padding,
                ),
                nn.LeakyReLU(negative_slope=config.negative_slope_default, inplace=True),
                nn.Conv2d(
                    num_channels,
                    num_channels,
                    config.patch_size,
                    config.multi_level_conv_stride[0],
                    config.padding,
                ),
            )

    def forward(
        self,
        masked_frames,
        flows_forward,
        flows_backward,
        mask,
        interpolation="bilinear",
    ):
        """
        masked_frames shape : [batch_size, timesteps, num_channels, height, width]
        return [batch_size, timesteps, num_channels, height, width]
        """

        batch_size, timesteps, num_channels, height, width = masked_frames.shape
        features, masks = {}, {}
        features["input"] = [masked_frames[:, i, :, :, :] for i in range(0, timesteps)]
        masks["input"] = [mask[:, i, :, :, :] for i in range(0, timesteps)]

        propagation_list = ["backward_1", "forward_1"]
        cache_list = ["input"] + propagation_list

        for propagation_index, module_name in enumerate(propagation_list):
            features[module_name] = []
            masks[module_name] = []

            is_backward = "backward" in module_name

            frame_indices = range(0, timesteps)[::-1] if is_backward else range(timesteps)
            flow_idx = frame_indices if is_backward else range(-1, timesteps - 1)

            flows_for_prop, flows_for_check = (
                (flows_forward, flows_backward) if is_backward else (flows_backward, flows_forward)
            )

            for frame_count, frame_id in enumerate(frame_indices):
                feat_current = features[cache_list[propagation_index]][frame_id]
                mask_current = masks[cache_list[propagation_index]][frame_id]

                if frame_count == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[frame_count], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[frame_count], :, :, :]
                    flow_valid_mask = forward_backward_consistency_check(flow_prop, flow_check)
                    feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)

                    if self.learnable:
                        condition_features = torch.cat(
                            [
                                feat_current,
                                feat_warped,
                                flow_prop,
                                flow_valid_mask,
                                mask_current,
                            ],
                            dim=1,
                        )
                        feat_prop = self.deform_align[module_name](feat_prop, condition_features, flow_prop)
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = flow_warp(mask_prop, flow_prop.permute(0, 2, 3, 1))
                        mask_prop_valid = torch.where(mask_prop_valid > 0.1, 1, 0).to(mask_prop_valid)

                        union_valid_mask = mask_current * flow_valid_mask * (1 - mask_prop_valid)
                        union_valid_mask = torch.where(union_valid_mask > 0.1, 1, 0).to(union_valid_mask)

                        feat_prop = union_valid_mask * feat_warped + (1 - union_valid_mask) * feat_current
                        mask_prop = mask_current * (1 - (flow_valid_mask * (1 - mask_prop_valid)))
                        mask_prop = torch.where(mask_prop > 0.1, 1, 0).to(mask_prop)

                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)

                features[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)
            if "backward" in module_name:
                features[module_name] = features[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        outputs_backward = torch.stack(features["backward_1"], dim=1).view(-1, num_channels, height, width)
        outputs_forward = torch.stack(features["forward_1"], dim=1).view(-1, num_channels, height, width)

        if self.learnable:
            mask_in = mask.view(-1, 2, height, width)
            masks_forward = None
            outputs = self.fuse(torch.cat([outputs_backward, outputs_forward, mask_in], dim=1)) + masked_frames.view(
                -1, num_channels, height, width
            )
        else:
            masks_forward = torch.stack(masks["forward_1"], dim=1)
            outputs = outputs_forward

        return (
            outputs_backward.view(batch_size, -1, num_channels, height, width),
            outputs_forward.view(batch_size, -1, num_channels, height, width),
            outputs.view(batch_size, -1, num_channels, height, width),
            masks_forward,
        )


class ProPainterDeconv(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel_size: int = 3,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2, mode="bilinear", align_corners=True)
        return self.conv(hidden_states)


class ProPainterDeformableAlignment(nn.Module):
    """Second-order deformable alignment module."""

    def __init__(
        self,
        config: ProPainterConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        **kwargs,
    ):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 3)

        super().__init__()

        self.config = config

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(config.patch_size)
        self.stride = stride
        self.padding = config.padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = config.deform_groups
        self.with_bias = bias

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                2 * self.out_channels + 2 + 1 + 2,
                self.out_channels,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
            nn.LeakyReLU(negative_slope=config.negative_slope_1, inplace=True),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
            nn.LeakyReLU(negative_slope=config.negative_slope_1, inplace=True),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
            nn.LeakyReLU(negative_slope=config.negative_slope_1, inplace=True),
            nn.Conv2d(
                self.out_channels,
                27 * self.deform_groups,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
        )

    def forward(self, features_propagation, condition_features, flow):
        output = self.conv_offset(condition_features)
        output1, output2, mask = torch.chunk(output, 3, dim=1)

        offset = self.max_residue_magnitude * torch.tanh(torch.cat((output1, output2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        mask = torch.sigmoid(mask)
        hidden_states = torchvision.ops.deform_conv2d(
            features_propagation,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )

        return hidden_states


class ProPainterSecondOrderDeformableAlignment(nn.Module):
    """Second-order deformable alignment module."""

    def __init__(
        self,
        config: ProPainterConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = True,
        **kwargs,
    ):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 5)

        super().__init__()

        self.config = config

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.with_bias = bias

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                3 * self.out_channels,
                self.out_channels,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
            nn.LeakyReLU(negative_slope=config.negative_slope_1, inplace=True),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
            nn.LeakyReLU(negative_slope=config.negative_slope_1, inplace=True),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
            nn.LeakyReLU(negative_slope=config.negative_slope_1, inplace=True),
            nn.Conv2d(
                self.out_channels,
                27 * self.deform_groups,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
        )

    def forward(self, features, extra_features):
        output = self.conv_offset(extra_features)
        output1, output2, mask = torch.chunk(output, 3, dim=1)

        offset = self.max_residue_magnitude * torch.tanh(torch.cat((output1, output2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)

        hidden_states = torchvision.ops.deform_conv2d(
            features,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )

        return hidden_states


class ProPainterRecurrentFlowCompleteNet(nn.Module):
    def __init__(self, config: ProPainterConfig):
        super().__init__()
        self.config = config
        self.downsample = nn.Sequential(
            nn.Conv3d(
                3,
                config.num_channels // 4,
                kernel_size=config.kernel_size_3d_downsample,
                stride=config.multi_level_conv_stride,
                padding=config.padding_downsample,
                padding_mode=config.padding_mode,
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
        )

        self.encoder_stage_1 = nn.Sequential(
            ProPainterP3DBlock(
                config,
                config.num_channels // 4,
                config.num_channels // 4,
                config.multi_level_conv_stride[0],
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            ProPainterP3DBlock(
                config,
                config.num_channels // 4,
                config.in_channels[0],
                config.multi_level_conv_stride[1],
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
        )  # 4x

        self.encoder_stage_2 = nn.Sequential(
            ProPainterP3DBlock(
                config,
                config.in_channels[0],
                config.in_channels[0],
                config.multi_level_conv_stride[0],
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            ProPainterP3DBlock(
                config,
                config.in_channels[0],
                self.config.num_channels,
                config.multi_level_conv_stride[1],
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
        )  # 8x

        self.intermediate_dilation = nn.Sequential(
            nn.Conv3d(
                self.config.num_channels,
                self.config.num_channels,
                config.kernel_size_3d,
                config.conv3d_stride,
                padding=config.intermediate_dilation_padding[0],
                dilation=config.intermediate_dilation_levels[0],
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            nn.Conv3d(
                self.config.num_channels,
                self.config.num_channels,
                config.kernel_size_3d,
                config.conv3d_stride,
                padding=config.intermediate_dilation_padding[1],
                dilation=config.intermediate_dilation_levels[1],
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            nn.Conv3d(
                self.config.num_channels,
                self.config.num_channels,
                config.kernel_size_3d,
                config.conv3d_stride,
                padding=config.intermediate_dilation_padding[2],
                dilation=config.intermediate_dilation_levels[2],
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
        )

        # feature propagation module
        self.feature_propagation_module = ProPainterBidirectionalPropagationFlowComplete(config)

        self.decoder_stage_2 = nn.Sequential(
            nn.Conv2d(
                self.config.num_channels,
                self.config.num_channels,
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            ProPainterDeconv(self.config.num_channels, config.in_channels[0], config.patch_size, 1),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
        )  # 4x

        self.decoder_stage_1 = nn.Sequential(
            nn.Conv2d(
                config.in_channels[0],
                config.in_channels[0],
                config.patch_size,
                config.multi_level_conv_stride[0],
                config.padding,
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            ProPainterDeconv(config.in_channels[0], config.num_channels // 4, config.patch_size, 1),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
        )  # 2x

        self.upsample = nn.Sequential(
            nn.Conv2d(
                config.num_channels // 4,
                config.num_channels // 4,
                config.patch_size,
                padding=config.padding,
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            ProPainterDeconv(config.num_channels // 4, 2, config.patch_size, 1),
        )

        # edge loss
        self.edgeDetector = ProPainterEdgeDetection(config, in_channel=2, out_channel=1, intermediate_channel=16)

    def forward(self, masked_flows, masks):
        batch_size, timesteps, _, height, width = masked_flows.size()
        masked_flows = masked_flows.permute(0, 2, 1, 3, 4)
        masks = masks.permute(0, 2, 1, 3, 4)

        inputs = torch.cat((masked_flows, masks), dim=1)

        downsample_inputs = self.downsample(inputs)

        encoded_features_stage_1 = self.encoder_stage_1(downsample_inputs)
        encoded_features_stage_2 = self.encoder_stage_2(encoded_features_stage_1)
        features_intermediate = self.intermediate_dilation(encoded_features_stage_2)
        features_intermediate = features_intermediate.permute(0, 2, 1, 3, 4)

        features_prop = self.feature_propagation_module(features_intermediate)
        features_prop = features_prop.view(-1, self.config.num_channels, height // 8, width // 8)

        _, num_channels, _, feature_height, feature_width = encoded_features_stage_1.shape
        encoded_features_stage_1 = (
            encoded_features_stage_1.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(-1, num_channels, feature_height, feature_width)
        )
        decoded_features_stage_2 = self.decoder_stage_2(features_prop) + encoded_features_stage_1

        _, num_channels, _, feature_height, feature_width = downsample_inputs.shape
        downsample_inputs = (
            downsample_inputs.permute(0, 2, 1, 3, 4).contiguous().view(-1, num_channels, feature_height, feature_width)
        )

        decoded_features_stage_1 = self.decoder_stage_1(decoded_features_stage_2)

        flow = self.upsample(decoded_features_stage_1)
        edge = self.edgeDetector(flow)
        edge = edge.view(batch_size, timesteps, 1, height, width)

        flow = flow.view(batch_size, timesteps, 2, height, width)

        return flow, edge

    def forward_bidirectional_flow(self, masked_flows_bidirectional, masks):
        """
        Args:
            masked_flows_bidirectional: [masked_flows_f, masked_flows_b] | (batch_size, timesteps-1, 2, height, width), (batch_size, timesteps-1, 2, height, width)
            masks: batch_size, timesteps, 1, height, width
        """
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        # mask flow
        masked_flows_forward = masked_flows_bidirectional[0] * (1 - masks_forward)
        masked_flows_backward = masked_flows_bidirectional[1] * (1 - masks_backward)

        # -- completion --
        pred_flows_forward, pred_edges_forward = self.forward(masked_flows_forward, masks_forward)

        # backward
        masked_flows_backward = torch.flip(masked_flows_backward, dims=[1])
        masks_backward = torch.flip(masks_backward, dims=[1])
        pred_flows_backward, pred_edges_backward = self.forward(masked_flows_backward, masks_backward)
        pred_flows_backward = torch.flip(pred_flows_backward, dims=[1])
        if self.training:
            pred_edges_backward = torch.flip(pred_edges_backward, dims=[1])

        return [pred_flows_forward, pred_flows_backward], [
            pred_edges_forward,
            pred_edges_backward,
        ]

    def combine_flow(self, masked_flows_bidirectional, pred_flows_bidirectional, masks):
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        pred_flows_forward = pred_flows_bidirectional[0] * masks_forward + masked_flows_bidirectional[0] * (
            1 - masks_forward
        )
        pred_flows_backward = pred_flows_bidirectional[1] * masks_backward + masked_flows_bidirectional[1] * (
            1 - masks_backward
        )

        return pred_flows_forward, pred_flows_backward


class ProPainterEncoder(nn.Module):
    def __init__(self, config: ProPainterConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    5,
                    config.in_channels[0],
                    kernel_size=config.patch_size,
                    stride=config.multi_level_conv_stride[1],
                    padding=config.padding,
                ),
                nn.LeakyReLU(config.negative_slope_default, inplace=True),
                nn.Conv2d(
                    config.in_channels[0],
                    config.in_channels[0],
                    kernel_size=config.patch_size,
                    stride=1,
                    padding=config.padding,
                ),
                nn.LeakyReLU(config.negative_slope_default, inplace=True),
                nn.Conv2d(
                    config.in_channels[0],
                    config.num_channels,
                    kernel_size=config.patch_size,
                    stride=config.multi_level_conv_stride[1],
                    padding=config.padding,
                ),
                nn.LeakyReLU(config.negative_slope_default, inplace=True),
                nn.Conv2d(
                    config.num_channels,
                    config.num_channels * 2,
                    kernel_size=config.patch_size,
                    stride=1,
                    padding=config.padding,
                ),
                nn.LeakyReLU(config.negative_slope_default, inplace=True),
                nn.Conv2d(
                    config.num_channels * 2,
                    config.hidden_size - config.num_channels,
                    kernel_size=config.patch_size,
                    stride=1,
                    padding=config.padding,
                    groups=1,
                ),
                nn.LeakyReLU(config.negative_slope_default, inplace=True),
                nn.Conv2d(
                    config.hidden_size + config.num_channels,
                    config.hidden_size,
                    kernel_size=config.patch_size,
                    stride=1,
                    padding=config.padding,
                    groups=2,
                ),
                nn.LeakyReLU(config.negative_slope_default, inplace=True),
                nn.Conv2d(
                    config.hidden_size + config.num_channels * 2,
                    config.hidden_size - config.num_channels,
                    kernel_size=config.patch_size,
                    stride=1,
                    padding=config.padding,
                    groups=4,
                ),
                nn.LeakyReLU(config.negative_slope_default, inplace=True),
                nn.Conv2d(
                    config.hidden_size + config.num_channels,
                    config.num_channels * 2,
                    kernel_size=config.patch_size,
                    stride=1,
                    padding=config.padding,
                    groups=8,
                ),
                nn.LeakyReLU(config.negative_slope_default, inplace=True),
                nn.Conv2d(
                    config.hidden_size,
                    config.num_channels,
                    kernel_size=config.patch_size,
                    stride=config.multi_level_conv_stride[0],
                    padding=config.padding,
                    groups=1,
                ),
                nn.LeakyReLU(config.negative_slope_default, inplace=True),
            ]
        )

    def forward(self, masked_inputs):
        batch_size, _, _, _ = masked_inputs.size()
        features = masked_inputs
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = features  # Store the features from layer 8 as a reference point
                _, _, height, width = x0.size()
            if i > 8 and i % 2 == 0:
                # For even layers after 8, group the channels and concatenate the reference features (x0)
                group = self.config.group[(i - 8) // 2]  # Adjust the grouping based on layer index
                masked_inputs = x0.view(batch_size, group, -1, height, width)
                feature = features.view(batch_size, group, -1, height, width)
                features = torch.cat([masked_inputs, feature], 2).view(batch_size, -1, height, width)
            # For layers before 8 and odd-numbered layers after 8, features are passed through as-is
            features = layer(features)

        return features


class ProPainterSoftSplit(nn.Module):
    def __init__(self, config: ProPainterConfig):
        super().__init__()
        self.config = config

        self.kernel_size = config.kernel_size
        self.stride = config.conv2d_stride
        self.padding = config.padding_inpaint_generator
        self.unfold = nn.Unfold(
            kernel_size=config.kernel_size,
            stride=config.conv2d_stride,
            padding=config.padding_inpaint_generator,
        )
        input_features = reduce((lambda x, y: x * y), config.kernel_size) * config.num_channels
        self.embedding = nn.Linear(input_features, config.hidden_size)

    def forward(self, hidden_states, batch_size, output_size):
        features_height = int(
            (output_size[0] + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        )
        features_width = int(
            (output_size[1] + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
        )

        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.embedding(hidden_states)
        hidden_states = hidden_states.view(batch_size, -1, features_height, features_width, hidden_states.size(2))

        return hidden_states


class ProPainterSoftComp(nn.Module):
    def __init__(self, config: ProPainterConfig):
        super().__init__()
        self.config = config
        self.relu = nn.LeakyReLU(config.negative_slope_default, inplace=True)
        output_features = reduce((lambda x, y: x * y), config.kernel_size) * config.num_channels
        self.embedding = nn.Linear(config.hidden_size, output_features)
        self.kernel_size = config.kernel_size
        self.stride = config.conv2d_stride
        self.padding = config.padding_inpaint_generator
        self.bias_conv = nn.Conv2d(
            config.num_channels,
            config.num_channels,
            kernel_size=config.patch_size,
            stride=1,
            padding=config.padding,
        )

    def forward(self, hidden_state, timestep, output_size):
        num_batch_, _, _, _, channel_ = hidden_state.shape
        hidden_state = hidden_state.view(num_batch_, -1, channel_)
        hidden_state = self.embedding(hidden_state)
        batch_size, _, num_channels = hidden_state.size()
        hidden_state = hidden_state.view(batch_size * timestep, -1, num_channels).permute(0, 2, 1)
        hidden_state = F.fold(
            hidden_state,
            output_size=output_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        hidden_state = self.bias_conv(hidden_state)

        return hidden_state


def window_partition(input_feature, window_size, num_attention_heads):
    """
    Args:
        input_feature: shape is (batch_size, timesteps, height, width, num_channels)
        window_size (tuple[int]): window size
    Returns:
        windows: (batch_size, num_windows_h, num_windows_w, num_attention_heads, timesteps, window_size, window_size, num_channels//num_attention_heads)
    """
    batch_size, timesteps, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size,
        timesteps,
        height // window_size[0],  # Reduce height by window_size
        window_size[0],  # Store windowed height dimension
        width // window_size[1],  # Reduce width by window_size
        window_size[1],  # Store windowed width dimension
        num_attention_heads,  # Split channels across attention heads
        num_channels // num_attention_heads,  # Channels per head
    )

    # Permute the dimensions to bring attention heads next to the spatial patches, keeping timesteps and the per-head channels intact.
    windows = input_feature.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows


class ProPainterSparseWindowAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        window_size: Tuple[int, int],
        pool_size: Tuple[int, int] = (4, 4),
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pooling_token: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(hidden_size, hidden_size, qkv_bias)
        self.query = nn.Linear(hidden_size, hidden_size, qkv_bias)
        self.value = nn.Linear(hidden_size, hidden_size, qkv_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # output projection
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.pooling_token = pooling_token
        if self.pooling_token:
            kernel_size, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(
                hidden_size,
                hidden_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=(0, 0),
                groups=hidden_size,
            )
            self.pool_layer.weight.data.fill_(1.0 / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

        if any(i > 0 for i in self.expand_size):
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[: -self.expand_size[0], : -self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[: -self.expand_size[0], self.expand_size[1] :] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0] :, : -self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0] :, self.expand_size[1] :] = 0
            masked_rolled_key = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", masked_rolled_key.nonzero(as_tuple=False).view(-1))

        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))

    def forward(
        self,
        hidden_states,
        mask=None,
        token_indices=None,
        output_attentions: bool = False,
    ):
        all_self_attentions = () if output_attentions else None

        batch_size, timesteps, height, width, num_channels = hidden_states.shape  # 20 36
        window_height, window_width = self.window_size[0], self.window_size[1]
        channel_head = num_channels // self.num_attention_heads
        n_window_height = math.ceil(height / self.window_size[0])
        n_window_width = math.ceil(width / self.window_size[1])
        new_height = n_window_height * self.window_size[0]  # 20
        new_width = n_window_width * self.window_size[1]  # 36
        padding_right = new_width - width
        padding_bottom = new_height - height
        if padding_right > 0 or padding_bottom > 0:
            hidden_states = F.pad(
                hidden_states,
                (0, 0, 0, padding_right, 0, padding_bottom, 0, 0),
                mode="constant",
                value=0,
            )
            mask = F.pad(mask, (0, 0, 0, padding_right, 0, padding_bottom, 0, 0), mode="constant", value=0)

        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        window_query = window_partition(query.contiguous(), self.window_size, self.num_attention_heads).view(
            batch_size,
            n_window_height * n_window_width,
            self.num_attention_heads,
            timesteps,
            window_height * window_width,
            channel_head,
        )
        window_key = window_partition(key.contiguous(), self.window_size, self.num_attention_heads).view(
            batch_size,
            n_window_height * n_window_width,
            self.num_attention_heads,
            timesteps,
            window_height * window_width,
            channel_head,
        )
        window_value = window_partition(value.contiguous(), self.window_size, self.num_attention_heads).view(
            batch_size,
            n_window_height * n_window_width,
            self.num_attention_heads,
            timesteps,
            window_height * window_width,
            channel_head,
        )
        if any(i > 0 for i in self.expand_size):
            key_top_left, value_top_left = (
                torch.roll(a, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3)) for a in (key, value)
            )

            key_top_right, value_top_right = (
                torch.roll(a, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3)) for a in (key, value)
            )

            key_bottom_left, value_bottom_left = (
                torch.roll(a, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3)) for a in (key, value)
            )

            key_bottom_right, value_bottom_right = (
                torch.roll(a, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3)) for a in (key, value)
            )

            (
                key_top_left_windows,
                key_top_right_windows,
                key_bottom_left_windows,
                key_bottom_right_windows,
            ) = (
                window_partition(a, self.window_size, self.num_attention_heads).view(
                    batch_size,
                    n_window_height * n_window_width,
                    self.num_attention_heads,
                    timesteps,
                    window_height * window_width,
                    channel_head,
                )
                for a in (
                    key_top_left,
                    key_top_right,
                    key_bottom_left,
                    key_bottom_right,
                )
            )

            (
                value_top_left_windows,
                value_top_right_windows,
                value_bottom_left_windows,
                value_bottom_right_windows,
            ) = (
                window_partition(a, self.window_size, self.num_attention_heads).view(
                    batch_size,
                    n_window_height * n_window_width,
                    self.num_attention_heads,
                    timesteps,
                    window_height * window_width,
                    channel_head,
                )
                for a in (
                    value_top_left,
                    value_top_right,
                    value_bottom_left,
                    value_bottom_right,
                )
            )

            rool_key = torch.cat(
                (
                    key_top_left_windows,
                    key_top_right_windows,
                    key_bottom_left_windows,
                    key_bottom_right_windows,
                ),
                4,
            ).contiguous()
            rool_value = torch.cat(
                (
                    value_top_left_windows,
                    value_top_right_windows,
                    value_bottom_left_windows,
                    value_bottom_right_windows,
                ),
                4,
            ).contiguous()  # [batch_size, n_window_height*n_window_width, num_attention_heads, timesteps, window_height*window_width, channel_head]
            rool_key = rool_key[:, :, :, :, self.valid_ind_rolled]
            rool_value = rool_value[:, :, :, :, self.valid_ind_rolled]
            roll_N = rool_key.shape[4]
            rool_key = rool_key.view(
                batch_size,
                n_window_height * n_window_width,
                self.num_attention_heads,
                timesteps,
                roll_N,
                num_channels // self.num_attention_heads,
            )
            rool_value = rool_value.view(
                batch_size,
                n_window_height * n_window_width,
                self.num_attention_heads,
                timesteps,
                roll_N,
                num_channels // self.num_attention_heads,
            )
            window_key = torch.cat((window_key, rool_key), dim=4)
            window_value = torch.cat((window_value, rool_value), dim=4)
        else:
            window_key = window_key
            window_value = window_value

        if self.pooling_token:
            pool_x = self.pool_layer(
                hidden_states.view(batch_size * timesteps, new_height, new_width, num_channels).permute(0, 3, 1, 2)
            )
            _, _, p_h, p_w = pool_x.shape
            pool_x = pool_x.permute(0, 2, 3, 1).view(batch_size, timesteps, p_h, p_w, num_channels)
            pool_k = (
                self.key(pool_x).unsqueeze(1).repeat(1, n_window_height * n_window_width, 1, 1, 1, 1)
            )  # [batch_size, n_window_height*n_window_width, timesteps, p_h, p_w, num_channels]
            pool_k = pool_k.view(
                batch_size,
                n_window_height * n_window_width,
                timesteps,
                p_h,
                p_w,
                self.num_attention_heads,
                channel_head,
            ).permute(0, 1, 5, 2, 3, 4, 6)
            pool_k = pool_k.contiguous().view(
                batch_size,
                n_window_height * n_window_width,
                self.num_attention_heads,
                timesteps,
                p_h * p_w,
                channel_head,
            )
            window_key = torch.cat((window_key, pool_k), dim=4)
            pool_v = (
                self.value(pool_x).unsqueeze(1).repeat(1, n_window_height * n_window_width, 1, 1, 1, 1)
            )  # [batch_size, n_window_height*n_window_width, timesteps, p_h, p_w, num_channels]
            pool_v = pool_v.view(
                batch_size,
                n_window_height * n_window_width,
                timesteps,
                p_h,
                p_w,
                self.num_attention_heads,
                channel_head,
            ).permute(0, 1, 5, 2, 3, 4, 6)
            pool_v = pool_v.contiguous().view(
                batch_size,
                n_window_height * n_window_width,
                self.num_attention_heads,
                timesteps,
                p_h * p_w,
                channel_head,
            )
            window_value = torch.cat((window_value, pool_v), dim=4)

        # [batch_size, n_window_height*n_window_width, num_attention_heads, timesteps, window_height*window_width, channel_head]
        output = torch.zeros_like(window_query)
        l_t = mask.size(1)

        mask = self.max_pool(mask.view(batch_size * l_t, new_height, new_width))
        mask = mask.view(batch_size, l_t, n_window_height * n_window_width)
        mask = torch.sum(mask, dim=1)  # [batch_size, n_window_height*n_window_width]
        for i in range(window_query.shape[0]):
            mask_ind_i = mask[i].nonzero(as_tuple=False).view(-1)
            # [batch_size, n_window_height*n_window_width, num_attention_heads, timesteps, window_height*window_width, channel_head]
            num_masked_indices = len(mask_ind_i)
            if num_masked_indices > 0:
                window_query_masked = window_query[i, mask_ind_i].view(
                    num_masked_indices,
                    self.num_attention_heads,
                    timesteps * window_height * window_width,
                    channel_head,
                )
                window_key_masked = window_key[i, mask_ind_i]
                window_value_masked = window_value[i, mask_ind_i]
                if token_indices is not None:
                    # key [n_window_height*n_window_width, num_attention_heads, timesteps, window_height*window_width, channel_head]
                    window_key_masked = window_key_masked[:, :, token_indices.view(-1)].view(
                        num_masked_indices, self.num_attention_heads, -1, channel_head
                    )
                    window_value_masked = window_value_masked[:, :, token_indices.view(-1)].view(
                        num_masked_indices, self.num_attention_heads, -1, channel_head
                    )
                else:
                    window_key_masked = window_key_masked.view(
                        n_window_height * n_window_width,
                        self.num_attention_heads,
                        timesteps * window_height * window_width,
                        channel_head,
                    )
                    window_value_masked = window_value_masked.view(
                        n_window_height * n_window_width,
                        self.num_attention_heads,
                        timesteps * window_height * window_width,
                        channel_head,
                    )

                attention_scores_masked = (window_query_masked @ window_key_masked.transpose(-2, -1)) * (
                    1.0 / math.sqrt(window_query_masked.size(-1))
                )
                attention_scores_masked = F.softmax(attention_scores_masked, dim=-1)
                attention_scores_masked = self.attn_drop(attention_scores_masked)
                y_t = attention_scores_masked @ window_value_masked

                output[i, mask_ind_i] = y_t.view(
                    -1,
                    self.num_attention_heads,
                    timesteps,
                    window_height * window_width,
                    channel_head,
                )

            unmask_ind_i = (mask[i] == 0).nonzero(as_tuple=False).view(-1)
            # [batch_size, n_window_height*n_window_width, num_attention_heads, timesteps, window_height*window_width, channel_head]
            window_query_unmasked = window_query[i, unmask_ind_i]
            window_key_unmasked = window_key[i, unmask_ind_i, :, :, : window_height * window_width]
            window_value_unmasked = window_value[i, unmask_ind_i, :, :, : window_height * window_width]

            attention_scores_unmasked = (window_query_unmasked @ window_key_unmasked.transpose(-2, -1)) * (
                1.0 / math.sqrt(window_query_unmasked.size(-1))
            )
            attention_scores_unmasked = F.softmax(attention_scores_unmasked, dim=-1)
            attention_scores_unmasked = self.attn_drop(attention_scores_unmasked)
            y_s = attention_scores_unmasked @ window_value_unmasked
            output[i, unmask_ind_i] = y_s
        if output_attentions:
            all_self_attentions = all_self_attentions + (attention_scores_masked,) + (attention_scores_unmasked,)
        output = output.view(
            batch_size,
            n_window_height,
            n_window_width,
            self.num_attention_heads,
            timesteps,
            window_height,
            window_width,
            channel_head,
        )
        output = (
            output.permute(0, 4, 1, 5, 2, 6, 3, 7)
            .contiguous()
            .view(batch_size, timesteps, new_height, new_width, num_channels)
        )

        if padding_right > 0 or padding_bottom > 0:
            output = output[:, :, :height, :width, :]

        output = self.proj_drop(self.proj(output))
        return output, all_self_attentions


class ProPainterFusionFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_dim: int = 1960,
        token_to_token_params: Dict = None,
    ):
        super().__init__()
        # We set hidden_dim as a default to 1960
        self.fc1 = nn.Sequential(nn.Linear(hidden_size, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, hidden_size))
        assert token_to_token_params is not None
        self.token_to_token_params = token_to_token_params
        self.kernel_shape = reduce((lambda x, y: x * y), token_to_token_params["kernel_size"])  # 49

    def forward(self, hidden_state, output_size):
        num_vecs = 1
        for i, d in enumerate(self.token_to_token_params["kernel_size"]):
            num_vecs *= int(
                (output_size[i] + 2 * self.token_to_token_params["padding"][i] - (d - 1) - 1)
                / self.token_to_token_params["stride"][i]
                + 1
            )

        hidden_state = self.fc1(hidden_state)
        batch_size, timestep, num_channel = hidden_state.size()
        normalizer = (
            hidden_state.new_ones(batch_size, timestep, self.kernel_shape)
            .view(-1, num_vecs, self.kernel_shape)
            .permute(0, 2, 1)
        )
        normalizer = F.fold(
            normalizer,
            output_size=output_size,
            kernel_size=self.token_to_token_params["kernel_size"],
            padding=self.token_to_token_params["padding"],
            stride=self.token_to_token_params["stride"],
        )

        hidden_state = F.fold(
            hidden_state.view(-1, num_vecs, num_channel).permute(0, 2, 1),
            output_size=output_size,
            kernel_size=self.token_to_token_params["kernel_size"],
            padding=self.token_to_token_params["padding"],
            stride=self.token_to_token_params["stride"],
        )
        hidden_state = (
            F.unfold(
                hidden_state / normalizer,
                kernel_size=self.token_to_token_params["kernel_size"],
                padding=self.token_to_token_params["padding"],
                stride=self.token_to_token_params["stride"],
            )
            .permute(0, 2, 1)
            .contiguous()
            .view(batch_size, timestep, num_channel)
        )
        hidden_state = self.fc2(hidden_state)

        return hidden_state


class ProPainterTemporalSparseTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        window_size: Tuple[int, int],
        pool_size: Tuple[int, int],
        layer_norm=nn.LayerNorm,
        token_to_token_params=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.attention = ProPainterSparseWindowAttention(hidden_size, num_attention_heads, window_size, pool_size)
        self.layer_norm1 = layer_norm(hidden_size)
        self.layer_norm2 = layer_norm(hidden_size)
        self.mlp = ProPainterFusionFeedForward(hidden_size, token_to_token_params=token_to_token_params)

    def forward(
        self,
        image_tokens,
        fold_x_size,
        mask=None,
        token_indices=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            image_tokens: shape [batch_size, timesteps, height, width, num_channels]
            fold_x_size: fold feature size, shape [60 108]
            mask: mask tokens, shape [batch_size, timesteps, height, width, 1]
        Returns:
            out_tokens: shape [batch_size, timesteps, height, width, 1]
        """

        batch_size, timesteps, height, width, num_channels = image_tokens.shape  # 20 36

        shortcut = image_tokens
        image_tokens = self.layer_norm1(image_tokens)
        att_x, all_self_attentions = self.attention(
            image_tokens, mask, token_indices, output_attentions=output_attentions
        )

        image_tokens = shortcut + att_x
        y = self.layer_norm2(image_tokens)
        hidden_states = self.mlp(y.view(batch_size, timesteps * height * width, num_channels), fold_x_size)
        hidden_states = hidden_states.view(batch_size, timesteps, height, width, num_channels)

        image_tokens = image_tokens + hidden_states

        return image_tokens, all_self_attentions


class ProPainterTemporalSparseTransformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        window_size: Tuple[int, int],
        pool_size: Tuple[int, int],
        num_hidden_layers: int,
        token_to_token_params: Dict = None,
    ):
        super().__init__()
        blocks = []
        for _ in range(num_hidden_layers):
            blocks.append(
                ProPainterTemporalSparseTransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    window_size,
                    pool_size,
                    token_to_token_params=token_to_token_params,
                )
            )
        self.transformer = nn.Sequential(*blocks)
        self.num_hidden_layers = num_hidden_layers

    def forward(
        self,
        image_tokens,
        fold_x_size,
        local_mask=None,
        t_dilation=2,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """
        Args:
            image_tokens: shape [batch_size, timesteps, height, width, num_channels]
            fold_x_size: fold feature size, shape [60 108]
            local_mask: local mask tokens, shape [batch_size, timesteps, height, width, 1]
        Returns:
            out_tokens: shape [batch_size, timesteps, height, width, num_channels]
        """
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        assert self.num_hidden_layers % t_dilation == 0, "wrong t_dilation input."
        timesteps = image_tokens.size(1)
        token_indices = [torch.arange(i, timesteps, t_dilation) for i in range(t_dilation)] * (
            self.num_hidden_layers // t_dilation
        )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (image_tokens,)

        for i in range(0, self.num_hidden_layers):
            image_tokens, _all_self_attentions = self.transformer[i](
                image_tokens,
                fold_x_size,
                local_mask,
                token_indices[i],
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_self_attentions = all_self_attentions + (_all_self_attentions,)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (image_tokens,)
        return image_tokens, all_hidden_states, all_self_attentions


class ProPainterInpaintGenerator(nn.Module):
    def __init__(self, config: ProPainterConfig):
        super().__init__()

        self.config = config
        self.encoder = ProPainterEncoder(config)

        # decoder
        self.decoder = nn.Sequential(
            ProPainterDeconv(
                config.num_channels,
                config.num_channels,
                kernel_size=config.patch_size,
                padding=config.padding,
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            nn.Conv2d(
                config.num_channels,
                config.in_channels[0],
                kernel_size=config.patch_size,
                stride=1,
                padding=config.padding,
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            ProPainterDeconv(
                config.in_channels[0],
                config.in_channels[0],
                kernel_size=config.patch_size,
                padding=config.padding,
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            nn.Conv2d(
                config.in_channels[0],
                out_channels=3,
                kernel_size=config.patch_size,
                stride=config.multi_level_conv_stride[0],
                padding=config.padding,
            ),
        )

        # soft split and soft composition
        token_to_token_params = {
            "kernel_size": config.kernel_size,
            "stride": config.conv2d_stride,
            "padding": config.padding_inpaint_generator,
        }
        self.soft_split = ProPainterSoftSplit(config)

        self.soft_comp = ProPainterSoftComp(config)

        self.max_pool = nn.MaxPool2d(config.kernel_size, config.conv2d_stride, config.padding_inpaint_generator)

        # feature propagation module
        self.img_prop_module = ProPainterBidirectionalPropagationInPaint(
            config, num_channels=config.num_channels_img_prop_module, learnable=False
        )
        self.feature_propagation_module = ProPainterBidirectionalPropagationInPaint(
            config, config.num_channels, learnable=True
        )

        self.transformers = ProPainterTemporalSparseTransformer(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            window_size=config.window_size,
            pool_size=config.pool_size,
            num_hidden_layers=config.num_hidden_layers,
            token_to_token_params=token_to_token_params,
        )

    def img_propagation(self, masked_frames, completed_flows, masks, interpolation="nearest"):
        _, _, prop_frames, updated_masks = self.img_prop_module(
            masked_frames, completed_flows[0], completed_flows[1], masks, interpolation
        )

        return prop_frames, updated_masks

    def forward(
        self,
        masked_frames,
        completed_flows,
        masks_in,
        masks_updated,
        num_local_frames,
        interpolation="bilinear",
        t_dilation=2,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Args:
            masks_in: original mask
            masks_updated: updated mask after image propagation
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        local_timestep = num_local_frames
        batch_size, timestep, _, original_height, original_width = masked_frames.size()

        encoder_hidden_states = self.encoder(
            torch.cat(
                [
                    masked_frames.view(batch_size * timestep, 3, original_height, original_width),
                    masks_in.view(batch_size * timestep, 1, original_height, original_width),
                    masks_updated.view(batch_size * timestep, 1, original_height, original_width),
                ],
                dim=1,
            )
        )
        _, num_channels, height, width = encoder_hidden_states.size()
        local_features = encoder_hidden_states.view(batch_size, timestep, num_channels, height, width)[
            :, :local_timestep, ...
        ]
        reference_features = encoder_hidden_states.view(batch_size, timestep, num_channels, height, width)[
            :, local_timestep:, ...
        ]
        folded_feature_size = (height, width)

        downsampled_flows_forward = (
            F.interpolate(
                completed_flows[0].view(-1, 2, original_height, original_width),
                scale_factor=1 / 4,
                mode="bilinear",
                align_corners=False,
            ).view(batch_size, local_timestep - 1, 2, height, width)
            / 4.0
        )
        downsampled_flows_backward = (
            F.interpolate(
                completed_flows[1].view(-1, 2, original_height, original_width),
                scale_factor=1 / 4,
                mode="bilinear",
                align_corners=False,
            ).view(batch_size, local_timestep - 1, 2, height, width)
            / 4.0
        )
        downsampled_mask_input = F.interpolate(
            masks_in.reshape(-1, 1, original_height, original_width),
            scale_factor=1 / 4,
            mode="nearest",
        ).view(batch_size, timestep, 1, height, width)
        downsampled_mask_input_local = downsampled_mask_input[:, :local_timestep]
        downsampled_mask_updated_local = F.interpolate(
            masks_updated[:, :local_timestep].reshape(-1, 1, original_height, original_width),
            scale_factor=1 / 4,
            mode="nearest",
        ).view(batch_size, local_timestep, 1, height, width)

        if self.training:
            mask_pool_local = self.max_pool(downsampled_mask_input.view(-1, 1, height, width))
            mask_pool_local = mask_pool_local.view(
                batch_size, timestep, 1, mask_pool_local.size(-2), mask_pool_local.size(-1)
            )
        else:
            mask_pool_local = self.max_pool(downsampled_mask_input_local.view(-1, 1, height, width))
            mask_pool_local = mask_pool_local.view(
                batch_size,
                local_timestep,
                1,
                mask_pool_local.size(-2),
                mask_pool_local.size(-1),
            )

        propagated_mask_input = torch.cat([downsampled_mask_input_local, downsampled_mask_updated_local], dim=2)
        _, _, local_features, _ = self.feature_propagation_module(
            local_features, downsampled_flows_forward, downsampled_flows_backward, propagated_mask_input, interpolation
        )
        encoder_hidden_states = torch.cat((local_features, reference_features), dim=1)

        transformed_features = self.soft_split(
            encoder_hidden_states.view(-1, num_channels, height, width),
            batch_size,
            folded_feature_size,
        )
        mask_pool_local = mask_pool_local.permute(0, 1, 3, 4, 2).contiguous()
        transformed_features, all_hidden_states, all_self_attentions = self.transformers(
            transformed_features,
            folded_feature_size,
            mask_pool_local,
            t_dilation=t_dilation,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        transformed_features = self.soft_comp(transformed_features, timestep, folded_feature_size)
        transformed_features = transformed_features.view(batch_size, timestep, -1, height, width)

        encoder_hidden_states = encoder_hidden_states + transformed_features

        if self.training:
            output = self.decoder(encoder_hidden_states.view(-1, num_channels, height, width))
            output = torch.tanh(output).view(batch_size, timestep, 3, original_height, original_width)
        else:
            output = self.decoder(encoder_hidden_states[:, :local_timestep].view(-1, num_channels, height, width))
            output = torch.tanh(output).view(batch_size, local_timestep, 3, original_height, original_width)

        if not return_dict:
            return tuple(v for v in [output, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=output,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ProPainterSpectralNorm(object):
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 1

    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.

    def __init__(self, name="weight", num_power_iterations=1, dimension=0, eps=1e-12):
        self.name = name
        self.dimension = dimension
        if num_power_iterations <= 0:
            raise ValueError(
                "Expected num_power_iterations to be positive, but " "got num_power_iterations={}".format(
                    num_power_iterations
                )
            )
        self.num_power_iterations = num_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dimension != 0:
            # permute dimension to front
            weight_mat = weight_mat.permute(
                self.dimension,
                *[d for d in range(weight_mat.dim()) if d != self.dimension],
            )
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, do_power_iteration):
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.num_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.num_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_v")
        delattr(module, self.name + "_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, num_power_iterations, dimension, eps):
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, ProPainterSpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on " "the same parameter {}".format(name))

        func = ProPainterSpectralNorm(name, num_power_iterations, dimension, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = func.reshape_weight_to_matrix(weight)

            height, width = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=func.eps)
            v = normalize(weight.new_empty(width).normal_(0, 1), dim=0, eps=func.eps)

        delattr(module, func.name)
        module.register_parameter(func.name + "_orig", weight)
        # We still need to assign weight back as func.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, func.name, weight.data)
        module.register_buffer(func.name + "_u", u)
        module.register_buffer(func.name + "_v", v)

        module.register_forward_pre_hook(func)

        module._register_state_dict_hook(ProPainterSpectralNormStateDictHook(func))
        module._register_load_state_dict_pre_hook(ProPainterSpectralNormLoadStateDictPreHook(func))
        return func


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class ProPainterSpectralNormLoadStateDictPreHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, func):
        self.func = func

    def __call__(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        func = self.func
        version = local_metadata.get("spectral_norm", {}).get(func.name + ".version", None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + func.name + "_orig"]
                weight_mat = func.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + func.name + "_u"]
                _, _ = weight_mat, u


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class ProPainterSpectralNormStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, func):
        self.func = func

    def __call__(self, module, state_dict, prefix, local_metadata):
        if "spectral_norm" not in local_metadata:
            local_metadata["spectral_norm"] = {}
        key = self.func.name + ".version"
        if key in local_metadata["spectral_norm"]:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata["spectral_norm"][key] = self.func._version


def spectral_norm(module, name="weight", num_power_iterations=1, eps=1e-12, dimension=None):
    r"""Applies spectral normalization to a parameter in the given module.

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        num_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dimension (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dimension is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dimension = 1
        else:
            dimension = 0
    ProPainterSpectralNorm.apply(module, name, num_power_iterations, dimension, eps)
    return module


#  ProPainterDiscriminator for Temporal Patch GAN
class ProPainterDiscriminator(nn.Module):
    def __init__(
        self,
        config: ProPainterConfig,
        in_channels: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.config = config
        num_features = config.num_channels // 4

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=num_features * 1,
                    kernel_size=config.kernel_size_3d_discriminator,
                    stride=config.multi_level_conv_stride,
                    padding=config.padding,
                    bias=not use_spectral_norm,
                )
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    num_features * 1,
                    num_features * 2,
                    kernel_size=config.kernel_size_3d_discriminator,
                    stride=config.multi_level_conv_stride,
                    padding=config.multi_level_conv_stride,
                    bias=not use_spectral_norm,
                )
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=config.kernel_size_3d_discriminator,
                    stride=config.multi_level_conv_stride,
                    padding=config.multi_level_conv_stride,
                    bias=not use_spectral_norm,
                )
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    num_features * 4,
                    num_features * 4,
                    kernel_size=config.kernel_size_3d_discriminator,
                    stride=config.multi_level_conv_stride,
                    padding=config.multi_level_conv_stride,
                    bias=not use_spectral_norm,
                )
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    num_features * 4,
                    num_features * 4,
                    kernel_size=config.kernel_size_3d_discriminator,
                    stride=config.multi_level_conv_stride,
                    padding=config.multi_level_conv_stride,
                    bias=not use_spectral_norm,
                )
            ),
            nn.LeakyReLU(config.negative_slope_default, inplace=True),
            nn.Conv3d(
                num_features * 4,
                num_features * 4,
                kernel_size=config.kernel_size_3d_discriminator,
                stride=config.multi_level_conv_stride,
                padding=config.multi_level_conv_stride,
            ),
        )

    def forward(self, completed_frames):
        completed_frames_t = torch.transpose(completed_frames, 1, 2)
        hidden_states = self.conv(completed_frames_t)
        if self.config.gan_loss != "hinge":
            hidden_states = torch.sigmoid(hidden_states)
        hidden_states = torch.transpose(hidden_states, 1, 2)  # batch_size, timesteps, num_channels, height, width
        return hidden_states


# Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/pretrained_networks.py
class ProPainterVgg16(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True, is_training: bool = False):
        super().__init__()
        self.is_training = is_training
        self.requires_grad = requires_grad
        self.pretrained = pretrained
        # This attribute will initiate lazy loading for such a huge model to save on memory and prevent OOM in cases.
        self.vgg_initialized = False  # Will still lazy load if training

    def _init_vgg(self):
        vgg_pretrained_features = tv.vgg16(pretrained=self.pretrained).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.vgg_initialized = True

    def forward(self, frames):
        device = frames.device
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])

        # Skip VGG16 initialization if not in training mode
        if self.is_training:
            if not self.vgg_initialized:
                self._init_vgg()
            self.to(device)
            hidden_states = self.slice1(frames)
            hidden_states_relu1_2 = hidden_states
            hidden_states = self.slice2(hidden_states)
            hidden_states_relu2_2 = hidden_states
            hidden_states = self.slice3(hidden_states)
            hidden_states_relu3_3 = hidden_states
            hidden_states = self.slice4(hidden_states)
            hidden_states_relu4_3 = hidden_states
            hidden_states = self.slice5(hidden_states)
            hidden_states_relu5_3 = hidden_states
            hidden_states = vgg_outputs(
                hidden_states_relu1_2,
                hidden_states_relu2_2,
                hidden_states_relu3_3,
                hidden_states_relu4_3,
                hidden_states_relu5_3,
            )
        else:
            # In inference mode, return dummy tensors with the same shape as the VGG outputs
            batch_size, _, H, W = frames.size()
            device = frames.device

            slice1_output = torch.zeros((batch_size, 64, H // 1, W // 1), device=device)
            slice2_output = torch.zeros((batch_size, 128, H // 2, W // 2), device=device)
            slice3_output = torch.zeros((batch_size, 256, H // 4, W // 4), device=device)
            slice4_output = torch.zeros((batch_size, 512, H // 8, W // 8), device=device)
            slice5_output = torch.zeros((batch_size, 512, H // 16, W // 16), device=device)

            # Return namedtuple with dummy outputs
            hidden_states = vgg_outputs(slice1_output, slice2_output, slice3_output, slice4_output, slice5_output)

        return hidden_states


# Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
class ProPainterScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, frames):
        device = frames.device
        shift = self.shift.to(device)
        scale = self.scale.to(device)
        return (frames - shift) / scale


# Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
class ProPainterIntermediateLossLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, num_channels: int, use_dropout: bool = False):
        super().__init__()

        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(num_channels, num_channels, 1, stride=1, padding=0, bias=False),
        ]
        self.loss_layers = nn.Sequential(*layers)

    def forward(self, hidden_states):
        return self.loss_layers(hidden_states)


def spatial_average(input_tensor, keepdim=True):
    return input_tensor.mean([2, 3], keepdim=keepdim)


def upsample(input_tensor, out_HW=(64, 64)):  # assumes scale factor is same for height and W
    return nn.Upsample(size=out_HW, mode="bilinear", align_corners=False)(input_tensor)


def normalize_tensor(hidden_states, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(hidden_states**2, dim=1, keepdim=True))
    return hidden_states / (norm_factor + eps)


# Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
# Learned perceptual metric
class ProPainterLpips(nn.Module):
    def __init__(
        self,
        config: ProPainterConfig,
        use_dropout: bool = True,
        is_training: bool = False,
    ):
        """Initializes a perceptual loss torch.nn.Module
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super().__init__()
        self.config = config
        self.scaling_layer = ProPainterScalingLayer()

        self.num_channels = [
            config.num_channels // 2,
            config.num_channels,
            config.num_channels * 2,
            config.num_channels * 4,
            config.num_channels * 4,
        ]
        self.length = len(self.num_channels)

        self.network = ProPainterVgg16(is_training=is_training)

        if is_training:
            use_dropout = True
        else:
            use_dropout = False

        self.layer0 = ProPainterIntermediateLossLayer(self.num_channels[0], use_dropout=use_dropout)
        self.layer1 = ProPainterIntermediateLossLayer(self.num_channels[1], use_dropout=use_dropout)
        self.layer2 = ProPainterIntermediateLossLayer(self.num_channels[2], use_dropout=use_dropout)
        self.layer3 = ProPainterIntermediateLossLayer(self.num_channels[3], use_dropout=use_dropout)
        self.layer4 = ProPainterIntermediateLossLayer(self.num_channels[4], use_dropout=use_dropout)
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, frames, pred_images):
        device = frames.device
        self.layers.to(device)
        frames = 2 * frames - 1
        pred_images = 2 * pred_images - 1

        frames, pred_images = (
            self.scaling_layer(frames),
            self.scaling_layer(pred_images),
        )
        hidden_states0, hidden_states1 = self.network.forward(frames), self.network.forward(pred_images)
        feats0, feats1, diffs = {}, {}, {}

        for i in range(self.length):
            feats0[i], feats1[i] = normalize_tensor(hidden_states0[i]), normalize_tensor(hidden_states1[i])
            diffs[i] = (feats0[i] - feats1[i]) ** 2

        layer_perceptual_losses = [
            spatial_average(self.layers[i](diffs[i]), keepdim=True).mean() for i in range(self.length)
        ]

        return sum(layer_perceptual_losses)


class ProPainterLpipsLoss(nn.Module):
    def __init__(
        self,
        config: ProPainterConfig,
        loss_weight: float = 1.0,
        use_input_norm: bool = True,
        range_norm: bool = False,
        is_training: bool = False,
    ):
        super().__init__()
        self.config = config
        self.perceptual = ProPainterLpips(config, is_training=is_training).eval()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred_images, frames):
        device = pred_images.device
        mean = self.mean.to(device)
        std = self.std.to(device)

        if self.range_norm:
            pred_images = (pred_images + 1) / 2
            frames = (frames + 1) / 2
        if self.use_input_norm:
            pred_images = (pred_images - mean) / std
            frames = (frames - mean) / std
        lpips_loss = self.perceptual(frames.contiguous(), pred_images.contiguous())
        return self.loss_weight * lpips_loss.mean(), None


class ProPainterAdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(
        self,
        type: str = "nsgan",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ):
        r"""
        type = nsgan | lsgan | hinge
        """
        super().__init__()
        self.type = type
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        if type == "nsgan":
            self.criterion = nn.BCELoss()
        elif type == "lsgan":
            self.criterion = nn.MSELoss()
        elif type == "hinge":
            self.criterion = nn.ReLU()

    def __call__(self, generated_frames, is_real, is_disc=None):
        device = generated_frames.device
        real_label = self.real_label.to(device)
        fake_label = self.fake_label.to(device)
        if self.type == "hinge":
            if is_disc:
                if is_real:
                    generated_frames = -generated_frames
                return self.criterion(1 + generated_frames).mean()
            else:
                return (-generated_frames).mean()
        else:
            labels = (real_label if is_real else fake_label).expand_as(generated_frames)
            loss = self.criterion(generated_frames, labels)
            return loss


def create_mask(flow, paddings):
    """
    flow shape: [batch_size, num_channels, height, width]
    paddings: [2 x 2] shape list, the first row indicates up and down paddings
    the second row indicates left and right paddings
    |            |
    |       x    |
    |     x * x  |
    |       x    |
    |            |
    """
    shape = flow.shape
    inner_height = shape[2] - (paddings[0][0] + paddings[0][1])
    inner_width = shape[3] - (paddings[1][0] + paddings[1][1])
    inner = torch.ones([inner_height, inner_width])
    torch_paddings = [
        paddings[1][0],
        paddings[1][1],
        paddings[0][0],
        paddings[0][1],
    ]  # left, right, up and down
    mask2d = F.pad(inner, pad=torch_paddings)
    mask3d = mask2d.unsqueeze(0).repeat(shape[0], 1, 1)
    mask4d = mask3d.unsqueeze(1)
    return mask4d.detach()


def smoothness_deltas(config: ProPainterConfig, flow):
    """
    flow: [batch_size, num_channels, height, width]
    """
    mask_x = create_mask(flow, [[0, 0], [0, 1]])
    mask_y = create_mask(flow, [[0, 1], [0, 0]])
    mask = torch.cat((mask_x, mask_y), dim=1)
    mask = mask.to(flow.device)
    filter_x = torch.tensor([[0, 0, 0.0], [0, 1, -1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 0, 0.0], [0, 1, 0], [0, -1, 0]])
    weights = torch.ones([2, 1, 3, 3])
    weights[0, 0] = filter_x
    weights[1, 0] = filter_y
    weights = weights.to(flow.device)

    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=config.padding)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=config.padding)
    return delta_u, delta_v, mask


def charbonnier_loss(delta, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """
    Compute the generalized charbonnier loss of the difference tensor x
    All positions where mask == 0 are not taken into account
    delta: a tensor of shape [batch_size, num_channels, height, width]
    mask: a mask of shape [batch_size, mc, height, width], where mask channels must be either 1 or the same as
    the number of channels of delta. Entries should be 0 or 1
    return: loss
    """
    batch_size, num_channels, height, width = delta.shape
    norm = batch_size * num_channels * height * width
    error = torch.pow(torch.square(delta * beta) + torch.square(torch.tensor(epsilon)), alpha)
    if mask is not None:
        error = mask * error
    if truncate is not None:
        error = torch.min(error, truncate)
    return torch.sum(error) / norm


def second_order_deltas(config: ProPainterConfig, flow):
    """
    consider the single flow first
    flow shape: [batch_size, num_channels, height, width]
    """
    # create mask
    mask_x = create_mask(flow, [[0, 0], [1, 1]])
    mask_y = create_mask(flow, [[1, 1], [0, 0]])
    mask_diag = create_mask(flow, [[1, 1], [1, 1]])
    mask = torch.cat((mask_x, mask_y, mask_diag, mask_diag), dim=1)
    mask = mask.to(flow.device)

    filter_x = torch.tensor([[0, 0, 0.0], [1, -2, 1], [0, 0, 0]])
    filter_y = torch.tensor([[0, 1, 0.0], [0, -2, 0], [0, 1, 0]])
    filter_diag1 = torch.tensor([[1, 0, 0.0], [0, -2, 0], [0, 0, 1]])
    filter_diag2 = torch.tensor([[0, 0, 1.0], [0, -2, 0], [1, 0, 0]])
    weights = torch.ones([4, 1, 3, 3])
    weights[0] = filter_x
    weights[1] = filter_y
    weights[2] = filter_diag1
    weights[3] = filter_diag2
    weights = weights.to(flow.device)

    # split the flow into flow_u and flow_v, conv them with the weights
    flow_u, flow_v = torch.split(flow, split_size_or_sections=1, dim=1)
    delta_u = F.conv2d(flow_u, weights, stride=1, padding=config.padding)
    delta_v = F.conv2d(flow_v, weights, stride=1, padding=config.padding)
    return delta_u, delta_v, mask


def smoothness_loss(config, flow, cmask):
    delta_u, delta_v, _ = smoothness_deltas(config, flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v


def second_order_loss(config, flow, cmask):
    delta_u, delta_v, _ = second_order_deltas(config, flow)
    loss_u = charbonnier_loss(delta_u, cmask)
    loss_v = charbonnier_loss(delta_v, cmask)
    return loss_u + loss_v


def convert_rgb_to_grayscale(image, rgb_weights=None):
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, height, width). Got {image.shape}")

    if rgb_weights is None:
        # 8 bit images
        if image.dtype == torch.uint8:
            rgb_weights = torch.tensor([76, 150, 29], device=image.device, dtype=torch.uint8)
        # floating point images
        elif image.dtype in (torch.float16, torch.float32, torch.float64):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")
    else:
        # is tensor that we make sure is in the same device/dtype
        rgb_weights = rgb_weights.to(image)

    # unpack the color image channels with RGB order
    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    w_r, w_g, w_b = rgb_weights.unbind()
    return w_r * r + w_g * g + w_b * b


def ternary_transform(config: ProPainterConfig, image, max_distance=1):
    device = image.device
    patch_size = 2 * max_distance + 1
    intensities = convert_rgb_to_grayscale(image) * 255
    out_channels = patch_size * patch_size
    weights = np.eye(out_channels).reshape(out_channels, 1, patch_size, patch_size)
    weights = torch.from_numpy(weights).float().to(device)
    patches = F.conv2d(intensities, weights, stride=1, padding=config.padding)
    transf = patches - intensities
    transf_norm = transf / torch.sqrt(0.81 + torch.square(transf))
    return transf_norm


def hamming_distance(ternary_transform_frame1, ternary_transform_frame2):
    distance = torch.square(ternary_transform_frame1 - ternary_transform_frame2)
    distance_norm = distance / (0.1 + distance)
    distance_sum = torch.sum(distance_norm, dim=1, keepdim=True)
    return distance_sum


def ternary_loss(config, flow_computed, flow_ground_truth, mask, current_frame, shift_frame, scale_factor=1):
    if scale_factor != 1:
        current_frame = F.interpolate(current_frame, scale_factor=1 / scale_factor, mode="bilinear")
        shift_frame = F.interpolate(shift_frame, scale_factor=1 / scale_factor, mode="bilinear")
    warped_sc = flow_warp(shift_frame, flow_ground_truth.permute(0, 2, 3, 1))
    confidence_mask = torch.exp(-50.0 * torch.sum(torch.abs(current_frame - warped_sc), dim=1).pow(2)).unsqueeze(1)
    warped_comp_sc = flow_warp(shift_frame, flow_computed.permute(0, 2, 3, 1))

    ternary_transform1 = ternary_transform(
        config, current_frame
    )  # current_frame: [batch_size * timesteps, num_channels, height, width]
    ternary_transform21 = ternary_transform(
        config, warped_comp_sc
    )  # warped_comp_sc: [batch_size * timesteps, num_channels, height, width]
    dist = hamming_distance(ternary_transform1, ternary_transform21)
    loss = torch.mean(dist * confidence_mask * mask) / torch.mean(
        mask
    )  # confidence_mask, mask: [batch_size * timesteps, num_channels, height, width]

    return loss


class ProPainterFlowLoss(nn.Module):
    def __init__(self, config: ProPainterConfig):
        super().__init__()
        self.config = config
        self.l1_criterion = nn.L1Loss()

    def forward(self, pred_flows, ground_truth_flows, masks, frames):
        loss = 0
        warp_loss = 0
        height, width = pred_flows[0].shape[-2:]
        masks = [masks[:, :-1, ...].contiguous(), masks[:, 1:, ...].contiguous()]
        frames0 = frames[:, :-1, ...]
        frames1 = frames[:, 1:, ...]
        current_frames = [frames0, frames1]
        next_frames = [frames1, frames0]
        for i in range(len(pred_flows)):
            combined_flow = pred_flows[i] * masks[i] + ground_truth_flows[i] * (1 - masks[i])
            l1_loss = self.l1_criterion(pred_flows[i] * masks[i], ground_truth_flows[i] * masks[i]) / torch.mean(
                masks[i]
            )
            l1_loss += self.l1_criterion(
                pred_flows[i] * (1 - masks[i]), ground_truth_flows[i] * (1 - masks[i])
            ) / torch.mean((1 - masks[i]))

            smooth_loss = smoothness_loss(
                self.config,
                combined_flow.reshape(-1, 2, height, width),
                masks[i].reshape(-1, 1, height, width),
            )
            smooth_loss2 = second_order_loss(
                self.config,
                combined_flow.reshape(-1, 2, height, width),
                masks[i].reshape(-1, 1, height, width),
            )

            warp_loss_i = ternary_loss(
                self.config,
                combined_flow.reshape(-1, 2, height, width),
                ground_truth_flows[i].reshape(-1, 2, height, width),
                masks[i].reshape(-1, 1, height, width),
                current_frames[i].reshape(-1, 3, height, width),
                next_frames[i].reshape(-1, 3, height, width),
            )

            loss += l1_loss + smooth_loss + smooth_loss2

            warp_loss += warp_loss_i

        return loss, warp_loss


class ProPainterEdgeLoss(nn.Module):
    def __init__(self, config: ProPainterConfig):
        super().__init__()
        self.config = config

    def edgeLoss(self, pred_edges, edges):
        """

        Args:
            pred_edges: with shape [batch_size, num_channels, height, width]
            edges: with shape [batch_size, num_channels, height, width]

        Returns: Edge losses

        """
        mask = (edges > 0.5).float()
        _, num_channels, height, width = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [batch_size,].
        num_neg = num_channels * height * width - num_pos  # Shape: [batch_size,].
        neg_weights = (num_neg / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        pos_weights = (num_pos / (num_pos + num_neg)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        weight = neg_weights * mask + pos_weights * (1 - mask)  # weight for debug
        losses = F.binary_cross_entropy_with_logits(pred_edges.float(), edges.float(), weight=weight, reduction="none")
        loss = torch.mean(losses)
        return loss

    def forward(self, pred_edges, gt_edges, masks):
        loss = 0
        height, width = pred_edges[0].shape[-2:]
        masks = [masks[:, :-1, ...].contiguous(), masks[:, 1:, ...].contiguous()]
        for i in range(len(pred_edges)):
            combined_edge = pred_edges[i] * masks[i] + gt_edges[i] * (1 - masks[i])
            edge_loss = self.edgeLoss(
                pred_edges[i].reshape(-1, 1, height, width),
                gt_edges[i].reshape(-1, 1, height, width),
            ) + 5 * self.edgeLoss(
                combined_edge.reshape(-1, 1, height, width),
                gt_edges[i].reshape(-1, 1, height, width),
            )
            loss += edge_loss

        return loss


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    offsets = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        offsets = offsets + 0.5

    gauss = torch.exp((-offsets.pow(2.0) / (2 * sigma**2)).float())
    return gauss / gauss.sum()


def get_gaussian_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size: filter size. It should be odd and positive.
        sigma: gaussian standard deviation.
        force_even: overrides requirement for odd kernel size.

    Returns:
        1D tensor with gaussian filter coefficients.
    """
    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def filter2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
) -> torch.Tensor:
    r"""Convolve a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth num_channels of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(batch_size, num_channels, height, width)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(batch_size, num_channels, height, width)`.
    """

    if border_type not in ["constant", "reflect", "replicate", "circular"]:
        raise ValueError(
            f"Invalid border type, we expect 'constant', \
        'reflect', 'replicate', 'circular'. Got:{border_type}"
        )

    if padding not in ["valid", "same"]:
        raise ValueError(f"Invalid padding mode, we expect 'valid' or 'same'. Got: {padding}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if (not len(kernel.shape) == 3) and not ((kernel.shape[0] == 0) or (kernel.shape[0] == input.shape[0])):
        raise ValueError(f"Invalid kernel shape, we expect 1xHxW or BxHxW. Got: {kernel.shape}")

    # prepare kernel
    batch_size, num_channels, height, width = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, num_channels, -1, -1)

    height_, width_ = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == "same":
        padding_shape: List[int] = _compute_padding([height_, width_])
        input = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height_, width_)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    if padding == "same":
        out = output.view(batch_size, num_channels, height, width)
    else:
        out = output.view(batch_size, num_channels, height - height_ + 1, width - width_ + 1)

    return out


def gaussian_blur2d(
    input: torch.Tensor,
    kernel_size: Tuple[int, int],
    sigma: Tuple[float, float],
    border_type: str = "reflect",
    separable: bool = True,
) -> torch.Tensor:
    r"""Create an operator that blurs a tensor using a Gaussian filter.
    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each num_channels. It supports batched operation.

    Arguments:
        input: the input tensor with shape :math:`(batch_size,num_channels,height,width)`.
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of two 1d-convolutions.

    Returns:
        the blurred tensor with shape :math:`(batch_size, num_channels, height, width)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       gaussian_blur.html>`__.
    """
    if separable:
        kernel_x: torch.Tensor = get_gaussian_kernel1d(kernel_size[1], sigma[1])
        kernel_y: torch.Tensor = get_gaussian_kernel1d(kernel_size[0], sigma[0])
        # Convolve a tensor with two 1d kernels, in x and y directions.The kernel is applied
        # independently at each depth num_channels of the tensor. Before applying the
        # kernel, the function applies padding according to the specified mode so
        # that the output remains in the same shape.
        output_x = filter2d(
            input,
            kernel_x[None].unsqueeze(0),
            border_type,
            normalized=False,
            padding="same",
        )
        output = filter2d(
            output_x,
            kernel_y[None].unsqueeze(-1),
            border_type,
            normalized=False,
            padding="same",
        )
    else:
        # returns Gaussian filter matrix coefficients.
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise TypeError(f"kernel_size must be a tuple of length two. Got {kernel_size}")
        if not isinstance(sigma, tuple) or len(sigma) != 2:
            raise TypeError(f"sigma must be a tuple of length two. Got {sigma}")
        ksize_x, ksize_y = kernel_size
        sigma_x, sigma_y = sigma
        kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even=False)
        kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even=False)
        kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
        output = filter2d(input, kernel_2d[None], border_type)

    return output


def get_sobel_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3."""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def get_sobel_kernel_5x5_2nd_order() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, 0.0, 2.0, 0.0, -1.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-1.0, 0.0, 2.0, 0.0, -1.0],
        ]
    )


def _get_sobel_kernel_5x5_2nd_order_xy() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -4.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, -4.0, -2.0],
            [1.0, 2.0, 0.0, -2.0, -1.0],
        ]
    )


def get_diff_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3."""
    return torch.tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]])


def get_sobel_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_sobel_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_diff_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_diff_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = get_sobel_kernel_5x5_2nd_order()
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = _get_sobel_kernel_5x5_2nd_order_xy()
    return torch.stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]])
    return torch.stack([gxx, gxy, gyy])


def get_spatial_gradient_kernel2d(mode: str, order: int) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients, using one of the following operators:

    sobel, diff.
    """
    if mode not in ["sobel", "diff"]:
        raise TypeError(
            "mode should be either sobel\
                         or diff. Got {}".format(mode)
        )
    if order not in [1, 2]:
        raise TypeError(
            "order should be either 1 or 2\
                         Got {}".format(order)
        )
    if mode == "sobel" and order == 1:
        kernel: torch.Tensor = get_sobel_kernel2d()
    elif mode == "sobel" and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()
    elif mode == "diff" and order == 1:
        kernel = get_diff_kernel2d()
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel2d_2nd_order()
    else:
        raise NotImplementedError("")
    return kernel


def normalize_kernel2d(kernel: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    if len(kernel.size()) < 2:
        raise TypeError(f"kernel should be at least 2D tensor. Got {kernel.size()}")
    norm: torch.Tensor = kernel.abs().sum(dim=-1).sum(dim=-1)
    return kernel / (norm.unsqueeze(-1).unsqueeze(-1))


def spatial_gradient(
    input: torch.Tensor, mode: str = "sobel", order: int = 1, normalized: bool = True
) -> torch.Tensor:
    r"""Compute the first order image derivative in both x and y using a Sobel operator.

    Args:
        input: input image tensor with shape :math:`(batch_size, num_channels, height, width)`.
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.

    Return:
        the derivatives of the input feature map. with shape :math:`(B, C, 2, height, width)`.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    # allocate kernel
    kernel: torch.Tensor = get_spatial_gradient_kernel2d(mode, order)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # prepare kernel
    batch_size, num_channels, height, width = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input).detach()
    tmp_kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)

    # convolve input tensor with sobel kernel
    kernel_flip: torch.Tensor = tmp_kernel.flip(-3)

    # Pad with "replicate for spatial dims, but with zeros for num_channels
    spatial_pad = [
        kernel.size(1) // 2,
        kernel.size(1) // 2,
        kernel.size(2) // 2,
        kernel.size(2) // 2,
    ]
    out_channels: int = 3 if order == 2 else 2
    padded_inp: torch.Tensor = F.pad(
        input.reshape(batch_size * num_channels, 1, height, width),
        spatial_pad,
        "replicate",
    )[:, :, None]

    return F.conv3d(padded_inp, kernel_flip, padding=0).view(batch_size, num_channels, out_channels, height, width)


def get_canny_nms_kernel(device=torch.device("cpu"), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns 3x3 kernels for the Canny Non-maximal suppression."""
    kernel: torch.Tensor = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)


def get_hysteresis_kernel(device=torch.device("cpu"), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns the 3x3 kernels for the Canny hysteresis."""
    kernel: torch.Tensor = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)


class ProPainterCanny(nn.Module):
    r"""Module that finds edges of the input image and filters them using the Canny algorithm.

    Args:
        input: input image tensor with shape :math:`(B,C,height,width)`.
        low_threshold: lower threshold for the hysteresis procedure.
        high_threshold: upper threshold for the hysteresis procedure.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the canny edge magnitudes map, shape of :math:`(B,1,height,width)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,height,width)`.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = Canny()(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """

    def __init__(
        self,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: Tuple[float, float] = (1, 1),
        hysteresis: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if low_threshold > high_threshold:
            raise ValueError(
                "Invalid input thresholds. low_threshold should be\
                             smaller than the high_threshold. Got: {}>{}".format(low_threshold, high_threshold)
            )

        if low_threshold < 0 or low_threshold > 1:
            raise ValueError(f"Invalid input threshold. low_threshold should be in range (0,1). Got: {low_threshold}")

        if high_threshold < 0 or high_threshold > 1:
            raise ValueError(
                f"Invalid input threshold. high_threshold should be in range (0,1). Got: {high_threshold}"
            )

        # Gaussian blur parameters
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Double threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Hysteresis
        self.hysteresis = hysteresis

        self.eps: float = eps

    def canny(
        self,
        input: torch.Tensor,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: Tuple[float, float] = (1, 1),
        hysteresis: bool = True,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Find edges of the input image and filters them using the Canny algorithm.
        Args:
            input: input image tensor with shape :math:`(B,C,height,width)`.
            low_threshold: lower threshold for the hysteresis procedure.
            high_threshold: upper threshold for the hysteresis procedure.
            kernel_size: the size of the kernel for the gaussian blur.
            sigma: the standard deviation of the kernel for the gaussian blur.
            hysteresis: if True, applies the hysteresis edge tracking.
                Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
            eps: regularization number to avoid NaN during backprop.

        Returns:
            - the canny edge magnitudes map, shape of :math:`(B,1,height,width)`.
            - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,height,width)`.

        .. note::
        See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
        canny.html>`__.
        """
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

        if not len(input.shape) == 4:
            raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

        if low_threshold > high_threshold:
            raise ValueError(
                "Invalid input thresholds. low_threshold should be smaller than the high_threshold. Got: {}>{}".format(
                    low_threshold, high_threshold
                )
            )

        if low_threshold < 0 and low_threshold > 1:
            raise ValueError(f"Invalid input threshold. low_threshold should be in range (0,1). Got: {low_threshold}")

        if high_threshold < 0 and high_threshold > 1:
            raise ValueError(
                f"Invalid input threshold. high_threshold should be in range (0,1). Got: {high_threshold}"
            )

        device: torch.device = input.device
        dtype: torch.dtype = input.dtype

        # To Grayscale
        if input.shape[1] == 3:
            input = convert_rgb_to_grayscale(input)

        # Gaussian filter
        blurred: torch.Tensor = gaussian_blur2d(input, kernel_size, sigma)

        # Compute the gradients
        gradients: torch.Tensor = spatial_gradient(blurred, normalized=False)

        # Unpack the edges
        gx: torch.Tensor = gradients[:, :, 0]
        gy: torch.Tensor = gradients[:, :, 1]

        # Compute gradient magnitude and angle
        magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy + eps)
        angle: torch.Tensor = torch.atan2(gy, gx)

        # Radians to Degrees
        angle = 180.0 * angle / math.pi

        # Round angle to the nearest 45 degree
        angle = torch.round(angle / 45) * 45

        # Non-maximal suppression
        nms_kernels: torch.Tensor = get_canny_nms_kernel(device, dtype)
        nms_magnitude: torch.Tensor = F.conv2d(magnitude, nms_kernels, padding=nms_kernels.shape[-1] // 2)

        # Get the indices for both directions
        positive_idx: torch.Tensor = (angle / 45) % 8
        positive_idx = positive_idx.long()

        negative_idx: torch.Tensor = ((angle / 45) + 4) % 8
        negative_idx = negative_idx.long()

        # Apply the non-maximum suppression to the different directions
        channel_select_filtered_positive: torch.Tensor = torch.gather(nms_magnitude, 1, positive_idx)
        channel_select_filtered_negative: torch.Tensor = torch.gather(nms_magnitude, 1, negative_idx)

        channel_select_filtered: torch.Tensor = torch.stack(
            [channel_select_filtered_positive, channel_select_filtered_negative], 1
        )

        is_max: torch.Tensor = channel_select_filtered.min(dim=1)[0] > 0.0

        magnitude = magnitude * is_max

        # Threshold
        edges: torch.Tensor = F.threshold(magnitude, low_threshold, 0.0)

        low: torch.Tensor = magnitude > low_threshold
        high: torch.Tensor = magnitude > high_threshold

        edges = low * 0.5 + high * 0.5
        edges = edges.to(dtype)

        # Hysteresis
        if hysteresis:
            edges_old: torch.Tensor = -torch.ones(edges.shape, device=edges.device, dtype=dtype)
            hysteresis_kernels: torch.Tensor = get_hysteresis_kernel(device, dtype)

            while ((edges_old - edges).abs() != 0).any():
                weak: torch.Tensor = (edges == 0.5).float()
                strong: torch.Tensor = (edges == 1).float()

                hysteresis_magnitude: torch.Tensor = F.conv2d(
                    edges, hysteresis_kernels, padding=hysteresis_kernels.shape[-1] // 2
                )
                hysteresis_magnitude = (hysteresis_magnitude == 1).any(1, keepdim=True).to(dtype)
                hysteresis_magnitude = hysteresis_magnitude * weak + strong

                edges_old = edges.clone()
                edges = hysteresis_magnitude + (hysteresis_magnitude == 0) * weak * 0.5

            edges = hysteresis_magnitude

        return magnitude, edges

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.canny(
            input,
            self.low_threshold,
            self.high_threshold,
            self.kernel_size,
            self.sigma,
            self.hysteresis,
            self.eps,
        )


class ProPainterLosses:
    def __init__(self, config: ProPainterConfig, is_training: bool) -> None:
        self.config = config
        self.l1_loss = L1Loss()
        self.perc_loss = ProPainterLpipsLoss(config, use_input_norm=True, range_norm=True, is_training=is_training)
        self.adversarial_loss = ProPainterAdversarialLoss(type=config.gan_loss)
        self.flow_loss = ProPainterFlowLoss(config)
        self.edge_loss = ProPainterEdgeLoss(config)
        self.canny = ProPainterCanny(sigma=(2, 2), low_threshold=0.1, high_threshold=0.2)

    def get_edges(self, flows):
        # (batch_size, timesteps, 2, height, width)
        batch_size, timesteps, _, height, width = flows.shape
        flows = flows.view(-1, 2, height, width)
        flows_gray = (flows[:, 0, None] ** 2 + flows[:, 1, None] ** 2) ** 0.5
        if flows_gray.max() < 1:
            flows_gray = flows_gray * 0
        else:
            flows_gray = flows_gray / flows_gray.max()

        _, edges = self.canny(flows_gray.float())
        edges = edges.view(batch_size, timesteps, 1, height, width)
        return edges

    def calculate_losses(
        self,
        pred_imgs,
        masks_dilated,
        frames,
        comp_frames,
        discriminator,
        pred_flows_bidirectional,
        ground_truth_flows_bidirectional,
        flow_masks,
        pred_edges_bidirectional,
    ):
        _, _, _, height, width = frames.size()

        gt_edges_forward = self.get_edges(ground_truth_flows_bidirectional[0])
        gt_edges_backward = self.get_edges(ground_truth_flows_bidirectional[1])
        gt_edges_bidirectional = [gt_edges_forward, gt_edges_backward]

        gen_loss = 0
        dis_loss = 0
        # generator l1 loss
        hole_loss = self.l1_loss(pred_imgs * masks_dilated, frames * masks_dilated)
        hole_loss = hole_loss / torch.mean(masks_dilated) * self.config.hole_weight
        gen_loss += hole_loss

        valid_loss = self.l1_loss(pred_imgs * (1 - masks_dilated), frames * (1 - masks_dilated))
        valid_loss = valid_loss / torch.mean(1 - masks_dilated) * self.config.valid_weight
        gen_loss += valid_loss

        # perceptual loss
        if self.config.perceptual_weight > 0:
            perc_loss = (
                self.perc_loss(
                    pred_imgs.view(-1, 3, height, width),
                    frames.view(-1, 3, height, width),
                )[0]
                * self.config.perceptual_weight
            )
            gen_loss += perc_loss

        # gan loss
        if self.config.use_discriminator:
            # generator adversarial loss
            gen_clip = discriminator(comp_frames)
            gan_loss = self.adversarial_loss(gen_clip, True, False)
            gan_loss = gan_loss * self.config.adversarial_weight
            gen_loss += gan_loss

        if self.config.use_discriminator:
            # discriminator adversarial loss
            real_clip = discriminator(frames)
            fake_clip = discriminator(comp_frames.detach())
            dis_real_loss = self.adversarial_loss(real_clip, True, True)
            dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # these losses are for training flow completion network
        # compulte flow_loss
        flow_loss, warp_loss = self.flow_loss(
            pred_flows_bidirectional, ground_truth_flows_bidirectional, flow_masks, frames
        )
        flow_loss = flow_loss * self.config.flow_weight_flow_complete_net

        # compute edge loss
        edge_loss = self.edge_loss(pred_edges_bidirectional, gt_edges_bidirectional, flow_masks)
        edge_loss = edge_loss * 1.0

        flow_complete_loss = flow_loss + warp_loss * 0.01 + edge_loss
        return gen_loss, dis_loss, flow_complete_loss


class ProPainterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ProPainterConfig
    base_model_prefix = "propainter"
    main_input_name = "pixel_values_videos"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.Conv3d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, ProPainterSecondOrderDeformableAlignment) or isinstance(
            module, ProPainterDeformableAlignment
        ):
            num_channels = module.in_channels
            for k in module.kernel_size:
                num_channels *= k
            stdv = 1.0 / math.sqrt(num_channels)
            module.weight.data.uniform_(-stdv, stdv)
            if module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module.conv_offset[-1], "weight") and module.conv_offset[-1].weight is not None:
                TORCH_INIT_FUNCTIONS["constant_"](module.conv_offset[-1].weight, 0)
            if hasattr(module.conv_offset[-1], "bias") and module.conv_offset[-1].bias is not None:
                TORCH_INIT_FUNCTIONS["constant_"](module.conv_offset[-1].bias, 0)
        elif isinstance(module, ProPainterInpaintGenerator) or isinstance(module, ProPainterDiscriminator):
            for child in module.children():
                classname = child.__class__.__name__
                if classname.find("InstanceNorm2d") != -1:
                    if hasattr(child, "weight") and child.weight is not None:
                        nn.init.constant_(child.weight.data, 1.0)
                    if hasattr(child, "bias") and child.bias is not None:
                        nn.init.constant_(child.bias.data, 0.0)
                elif hasattr(child, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                    nn.init.normal_(child.weight.data, 0.0, 0.02)
                    if hasattr(child, "bias") and child.bias is not None:
                        nn.init.constant_(child.bias.data, 0.0)
        elif isinstance(module, ProPainterBasicEncoder):
            for child in module.children():
                if isinstance(child, nn.Conv2d):
                    nn.init.kaiming_normal_(child.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(child, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                    if child.weight is not None:
                        nn.init.constant_(child.weight, 1)
                    if child.bias is not None:
                        nn.init.constant_(child.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


PROPAINTER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ProPainterConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PROPAINTER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values for videos. Pixel values for videos can be obtained using [`AutoImageProcessor`]. See [`ProPainterVideoProcessor.__call__`]
            for details.
        flow_masks: (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values for flow masks. Pixel values for flow masks can be obtained using [`AutoImageProcessor`]. See [`ProPainterVideoProcessor.__call__`]
            for details.
        masks_dilated: (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values for dilated masks. Pixel values for dilated masks can be obtained using [`AutoImageProcessor`]. See [`ProPainterVideoProcessor.__call__`]
            for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ProPainter Model outputting composed frames without any specific head on top.",
    PROPAINTER_START_DOCSTRING,
)
class ProPainterModel(ProPainterPreTrainedModel):
    _tied_weights_keys = [
        "optical_flow_model.context_network.resblocks.2.norm3",
        "optical_flow_model.context_network.resblocks.2.downsample",
        "optical_flow_model.context_network.resblocks.4.norm3",
        "optical_flow_model.context_network.resblocks.4.downsample",
    ]

    def __init__(self, config: ProPainterConfig):
        super().__init__(config)
        self.config = config
        self.optical_flow_model = ProPainterRaftOpticalFlow(config)
        self.flow_completion_net = ProPainterRecurrentFlowCompleteNet(config)
        self.inpaint_generator = ProPainterInpaintGenerator(config)
        self.discriminator = ProPainterDiscriminator(config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _get_ref_index(self, mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    def _get_short_clip_len(self, width):
        if width <= 640:
            return 12
        elif width <= 720:
            return 8
        elif width <= 1280:
            return 4
        else:
            return 2

    def compute_flow(self, pixel_values_videos):
        if self.training:
            ground_truth_local_frames = pixel_values_videos[
                :, : self.config.num_local_frames_propainter, ...
            ]  # batch_size, temporal_length, num_channels, height, width (before slicing)
            # get gt optical flow
            if self.gradient_checkpointing:
                ground_truth_flows_bidirectional = self._gradient_checkpointing_func(
                    self.optical_flow_model.__call__,
                    ground_truth_local_frames,
                    self.config.raft_iter,
                )
            else:
                ground_truth_flows_bidirectional = self.optical_flow_model(
                    ground_truth_local_frames, iters=self.config.raft_iter
                )
        else:
            short_clip_len = self._get_short_clip_len(pixel_values_videos.size(-1))
            if pixel_values_videos.size(1) > short_clip_len:
                ground_truth_flows_forward_list, ground_truth_flows_backward_list = [], []
                for f in range(0, self.video_length, short_clip_len):
                    end_f = min(self.video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.optical_flow_model(
                            pixel_values_videos[:, f:end_f], iters=self.config.raft_iter
                        )
                    else:
                        flows_f, flows_b = self.optical_flow_model(
                            pixel_values_videos[:, f - 1 : end_f],
                            iters=self.config.raft_iter,
                        )
                    ground_truth_flows_forward_list.append(flows_f)
                    ground_truth_flows_backward_list.append(flows_b)
                    torch.cuda.empty_cache()

                ground_truth_flows_forward = torch.cat(ground_truth_flows_forward_list, dim=1)
                ground_truth_flows_backward = torch.cat(ground_truth_flows_backward_list, dim=1)
                ground_truth_flows_bidirectional = (ground_truth_flows_forward, ground_truth_flows_backward)
            else:
                ground_truth_flows_bidirectional = self.optical_flow_model(
                    pixel_values_videos, iters=self.config.raft_iter
                )
                torch.cuda.empty_cache()
        return ground_truth_flows_bidirectional

    def complete_flow(self, ground_truth_flows_bidirectional, flow_masks):
        if self.training:
            local_masks = flow_masks[:, : self.config.num_local_frames_propainter, ...].contiguous()
            if self.gradient_checkpointing:
                pred_flows_bidirectional, pred_edges_bidirectional = self._gradient_checkpointing_func(
                    self.flow_completion_net.forward_bidirectional_flow.__call__,
                    ground_truth_flows_bidirectional,
                    local_masks,
                )
            else:
                pred_flows_bidirectional, pred_edges_bidirectional = (
                    self.flow_completion_net.forward_bidirectional_flow(ground_truth_flows_bidirectional, local_masks)
                )
            pred_flows_bidirectional_loss = pred_flows_bidirectional
            pred_flows_bidirectional = self.flow_completion_net.combine_flow(
                ground_truth_flows_bidirectional, pred_flows_bidirectional, local_masks
            )
        else:
            flow_length = ground_truth_flows_bidirectional[0].size(1)
            if flow_length > self.config.subvideo_length:
                pred_flows_f, pred_flows_b, pred_flows_bidirectional_loss, pred_edges_bidirectional_loss = (
                    [],
                    [],
                    [],
                )
                pad_len = 5
                for f in range(0, flow_length, self.config.subvideo_length):
                    start_frame = max(0, f - pad_len)
                    end_frame = min(flow_length, f + self.config.subvideo_length + pad_len)
                    pad_len_s = max(0, f) - start_frame
                    pad_len_e = end_frame - min(flow_length, f + self.config.subvideo_length)
                    pred_flows_bidirectional_sub, pred_edges_bidirectional = (
                        self.flow_completion_net.forward_bidirectional_flow(
                            (
                                ground_truth_flows_bidirectional[0][:, start_frame:end_frame],
                                ground_truth_flows_bidirectional[1][:, start_frame:end_frame],
                            ),
                            flow_masks[:, start_frame : end_frame + 1],
                        )
                    )
                    pred_flows_bidirectional_loss.append(pred_flows_bidirectional_sub)
                    pred_edges_bidirectional_loss.append(pred_edges_bidirectional)
                    pred_flows_bidirectional_sub = self.flow_completion_net.combine_flow(
                        (
                            ground_truth_flows_bidirectional[0][:, start_frame:end_frame],
                            ground_truth_flows_bidirectional[1][:, start_frame:end_frame],
                        ),
                        pred_flows_bidirectional_sub,
                        flow_masks[:, start_frame : end_frame + 1],
                    )

                    pred_flows_f.append(
                        pred_flows_bidirectional_sub[0][:, pad_len_s : end_frame - start_frame - pad_len_e]
                    )
                    pred_flows_b.append(
                        pred_flows_bidirectional_sub[1][:, pad_len_s : end_frame - start_frame - pad_len_e]
                    )

                    torch.cuda.empty_cache()

                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bidirectional = (pred_flows_f, pred_flows_b)

                pred_flows_bidirectional_loss = torch.cat(pred_flows_bidirectional_loss)
                pred_edges_bidirectional_loss = torch.cat(pred_edges_bidirectional_loss)
            else:
                pred_flows_bidirectional, pred_edges_bidirectional = (
                    self.flow_completion_net.forward_bidirectional_flow(ground_truth_flows_bidirectional, flow_masks)
                )
                pred_flows_bidirectional_loss = pred_flows_bidirectional

                pred_flows_bidirectional = self.flow_completion_net.combine_flow(
                    ground_truth_flows_bidirectional, pred_flows_bidirectional, flow_masks
                )

                torch.cuda.empty_cache()

        return pred_flows_bidirectional, pred_flows_bidirectional_loss, pred_edges_bidirectional

    def image_propagation(self, pixel_values_videos, masks_dilated, pred_flows_bidirectional):
        if self.training:
            batch_size, height, width = self.size[0], self.size[3], self.size[4]
            ground_truth_local_frames = pixel_values_videos[:, : self.config.num_local_frames_propainter, ...]
            local_masks = masks_dilated[:, : self.config.num_local_frames_propainter, ...].contiguous()
            masked_frames = pixel_values_videos * (1 - masks_dilated)
            masked_local_frames = masked_frames[:, : self.config.num_local_frames_propainter, ...]

            if self.gradient_checkpointing:
                prop_imgs, updated_local_masks = self._gradient_checkpointing_func(
                    self.inpaint_generator.img_propagation.__call__,
                    masked_local_frames,
                    pred_flows_bidirectional,
                    local_masks,
                    self.config.interp_mode,
                )
            else:
                prop_imgs, updated_local_masks = self.inpaint_generator.img_propagation(
                    masked_local_frames,
                    pred_flows_bidirectional,
                    local_masks,
                    interpolation=self.config.interp_mode,
                )

            updated_masks = masks_dilated.clone()
            updated_masks[:, : self.config.num_local_frames_propainter, ...] = updated_local_masks.view(
                batch_size,
                self.config.num_local_frames_propainter,
                1,
                height,
                width,
            )
            updated_frames = masked_frames.clone()
            prop_local_frames = (
                ground_truth_local_frames * (1 - local_masks)
                + prop_imgs.view(
                    batch_size,
                    self.config.num_local_frames_propainter,
                    3,
                    height,
                    width,
                )
                * local_masks
            )  # merge
            updated_frames[:, : self.config.num_local_frames_propainter, ...] = prop_local_frames

        else:
            height, width = self.size[3], self.size[4]
            masked_frames = pixel_values_videos * (1 - masks_dilated)
            subvideo_length_img_prop = min(
                100, self.config.subvideo_length
            )  # ensure a minimum of 100 frames for image propagation
            if self.video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, self.video_length, subvideo_length_img_prop):
                    start_frame = max(0, f - pad_len)
                    end_frame = min(self.video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - start_frame
                    pad_len_e = end_frame - min(self.video_length, f + subvideo_length_img_prop)

                    batch_size, timesteps, _, _, _ = masks_dilated[:, start_frame:end_frame].size()
                    pred_flows_bidirectional_sub = (
                        pred_flows_bidirectional[0][:, start_frame : end_frame - 1],
                        pred_flows_bidirectional[1][:, start_frame : end_frame - 1],
                    )
                    prop_imgs_sub, updated_local_masks_sub = self.inpaint_generator.img_propagation(
                        masked_frames[:, start_frame:end_frame],
                        pred_flows_bidirectional_sub,
                        masks_dilated[:, start_frame:end_frame],
                        "nearest",
                    )
                    updated_frames_sub = (
                        pixel_values_videos[:, start_frame:end_frame] * (1 - masks_dilated[:, start_frame:end_frame])
                        + prop_imgs_sub.view(batch_size, timesteps, 3, height, width)
                        * masks_dilated[:, start_frame:end_frame]
                    )
                    updated_masks_sub = updated_local_masks_sub.view(batch_size, timesteps, 1, height, width)

                    updated_frames.append(updated_frames_sub[:, pad_len_s : end_frame - start_frame - pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s : end_frame - start_frame - pad_len_e])
                    torch.cuda.empty_cache()

                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                batch_size, timesteps, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = self.inpaint_generator.img_propagation(
                    masked_frames, pred_flows_bidirectional, masks_dilated, "nearest"
                )
                updated_frames = (
                    pixel_values_videos * (1 - masks_dilated)
                    + prop_imgs.view(batch_size, timesteps, 3, height, width) * masks_dilated
                )
                updated_masks = updated_local_masks.view(batch_size, timesteps, 1, height, width)
                torch.cuda.empty_cache()

        return updated_frames, updated_masks

    def feature_propagation(
        self,
        pixel_values_videos,
        updated_frames,
        updated_masks,
        masks_dilated,
        pred_flows_bidirectional,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if self.training:
            batch_size, _, num_channels, height, width = self.size
            # ---- feature propagation + Transformer ----
            if self.gradient_checkpointing:
                inpaint_generator_outputs = self._gradient_checkpointing_func(
                    self.inpaint_generator.__call__,
                    updated_frames,
                    pred_flows_bidirectional,
                    masks_dilated,
                    updated_masks,
                    self.config.num_local_frames_propainter,
                    output_attentions,
                    output_hidden_states,
                    return_dict,
                )
            else:
                inpaint_generator_outputs = self.inpaint_generator(
                    updated_frames,
                    pred_flows_bidirectional,
                    masks_dilated,
                    updated_masks,
                    self.config.num_local_frames_propainter,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            pred_imgs = (
                inpaint_generator_outputs[0] if not return_dict else inpaint_generator_outputs.last_hidden_state
            )
            pred_imgs = pred_imgs.view(batch_size, -1, num_channels, height, width)

            all_hidden_states = (
                inpaint_generator_outputs[1:2] if not return_dict else inpaint_generator_outputs.hidden_states
            )
            all_self_attentions = (
                inpaint_generator_outputs[2:] if not return_dict else inpaint_generator_outputs.attentions
            )

            pred_imgs_loss = pred_imgs
            # get the local frames
            comp_frames = pixel_values_videos * (1.0 - masks_dilated) + pred_imgs * masks_dilated
            comp_frames_loss = comp_frames

        else:
            height, width = self.size[3], self.size[4]
            comp_frames = [[None] * self.video_length for _ in range(self.size[0])]
            pred_imgs_loss = [[None] * self.video_length for _ in range(self.size[0])]

            neighbor_stride = self.config.neighbor_length // 2
            if self.video_length > self.config.subvideo_length:
                ref_num = self.config.subvideo_length // self.config.ref_stride
            else:
                ref_num = -1

            # ---- feature propagation + transformer ----
            batch_idxs = range(self.size[0])
            for f in range(0, self.video_length, neighbor_stride):
                neighbor_ids = list(
                    range(
                        max(0, f - neighbor_stride),
                        min(self.video_length, f + neighbor_stride + 1),
                    )
                )
                ref_ids = self._get_ref_index(f, neighbor_ids, self.video_length, self.config.ref_stride, ref_num)
                selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
                selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
                selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
                selected_pred_flows_bidirectional = (
                    pred_flows_bidirectional[0][:, neighbor_ids[:-1], :, :, :],
                    pred_flows_bidirectional[1][:, neighbor_ids[:-1], :, :, :],
                )

                # 1.0 indicates mask
                num_neighbor_frames = len(neighbor_ids)

                # pred_img = selected_imgs # results of image propagation
                inpaint_generator_outputs = self.inpaint_generator(
                    selected_imgs,
                    selected_pred_flows_bidirectional,
                    selected_masks,
                    selected_update_masks,
                    num_neighbor_frames,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                pred_img = (
                    inpaint_generator_outputs[0] if not return_dict else inpaint_generator_outputs.last_hidden_state
                )

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 1, 3, 4, 2).detach().numpy() * 255

                binary_masks = (
                    masks_dilated[:, neighbor_ids, :, :, :].cpu().permute(0, 1, 3, 4, 2).numpy().astype(np.uint8)
                )

                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = [
                        np.array(pred_img[batch_idx][i]).astype(np.uint8) * binary_masks[batch_idx][i]
                        + self.original_frames[batch_idx][idx] * (1 - binary_masks[batch_idx][i])
                        for batch_idx in batch_idxs
                    ]

                    for batch_idx in batch_idxs:
                        if comp_frames[batch_idx][idx] is None:
                            comp_frames[batch_idx][idx] = img[batch_idx]
                        else:
                            comp_frames[batch_idx][idx] = (
                                comp_frames[batch_idx][idx].astype(np.float32) * 0.5
                                + img[batch_idx].astype(np.float32) * 0.5
                            )
                        comp_frames[batch_idx][idx] = comp_frames[batch_idx][idx].astype(np.uint8)

                        pred_imgs_loss[batch_idx][idx] = pred_img[batch_idx][i]

            if output_hidden_states:
                all_hidden_states = (
                    inpaint_generator_outputs[1:2] if not return_dict else inpaint_generator_outputs.hidden_states
                )
            if output_attentions:
                all_self_attentions = (
                    inpaint_generator_outputs[2:] if not return_dict else inpaint_generator_outputs.attentions
                )

            device = pixel_values_videos.device

            comp_frames_loss = torch.stack(
                [
                    torch.stack([torch.tensor(frame).permute(2, 0, 1) for frame in comp_frames[batch_idx]])
                    for batch_idx in batch_idxs
                ]
            )
            comp_frames_loss = comp_frames_loss.to(device).to(torch.float32)

            pred_imgs_loss = torch.stack(
                [
                    torch.stack([torch.tensor(frame).permute(2, 0, 1) for frame in pred_imgs_loss[batch_idx]])
                    for batch_idx in batch_idxs
                ]
            )
            pred_imgs_loss = pred_imgs_loss.to(device).to(torch.float32)

        return comp_frames, pred_imgs_loss, comp_frames_loss, all_hidden_states, all_self_attentions

    @add_start_docstrings_to_model_forward(PROPAINTER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedImageModelingOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values_videos: Optional[torch.Tensor] = None,
        flow_masks: Optional[torch.BoolTensor] = None,
        masks_dilated: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedImageModelingOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import av
        >>> import cv2
        >>> import imageio
        >>> import numpy as np
        >>> import os
        >>> import torch

        >>> from datasets import load_dataset
        >>> from huggingface_hub import hf_hub_download
        >>> from PIL import Image
        >>> from transformers import ProPainterVideoProcessor, ProPainterModel

        >>> np.random.seed(0)

        >>> def read_video_pyav(container, indices):
        ...    '''
        ...    Decode the video with PyAV decoder.
        ...    Args:
        ...        container (`av.container.input.InputContainer`): PyAV container.
        ...        indices (`List[int]`): List of frame indices to decode.
        ...    Returns:
        ...        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...    '''
        ...    frames = []
        ...    container.seek(0)
        ...    start_index = indices[0]
        ...    end_index = indices[-1]
        ...    for i, frame in enumerate(container.decode(video=0)):
        ...        if i > end_index:
        ...            break
        ...        if i >= start_index and i in indices:
        ...            frames.append(frame)
        ...    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...    '''
        ...    Sample a given number of frame indices from the video.
        ...    Args:
        ...        clip_len (`int`): Total number of frames to sample.
        ...        frame_sample_rate (`int`): Sample every n-th frame.
        ...        seg_len (`int`): Maximum allowed index of sample's last frame.
        ...    Returns:
        ...        indices (`List[int]`): List of sampled frame indices
        ...    '''
        ...    converted_len = int(clip_len * frame_sample_rate)
        ...    end_idx = np.random.randint(converted_len, seg_len)
        ...    start_idx = end_idx - converted_len
        ...    indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...    return indices


        >>> # Using .mp4 files for data:

        >>> # video clip consists of 80 frames(both masks and original video) (3 seconds at 24 FPS)
        >>> video_file_path = hf_hub_download(
        ...    repo_id="ruffy369/propainter-object-removal", filename="object_removal_bmx/bmx.mp4", repo_type="dataset"
        ... )
        >>> masks_file_path = hf_hub_download(
        ...    repo_id="ruffy369/propainter-object-removal", filename="object_removal_bmx/bmx_masks.mp4", repo_type="dataset"
        ... )
        >>> container_video = av.open(video_file_path)
        >>> container_masks = av.open(masks_file_path)

        >>> # sample 32 frames
        >>> indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container_video.streams.video[0].frames)
        >>> video = read_video_pyav(container=container_video, indices=indices)

        >>> masks = read_video_pyav(container=container_masks, indices=indices)
        >>> video = list(video)
        >>> masks = list(masks)

        >>> # Forward pass:

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> video_processor = ProPainterVideoProcessor()
        >>> inputs = video_processor(video, masks = masks, return_tensors="pt").to(device)

        >>> model = ProPainterModel.from_pretrained("ruffy369/ProPainter").to(device)

        >>> # The first input in this always has a value for inference as its not utilised during training
        >>> with torch.no_grad():
        ...    outputs = model(**inputs)

        >>> # To visualize the reconstructed frames with object removal video inpainting:
        >>> reconstructed_frames = outputs["reconstruction"][0] # As there is only a single video in batch for inferece
        >>> reconstructed_frames = [cv2.resize(frame, (240,432)) for frame in reconstructed_frames]
        >>> imageio.mimwrite(os.path.join(<PATH_TO_THE_FOLDER>, 'inpaint_out.mp4'), reconstructed_frames, fps=24, quality=7)

        >>> # Using .jpg files for data:

        >>> ds = load_dataset("ruffy369/propainter-object-removal")
        >>> ds_images = ds['train']["image"]
        >>> num_frames = 80
        >>> video = [np.array(ds_images[i]) for i in range(num_frames)]
        >>> #stack to convert H,W mask frame to compatible H,W,C frame as they are already in grayscale
        >>> masks = [np.stack([np.array(ds_images[i])], axis=-1) for i in range(num_frames, 2*num_frames)]

        >>> # Forward pass:

        >>> inputs = video_processor(video, masks = masks, return_tensors="pt").to(device)

        >>> # The first input in this always has a value for inference as its not utilised during training
        >>> with torch.no_grad():
        ...    outputs = model(**inputs)

        >>> # To visualize the reconstructed frames with object removal video inpainting:
        >>> reconstructed_frames = outputs["reconstruction"][0] # As there is only a single video in batch for inferece
        >>> reconstructed_frames = [cv2.resize(frame, (240,432)) for frame in reconstructed_frames]
        >>> imageio.mimwrite(os.path.join(<PATH_TO_THE_FOLDER>, 'inpaint_out.mp4'), reconstructed_frames, fps=24, quality=7)

        >>> # Performing video outpainting:

        >>> # Forward pass:

        >>> inputs = video_processor(video, masks = masks, video_painting_mode = "video_outpainting", scale_size = (1.0,1.2), return_tensors="pt").to(device)

        >>> # The first input in this always has a value for inference as its not utilised during training
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # To visualize the reconstructed frames with object removal video inpainting:
        >>> reconstructed_frames = outputs["reconstruction"][0] # As there is only a single video in batch for inferece
        >>> reconstructed_frames = [cv2.resize(frame, (240,512)) for frame in reconstructed_frames]
        >>> imageio.mimwrite(os.path.join(<PATH_TO_THE_FOLDER>, 'outpaint_out.mp4'), reconstructed_frames, fps=24, quality=7)
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if pixel_values_videos is None:
            raise ValueError("You have to specify pixel_values_videos")

        if not self.training:
            # original_frames are used for inference part only
            self.original_frames = pixel_values_videos
            self.original_frames = (self.original_frames * 255.0).to(torch.uint8).cpu().numpy()
            self.original_frames = [[frame.transpose(1, 2, 0) for frame in video] for video in self.original_frames]

            pixel_values_videos = pixel_values_videos * 2 - 1

        losses = ProPainterLosses(self.config, self.training)

        self.size = pixel_values_videos.size()
        self.video_length = pixel_values_videos.size(1)

        ground_truth_flows_bidirectional = self.compute_flow(pixel_values_videos)

        pred_flows_bidirectional, pred_flows_bidirectional_loss, pred_edges_bidirectional = self.complete_flow(
            ground_truth_flows_bidirectional, flow_masks
        )

        updated_frames, updated_masks = self.image_propagation(
            pixel_values_videos, masks_dilated, pred_flows_bidirectional
        )

        comp_frames, pred_imgs_loss, comp_frames_loss, all_hidden_states, all_self_attentions = (
            self.feature_propagation(
                pixel_values_videos,
                updated_frames,
                updated_masks,
                masks_dilated,
                pred_flows_bidirectional,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        )

        gen_loss, dis_loss, flow_complete_loss = losses.calculate_losses(
            pred_imgs_loss,
            masks_dilated,
            pixel_values_videos,
            comp_frames_loss,
            self.discriminator,
            pred_flows_bidirectional_loss,
            ground_truth_flows_bidirectional,
            flow_masks,
            pred_edges_bidirectional,
        )

        if not return_dict:
            return tuple(
                v
                for v in [
                    (gen_loss, dis_loss, flow_complete_loss),
                    comp_frames,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return MaskedImageModelingOutput(
            loss=(gen_loss, dis_loss, flow_complete_loss),
            reconstruction=comp_frames,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

# coding=utf-8
# Copyright 2024 The Apple Research Team Authors and The HuggingFace Team. All rights reserved.
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
"""PyTorch DepthPro model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)
from ..auto import AutoModel
from .configuration_depth_pro import DepthProConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "DepthProConfig"


DEPTH_PRO_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DepthProConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DEPTH_PRO_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`DPTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

DEPTH_PRO_FOR_DEPTH_ESTIMATION_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DepthProConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        use_fov_model (`bool`, *optional*, defaults to `True`):
            Whether to use `DepthProFOVModel` to generate the field of view.
"""


@dataclass
class DepthProOutput(ModelOutput):
    """
    Base class for DepthPro's outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, n_patches_per_batch, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        features (`List[torch.FloatTensor]`, *optional*:
            Features from scaled images and hidden_states.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, n_patches_per_batch, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer and the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, n_patches_per_batch, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    features: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class DepthProDepthEstimatorOutput(ModelOutput):
    """
    Base class for DepthProForDepthEstimation's output.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        predicted_depth (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            Predicted depth for each pixel.
        fov (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned when `use_fov_model` is provided):
            Field of View Scaler.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, n_patches_per_batch, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer and the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, n_patches_per_batch, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    predicted_depth: torch.FloatTensor = None
    fov: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


def patch_to_batch(data: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Converts tensor from shape:
    (num_patches, seq_len, hidden_size) -> (batch_size, n_patches_per_batch, seq_len, hidden_size)
    """
    data = data.reshape(-1, batch_size, *data.shape[1:])
    data = data.transpose(0, 1)
    return data


def batch_to_patch(data: torch.Tensor) -> torch.Tensor:
    """
    Converts tensor from shape:
    (batch_size, n_patches_per_batch, seq_len, hidden_size) -> (num_patches, seq_len, hidden_size)
    """
    data = data.transpose(0, 1)
    data = data.reshape(-1, *data.shape[2:])
    return data


class DepthProFeatureUpsample(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config

        self.upsample_blocks = nn.ModuleList()

        # for image_features
        self.upsample_blocks.append(
            self._create_upsample_block(
                input_dims=config.hidden_size,
                intermediate_dims=config.hidden_size,
                output_dims=config.scaled_images_feature_dims[0],
                n_upsample_layers=1,
                use_proj=False,
                bias=True,
            )
        )

        # for scaled_images_features
        for i, feature_dims in enumerate(config.scaled_images_feature_dims):
            upsample_block = self._create_upsample_block(
                input_dims=config.hidden_size,
                intermediate_dims=feature_dims,
                output_dims=feature_dims,
                n_upsample_layers=1,
            )
            self.upsample_blocks.append(upsample_block)

        # for intermediate_features
        for i, feature_dims in enumerate(config.intermediate_feature_dims):
            intermediate_dims = config.fusion_hidden_size if i == 0 else feature_dims
            upsample_block = self._create_upsample_block(
                input_dims=config.hidden_size,
                intermediate_dims=intermediate_dims,
                output_dims=feature_dims,
                n_upsample_layers=2 + i,
            )
            self.upsample_blocks.append(upsample_block)

    def _create_upsample_block(
        self,
        input_dims: int,
        intermediate_dims: int,
        output_dims: int,
        n_upsample_layers: int,
        use_proj: bool = True,
        bias: bool = False,
    ) -> nn.Module:
        upsample_block = nn.Sequential()

        # create first projection layer
        if use_proj:
            proj = nn.Conv2d(
                in_channels=input_dims,
                out_channels=intermediate_dims,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )
            upsample_block.append(proj)

        # create following upsample layers
        for i in range(n_upsample_layers):
            in_channels = intermediate_dims if i == 0 else output_dims
            layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=output_dims,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            )
            upsample_block.append(layer)

        return upsample_block

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        upsampled_features = []
        for i, upsample_block in enumerate(self.upsample_blocks):
            upsampled_feature = upsample_block(features[i])
            upsampled_features.append(upsampled_feature)
        return upsampled_features


class DepthProFeatureProjection(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config

        combined_feature_dims = config.scaled_images_feature_dims + config.intermediate_feature_dims
        self.projections = nn.ModuleList()
        for i, in_channels in enumerate(combined_feature_dims):
            if i == len(combined_feature_dims) - 1 and in_channels == config.fusion_hidden_size:
                # projection for last layer can be ignored if input and output channels already match
                self.projections.append(nn.Identity())
            else:
                self.projections.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=config.fusion_hidden_size,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    )
                )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        projected_features = []
        for i, projection in enumerate(self.projections):
            upsampled_feature = projection(features[i])
            projected_features.append(upsampled_feature)
        return projected_features


def interpolate(
    pixel_values: torch.Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    if mode == "nearest":
        align_corners = None
    return nn.functional.interpolate(
        pixel_values,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


def patch(pixel_values: torch.Tensor, patch_size: int, overlap_ratio: float) -> torch.Tensor:
    """Creates Patches from Batch."""
    batch_size, num_channels, height, width = pixel_values.shape

    if height == width == patch_size:
        # create patches only if scaled image is not already equal to patch size
        return pixel_values

    stride = int(patch_size * (1 - overlap_ratio))

    # (batch_size, num_channels, height, width)
    patches = torch.nn.functional.unfold(pixel_values, kernel_size=(patch_size, patch_size), stride=(stride, stride))
    # patches.shape (batch_size, patch_size**2 * num_channels, n_patches_per_batch)
    patches = patches.permute(2, 0, 1)
    # patches.shape (n_patches_per_batch, batch_size, patch_size**2 * C)
    patches = patches.reshape(-1, num_channels, patch_size, patch_size)
    # patches.shape (n_patches, num_channels, patch_size, patch_size)

    return patches


def reshape_feature(hidden_states: torch.Tensor) -> torch.Tensor:
    """Discard class token and reshape 1D feature map to a 2D grid."""
    n_samples, seq_len, hidden_size = hidden_states.shape
    size = int(math.sqrt(seq_len))

    # (n_samples, seq_len, hidden_size)
    hidden_states = hidden_states[:, -(size**2) :, :]  # remove mask tokens if there are any
    # (n_samples, seq_len, hidden_size)
    hidden_states = hidden_states.reshape(n_samples, size, size, hidden_size)
    # (n_samples, size, size, hidden_size)
    hidden_states = hidden_states.permute(0, 3, 1, 2)
    # (n_samples, hidden_size, size, size)
    return hidden_states


def merge(patches: torch.Tensor, batch_size: int, padding: int) -> torch.Tensor:
    n_patches, hidden_size, out_size, out_size = patches.shape
    n_patches_per_batch = n_patches // batch_size
    sqrt_n_patches_per_batch = int(math.sqrt(n_patches_per_batch))
    new_out_size = sqrt_n_patches_per_batch * out_size

    if n_patches == batch_size:
        # merge only if the patches were created from scaled image
        # patches are not created when scaled image size is equal to patch size
        return patches

    if n_patches_per_batch < 4:
        # for each batch, atleast 4 small patches are required to
        # recreate a large square patch from merging them and later padding is applied
        # 3 x (8x8) patches becomes 1 x ( 8x8 ) patch (extra patch ignored, no padding)
        # 4 x (8x8) patches becomes 1 x (16x16) patch (padding later)
        # 5 x (8x8) patches becomes 1 x (16x16) patch (extra patch ignored, padding later)
        # 9 x (8x8) patches becomes 1 x (24x24) patch (padding later)
        # thus the following code only rearranges the patches and removes extra ones
        padding = 0

    # make sure padding is not large enough to remove more than half of the patch
    padding = min(out_size // 4, padding)

    # patches.shape: (n_patches, hidden_size, out_size, out_size)

    merged = patches.reshape(n_patches_per_batch, batch_size, hidden_size, out_size, out_size)
    # (n_patches_per_batch, batch_size, hidden_size, out_size, out_size)
    merged = merged.permute(1, 2, 0, 3, 4)
    # (batch_size, hidden_size, n_patches_per_batch, out_size, out_size)
    merged = merged[:, :, : sqrt_n_patches_per_batch**2, :, :]
    # (batch_size, hidden_size, n_patches_per_batch, out_size, out_size)
    merged = merged.reshape(
        batch_size, hidden_size, sqrt_n_patches_per_batch, sqrt_n_patches_per_batch, out_size, out_size
    )
    # (batch_size, hidden_size, sqrt_n_patches_per_batch, sqrt_n_patches_per_batch, out_size, out_size)
    merged = merged.permute(0, 1, 2, 4, 3, 5)
    # (batch_size, hidden_size, sqrt_n_patches_per_batch, out_size, sqrt_n_patches_per_batch, out_size)
    merged = merged.reshape(batch_size, hidden_size, new_out_size, new_out_size)
    # (batch_size, hidden_size, sqrt_n_patches_per_batch * out_size, sqrt_n_patches_per_batch * out_size)

    # merged.shape: (batch_size, hidden_size, new_out_size, new_out_size)

    if padding > 0:
        # let out_size = 8, new_out_size = 32, padding = 2
        # each patch is separated by |
        # and padding is applied to the merging edges of each patch
        # 00 01 02 03 04 05 06 07 | 08 09 10 11 12 13 14 15 | 16 17 18 19 20 21 22 23 | 24 25 26 27 28 29 30 31
        # 00 01 02 03 04 05 -- -- | -- -- 10 11 12 13 -- -- | -- -- 18 19 20 21 -- -- | -- -- 26 27 28 29 30 31
        # starting_indexes = [2, 10, 18, 26]
        # valid_indexes = [ 0,  1,  2,  3,  4,  5, 10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29, 30, 31])

        starting_indexes = torch.arange(start=padding, end=new_out_size, step=out_size)
        valid_indexes = torch.concat(
            [
                torch.arange(padding),
                *[torch.arange(index, index + out_size - padding * 2) for index in starting_indexes],
                torch.arange(new_out_size - padding, new_out_size),
            ]
        )
        merged = merged[:, :, valid_indexes]
        merged = merged[:, :, :, valid_indexes]

    return merged


class DepthProEncoder(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.fusion_hidden_size = config.fusion_hidden_size

        self.intermediate_hook_ids = config.intermediate_hook_ids
        self.intermediate_feature_dims = config.intermediate_feature_dims
        self.scaled_images_ratios = config.scaled_images_ratios
        self.scaled_images_overlap_ratios = config.scaled_images_overlap_ratios
        self.scaled_images_feature_dims = config.scaled_images_feature_dims
        self.merge_padding_value = config.merge_padding_value

        # placeholder to avoid
        # ValueError: The following configuration classes contain unused attributes in the corresponding modeling files
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads

        self.n_scaled_images = len(self.scaled_images_ratios)
        self.n_intermediate_hooks = len(self.intermediate_hook_ids)

        # patch encoder
        self.patch_encoder = AutoModel.from_config(config.backbone_config, **self.config.backbone_kwargs)

        # image encoder
        self.image_encoder = AutoModel.from_config(config.backbone_config, **self.config.backbone_kwargs)

        # upsample features
        self.feature_upsample = DepthProFeatureUpsample(config)

        # for STEP 7: fuse low_res and image features
        self.fuse_image_with_low_res = nn.Conv2d(
            in_channels=config.scaled_images_feature_dims[0] * 2,
            out_channels=config.scaled_images_feature_dims[0],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # project features
        self.feature_projection = DepthProFeatureProjection(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, DepthProOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values.dim() != 4:
            raise ValueError("Input tensor must have shape (batch_size, num_channels, height, width).")

        batch_size, num_channels, height, width = pixel_values.shape

        if not (num_channels == self.config.num_channels):
            raise ValueError(
                f"Found {num_channels} channels in image, expected number of channels is {self.config.num_channels} from config."
            )

        if min(self.scaled_images_ratios) * min(height, width) < self.config.patch_size:
            raise ValueError(
                f"Image size {height}x{width} is too small to be scaled "
                f"with scaled_images_ratios={self.scaled_images_ratios} "
                f"when patch_size={self.config.patch_size}."
            )

        # pixel_values.shape (batch_size, num_channels, height, width)

        # STEP 1: create 3-level image

        scaled_images = []
        for ratio in self.scaled_images_ratios:
            scaled_images.append(interpolate(pixel_values, scale_factor=ratio))
            # (batch_size, num_channels, height*ratio, width*ratio)

        # STEP 2: create patches

        for i in range(self.n_scaled_images):
            scaled_images[i] = patch(
                scaled_images[i],
                patch_size=self.config.patch_size,
                overlap_ratio=self.scaled_images_overlap_ratios[i],
            )
            # (n_patches_per_scaled_image[i], num_channels, patch_size, patch_size)
        n_patches_per_scaled_image = [len(i) for i in scaled_images]
        patches = torch.cat(scaled_images[::-1], dim=0)  # -1 as patch encoder expects high res patches first
        # (n_patches, num_channels, patch_size, patch_size)

        # STEP 3: apply patch and image encoder

        patch_encodings = self.patch_encoder(
            patches,
            head_mask=head_mask,
            output_attentions=output_attentions,
            # required for intermediate features
            output_hidden_states=self.n_intermediate_hooks or output_hidden_states,
            return_dict=True,
        )
        # patch_encodings.last_hidden_state (batch_size, n_patches/batch_size, seq_len, hidden_size)
        # patch_encodings.hidden_states[i]  (batch_size, n_patches/batch_size, seq_len, hidden_size)
        # patch_encodings.attentions[i]     (batch_size, n_patches/batch_size, num_heads, seq_len, seq_len)

        last_hidden_state = patch_encodings.last_hidden_state
        # (n_patches, seq_len, hidden_size)
        scaled_images_last_hidden_state = torch.split_with_sizes(last_hidden_state, n_patches_per_scaled_image[::-1])
        # (n_patches_per_scaled_image[i], seq_len, hidden_size)
        scaled_images_last_hidden_state = scaled_images_last_hidden_state[::-1]
        # (n_patches_per_scaled_image[i], seq_len, hidden_size)
        # -1 (reverse list) as patch encoder returns high res patches first, we need low res first

        # scale the image to patch size for image_encoder
        image_scaled_to_patch_size = interpolate(
            pixel_values,
            size=(self.config.patch_size, self.config.patch_size),
        )
        image_encodings = self.image_encoder(
            pixel_values=image_scaled_to_patch_size,
            head_mask=head_mask,
            # TODO: return hidden_states from patch_encodings instead of image_encodings
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # image_encodings.last_hidden_state (batch_size, seq_len, hidden_size)
        # image_encodings.hidden_states[i]  (batch_size, seq_len, hidden_size)
        # image_encodings.attentions[i]     (batch_size, num_heads, seq_len, seq_len)

        # calculate base height and width
        # base height and width are the dimensions of the lowest resolution features
        out_size = int(math.sqrt(image_encodings.last_hidden_state.shape[1]))
        exponent_value = int(math.log2(width / out_size))
        base_height = height // 2**exponent_value
        base_width = width // 2**exponent_value

        # STEP 4: get patch features (high_res, med_res, low_res) - (3-5) in diagram

        scaled_images_features = []
        for i in range(self.n_scaled_images):
            # a. extract hidden_state
            hidden_state = scaled_images_last_hidden_state[i]
            # (n_patches_per_scaled_image[i], seq_len, hidden_size)

            # b. reshape back to image like
            features = reshape_feature(hidden_state)
            # (n_patches_per_scaled_image[i], hidden_size, out_size, out_size)

            # c. merge patches back together
            features = merge(
                features,
                batch_size=batch_size,
                padding=torch_int(self.merge_padding_value * (1 / self.scaled_images_ratios[i])),
            )  # (batch_size, hidden_size, merge_out_size, merge_out_size)

            # d. interpolate patches to base size
            features = interpolate(
                features, size=(base_height * 2**i, base_width * 2**i), mode="nearest"
            )  # (batch_size, hidden_size, base_height*2**i, base_width*2**i)

            scaled_images_features.append(features)

        # STEP 5: get intermediate features - (1-2) in diagram

        intermediate_features = []
        for i in range(self.n_intermediate_hooks):
            # a. extract hidden_state
            layer_id = (
                self.intermediate_hook_ids[i] + 1
            )  # +1 to correct index position as hidden_states contain embedding output as well
            hidden_state = patch_encodings.hidden_states[layer_id]
            hidden_state = hidden_state[
                : n_patches_per_scaled_image[-1]
            ]  # number of patches to be of same length as highest resolution
            # (n_patches_per_scaled_image[-1], seq_len, hidden_size)

            # b. reshape back to image like
            features = reshape_feature(hidden_state)
            # (n_patches_per_scaled_image[-1], hidden_size, out_size, out_size)

            # c. merge patches back together
            features = merge(
                features,
                batch_size=batch_size,
                padding=torch_int(self.merge_padding_value * (1 / self.scaled_images_ratios[-1])),
            )  # (batch_size, hidden_size, merge_out_size, merge_out_size)

            # d. interpolate patches to base size
            features = interpolate(
                features,
                size=(base_height * 2 ** (self.n_scaled_images - 1), base_width * 2 ** (self.n_scaled_images - 1)),
                mode="nearest",
            )  # (batch_size, hidden_size, base_height*2**(n_scaled_images - 1), base_width*2**(n_scaled_images - 1))

            intermediate_features.append(features)

        # STEP 6: get image features - (6) in diagram

        # a. extract hidden_state
        hidden_state = image_encodings.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # b. reshape back to image like
        image_features = reshape_feature(hidden_state)
        # (batch_size, hidden_size, out_size, out_size)

        # c. merge patches back together
        # no merge required for image_features as they are already in batches instead of patches

        # d. interpolate patches to base size
        image_features = interpolate(image_features, size=(base_height, base_width), mode="nearest")
        # (batch_size, hidden_size, base_height, base_width)

        # STEP 7: combine all features
        features = [
            image_features,
            # (batch_size, scaled_images_feature_dims[0], base_height*2, base_width*2)
            *scaled_images_features,
            # (batch_size, scaled_images_feature_dims[i], base_height*2**(i+1), base_width*2**(i+1))
            *intermediate_features,
            # (batch_size,  intermediate_feature_dims[i], base_height*2**(n_scaled_images+i+1), base_width*2**(n_scaled_images+i+1))
        ]

        # STEP 8: upsample features
        features = self.feature_upsample(features)

        # STEP 9: apply fusion
        # (global features = low res features + image features)
        # fuses image_features with lowest resolution features as they are of same size
        global_features = torch.cat((features[1], features[0]), dim=1)
        global_features = self.fuse_image_with_low_res(global_features)
        features = [global_features, *features[2:]]

        # STEP 10: project features
        features = self.feature_projection(features)

        # STEP 11: return output

        # TODO: return hidden_states from patch_encodings instead of image_encodings
        # last_hidden_state = patch_encodings.last_hidden_state
        # hidden_states = patch_encodings.hidden_states if output_hidden_states else None
        # attentions = patch_encodings.attentions if output_attentions else None
        last_hidden_state = image_encodings.last_hidden_state
        hidden_states = image_encodings.hidden_states
        attentions = image_encodings.attentions

        if not return_dict:
            return tuple(v for v in [last_hidden_state, features, hidden_states, attentions] if v is not None)

        return DepthProOutput(
            last_hidden_state=last_hidden_state,
            features=features,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class DepthProPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DepthProConfig
    base_model_prefix = "depth_pro"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@add_start_docstrings(
    "The bare DepthPro Model transformer outputting raw hidden-states without any specific head on top.",
    DEPTH_PRO_START_DOCSTRING,
)
class DepthProModel(DepthProPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = DepthProEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # TODO: return hidden_states from patch_encodings instead of image_encodings
        # return self.encoder.patch_encoder.embeddings.patch_embeddings
        return self.encoder.image_encoder.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.patch_encoder.encoder.layer[layer].attention.prune_heads(heads)
            self.encoder.image_encoder.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(DEPTH_PRO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DepthProOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, DepthProModel

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> checkpoint = "geetu040/DepthPro"
        >>> processor = AutoProcessor.from_pretrained(checkpoint)
        >>> model = DepthProModel.from_pretrained(checkpoint)

        >>> # prepare image for the model
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     output = model(**inputs)

        >>> output.last_hidden_state.shape
        torch.Size([1, 35, 577, 1024])
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encodings = self.encoder(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encodings


# Copied from transformers.models.dpt.modeling_dpt.DPTPreActResidualLayer DPT->DepthPro
class DepthProPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DepthProConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.use_batch_norm = config.use_batch_norm_in_fusion_residual
        use_bias_in_fusion_residual = (
            config.use_bias_in_fusion_residual
            if config.use_bias_in_fusion_residual is not None
            else not self.use_batch_norm
        )

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(config.fusion_hidden_size)
            self.batch_norm2 = nn.BatchNorm2d(config.fusion_hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)

        hidden_state = self.convolution1(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        return hidden_state + residual


# Modified from transformers.models.dpt.modeling_dpt.DPTFeatureFusionLayer
# except it uses deconv and skip_add and needs no interpolation
class DepthProFeatureFusionLayer(nn.Module):
    def __init__(self, config: DepthProConfig, use_deconv: bool = True):
        super().__init__()
        self.config = config
        self.use_deconv = use_deconv

        self.residual_layer1 = DepthProPreActResidualLayer(config)
        self.residual_layer2 = DepthProPreActResidualLayer(config)

        if self.use_deconv:
            self.deconv = nn.ConvTranspose2d(
                in_channels=config.fusion_hidden_size,
                out_channels=config.fusion_hidden_size,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )

        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, hidden_state: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        if residual is not None:
            hidden_state = self.skip_add.add(hidden_state, self.residual_layer1(residual))

        hidden_state = self.residual_layer2(hidden_state)
        if self.use_deconv:
            hidden_state = self.deconv(hidden_state)
        hidden_state = self.projection(hidden_state)

        return hidden_state


# Modified from transformers.models.dpt.modeling_dpt.DPTFeatureFusionStage with DPT->DepthPro
# with deconv and reversed layers
class DepthProFeatureFusionStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_layers = len(config.intermediate_hook_ids) + len(config.scaled_images_ratios)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.layers.append(DepthProFeatureFusionLayer(config))
        # final layer doesnot require deconvolution
        self.layers.append(DepthProFeatureFusionLayer(config, use_deconv=False))

    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.num_layers != len(hidden_states):
            raise ValueError(
                f"num_layers={self.num_layers} in DepthProFeatureFusionStage"
                f"doesnot match len(hidden_states)={len(hidden_states)}"
            )

        fused_hidden_states = []
        fused_hidden_state = None
        for hidden_state, layer in zip(hidden_states, self.layers):
            if fused_hidden_state is None:
                # first layer only uses the last hidden_state
                fused_hidden_state = layer(hidden_state)
            else:
                fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class DepthProFOVModel(nn.Module):
    def __init__(self, config: DepthProConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.fusion_hidden_size = config.fusion_hidden_size

        self.out_size = config.backbone_config.image_size // config.backbone_config.patch_size

        self.encoder = AutoModel.from_config(config.backbone_config, **self.config.backbone_kwargs)
        self.encoder_neck = nn.Linear(self.hidden_size, self.fusion_hidden_size // 2)
        self.global_neck = nn.Sequential(
            nn.Conv2d(self.fusion_hidden_size, self.fusion_hidden_size // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )

        # create initial head layers
        self.head = nn.Sequential()
        for i in range(config.num_fov_head_layers):
            self.head.append(
                nn.Conv2d(
                    math.ceil(self.fusion_hidden_size / 2 ** (i + 1)),
                    math.ceil(self.fusion_hidden_size / 2 ** (i + 2)),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            self.head.append(nn.ReLU(True))
        # calculate expected shapes to finally generate a scalar output from final head layer
        final_in_channels = math.ceil(self.fusion_hidden_size / 2 ** (config.num_fov_head_layers + 1))
        final_kernal_size = int((self.out_size - 1) / 2**config.num_fov_head_layers + 1)
        self.head.append(
            nn.Conv2d(
                in_channels=final_in_channels, out_channels=1, kernel_size=final_kernal_size, stride=1, padding=0
            )
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        global_features: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape

        # follow the steps same as with image features in DepthProEncoder
        # except for the extra encoder_neck layer applied

        image_scaled_to_patch_size = interpolate(
            pixel_values,
            size=(self.config.patch_size, self.config.patch_size),
        )
        encodings = self.encoder(
            image_scaled_to_patch_size,
            head_mask=head_mask,
        )

        # a. extract hidden_state
        hidden_state = encodings.last_hidden_state  # (batch_size, seq_len, hidden_size)
        # extra step
        hidden_state = self.encoder_neck(hidden_state)
        # (batch_size, seq_len, fusion_hidden_size//2)

        # b. reshape back to image like
        fov_features = reshape_feature(hidden_state)
        # (batch_size, fusion_hidden_size//2, out_size, out_size)

        # c. merge patches back together
        # no merge required for fov_features as they are already in batches instead of patches

        # d. interpolate patches to base size
        fov_features = interpolate(fov_features, size=(self.out_size, self.out_size), mode="nearest")

        global_features = self.global_neck(global_features)
        global_features = interpolate(global_features, size=(self.out_size, self.out_size), mode="nearest")

        fov_features = fov_features + global_features
        fov_output = self.head(fov_features)
        fov_output = fov_output.reshape(batch_size)

        return fov_output


class DepthProDepthEstimationHead(nn.Module):
    """
    The DepthProDepthEstimationHead module serves as the output head for depth estimation tasks.
    This module comprises a sequence of convolutional and transposed convolutional layers
    that process the feature map from the fusion to produce a single-channel depth map.
    Key operations include dimensionality reduction and upsampling to match the input resolution.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        features = config.fusion_hidden_size
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(
                in_channels=features // 2, out_channels=features // 2, kernel_size=2, stride=2, padding=0, bias=True
            ),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        predicted_depth = self.head(hidden_states)
        predicted_depth = predicted_depth.squeeze(dim=1)
        return predicted_depth


@add_start_docstrings(
    """
    DepthPro Model with a depth estimation head on top (consisting of 3 convolutional layers).
    """,
    DEPTH_PRO_FOR_DEPTH_ESTIMATION_START_DOCSTRING,
)
class DepthProForDepthEstimation(DepthProPreTrainedModel):
    def __init__(self, config, use_fov_model=None):
        super().__init__(config)
        self.config = config
        self.use_fov_model = use_fov_model if use_fov_model is not None else self.config.use_fov_model

        # dinov2 (vit) like encoders
        self.depth_pro = DepthProModel(config)

        # dpt (vit) like fusion stage
        self.fusion_stage = DepthProFeatureFusionStage(config)

        # depth estimation head
        self.head = DepthProDepthEstimationHead(config)

        # dinov2 (vit) like encoder
        self.fov_model = DepthProFOVModel(config) if self.use_fov_model else None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DEPTH_PRO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthProDepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, DepthProForDepthEstimation
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> checkpoint = "geetu040/DepthPro"
        >>> processor = AutoImageProcessor.from_pretrained(checkpoint)
        >>> model = DepthProForDepthEstimation.from_pretrained(checkpoint)

        >>> # prepare image for the model
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # interpolate to original size
        >>> post_processed_output = processor.post_process_depth_estimation(
        ...     outputs, target_sizes=[(image.height, image.width)],
        ... )

        >>> # visualize the prediction
        >>> predicted_depth = post_processed_output[0]["predicted_depth"]
        >>> depth = predicted_depth * 255 / predicted_depth.max()
        >>> depth = depth.detach().cpu().numpy()
        >>> depth = Image.fromarray(depth.astype("uint8"))
        ```"""
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        depth_pro_outputs = self.depth_pro(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        features = depth_pro_outputs.features
        fused_hidden_states = self.fusion_stage(features)
        predicted_depth = self.head(fused_hidden_states[-1])

        fov = (
            self.fov_model(
                pixel_values=pixel_values,
                # frozon features from encoder are used
                global_features=features[0].detach(),
                head_mask=head_mask,
            )
            if self.use_fov_model
            else None
        )

        if not return_dict:
            outputs = [loss, predicted_depth, fov, depth_pro_outputs.hidden_states, depth_pro_outputs.attentions]
            return tuple(v for v in outputs if v is not None)

        return DepthProDepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            fov=fov,
            hidden_states=depth_pro_outputs.hidden_states,
            attentions=depth_pro_outputs.attentions,
        )


__all__ = ["DepthProPreTrainedModel", "DepthProModel", "DepthProForDepthEstimation"]

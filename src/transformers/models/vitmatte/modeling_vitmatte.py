# coding=utf-8
# Copyright 2023 HUST-VL and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ViTMatte model."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...utils.backbone_utils import load_backbone
from .configuration_vitmatte import VitMatteConfig


# General docstring
_CONFIG_FOR_DOC = "VitMatteConfig"


@dataclass
class ImageMattingOutput(ModelOutput):
    """
    Class for outputs of image matting models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Loss.
        alphas (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
           Estimated alpha values.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    alphas: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class VitMattePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VitMatteConfig
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = []

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()


class VitMatteBasicConv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """

    def __init__(self, config, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps)
        self.relu = nn.ReLU()

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.batch_norm(hidden_state)
        hidden_state = self.relu(hidden_state)

        return hidden_state


class VitMatteConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """

    def __init__(self, config):
        super().__init__()

        # We use a default in-case there isn't a backbone config set. This is for backwards compatibility and
        # to enable loading HF backbone models.
        in_channels = 4
        if config.backbone_config is not None:
            in_channels = config.backbone_config.num_channels

        out_channels = config.convstream_hidden_sizes

        self.convs = nn.ModuleList()
        self.conv_chans = [in_channels] + out_channels

        for i in range(len(self.conv_chans) - 1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i + 1]
            self.convs.append(VitMatteBasicConv3x3(config, in_chan_, out_chan_))

    def forward(self, pixel_values):
        out_dict = {"detailed_feature_map_0": pixel_values}
        embeddings = pixel_values
        for i in range(len(self.convs)):
            embeddings = self.convs[i](embeddings)
            name_ = "detailed_feature_map_" + str(i + 1)
            out_dict[name_] = embeddings

        return out_dict


class VitMatteFusionBlock(nn.Module):
    """
    Simple fusion block to fuse features from ConvStream and Plain Vision Transformer.
    """

    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        self.conv = VitMatteBasicConv3x3(config, in_channels, out_channels, stride=1, padding=1)

    def forward(self, features, detailed_feature_map):
        upscaled_features = nn.functional.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        out = torch.cat([detailed_feature_map, upscaled_features], dim=1)
        out = self.conv(out)

        return out


class VitMatteHead(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """

    def __init__(self, config):
        super().__init__()

        in_channels = config.fusion_hidden_sizes[-1]
        mid_channels = 16

        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, hidden_state):
        hidden_state = self.matting_convs(hidden_state)

        return hidden_state


class VitMatteDetailCaptureModule(nn.Module):
    """
    Simple and lightweight Detail Capture Module for ViT Matting.
    """

    def __init__(self, config):
        super().__init__()
        if len(config.fusion_hidden_sizes) != len(config.convstream_hidden_sizes) + 1:
            raise ValueError(
                "The length of fusion_hidden_sizes should be equal to the length of convstream_hidden_sizes + 1."
            )

        self.config = config
        self.convstream = VitMatteConvStream(config)
        self.conv_chans = self.convstream.conv_chans

        self.fusion_blocks = nn.ModuleList()
        self.fusion_channels = [config.hidden_size] + config.fusion_hidden_sizes

        for i in range(len(self.fusion_channels) - 1):
            self.fusion_blocks.append(
                VitMatteFusionBlock(
                    config=config,
                    in_channels=self.fusion_channels[i] + self.conv_chans[-(i + 1)],
                    out_channels=self.fusion_channels[i + 1],
                )
            )

        self.matting_head = VitMatteHead(config)

    def forward(self, features, pixel_values):
        detail_features = self.convstream(pixel_values)
        for i in range(len(self.fusion_blocks)):
            detailed_feature_map_name = "detailed_feature_map_" + str(len(self.fusion_blocks) - i - 1)
            features = self.fusion_blocks[i](features, detail_features[detailed_feature_map_name])

        alphas = torch.sigmoid(self.matting_head(features))

        return alphas


VITMATTE_START_DOCSTRING = r"""
    Parameters:
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
        config ([`UperNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VITMATTE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`VitMatteImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers in case the backbone has them. See
            `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers of the backbone. See `hidden_states` under
            returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """ViTMatte framework leveraging any vision backbone e.g. for ADE20k, CityScapes.""",
    VITMATTE_START_DOCSTRING,
)
class VitMatteForImageMatting(VitMattePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.backbone = load_backbone(config)
        self.decoder = VitMatteDetailCaptureModule(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VITMATTE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=ImageMattingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth image matting for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import VitMatteImageProcessor, VitMatteForImageMatting
        >>> import torch
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download

        >>> processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
        >>> model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k")

        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="image.png", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")
        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="trimap.png", repo_type="dataset"
        ... )
        >>> trimap = Image.open(filepath).convert("L")

        >>> # prepare image + trimap for the model
        >>> inputs = processor(images=image, trimaps=trimap, return_tensors="pt")

        >>> with torch.no_grad():
        ...     alphas = model(**inputs).alphas
        >>> print(alphas.shape)
        torch.Size([1, 1, 640, 960])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )

        features = outputs.feature_maps[-1]
        alphas = self.decoder(features, pixel_values)

        if not return_dict:
            output = (alphas,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageMattingOutput(
            loss=loss,
            alphas=alphas,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["VitMattePreTrainedModel", "VitMatteForImageMatting"]

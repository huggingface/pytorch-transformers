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
"""PyTorch SuperPoint model."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
)
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig

from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "SuperPointConfig"

_CHECKPOINT_FOR_DOC = "magic-leap-community/superpoint"


def remove_keypoints_from_borders(
    keypoints: torch.Tensor, scores: torch.Tensor, border: int, height: int, width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Removes keypoints (and their associated scores) that are too close to the border"""
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints: torch.Tensor, scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keeps the k keypoints with highest score"""
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """Applies non-maximum suppression on scores"""
    if nms_radius < 0:
        raise ValueError("Expected positive values for nms_radius")

    def max_pool(x):
        return nn.functional.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


@dataclass
class SuperPointKeypointDescriptionOutput(ModelOutput):
    """
    Base class for outputs of image point description models. Due to the nature of keypoint detection, the number of
    keypoints is not fixed and can vary from image to image, which makes batching non-trivial. In the batch of images,
    the maximum number of keypoints is set as the dimension of the keypoints, scores and descriptors tensors. The mask
    tensor is used to indicate which values in the keypoints, scores and descriptors tensors are keypoint information
    and which are padding.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss computed during training.
        keypoints (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`):
            Relative (x, y) coordinates of predicted keypoints in a given image.
        scores (`torch.FloatTensor` of shape `(batch_size, num_keypoints)`):
            Scores of predicted keypoints.
        descriptors (`torch.FloatTensor` of shape `(batch_size, num_keypoints, descriptor_size)`):
            Descriptors of predicted keypoints.
        mask (`torch.BoolTensor` of shape `(batch_size, num_keypoints)`):
            Mask indicating which values in keypoints, scores and descriptors are keypoint information.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
    """

    loss: Optional[torch.FloatTensor] = None
    keypoints: Optional[torch.IntTensor] = None
    scores: Optional[torch.FloatTensor] = None
    descriptors: Optional[torch.FloatTensor] = None
    mask: Optional[torch.BoolTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class SuperPointConvBlock(nn.Module):
    def __init__(
        self, config: SuperPointConfig, in_channels: int, out_channels: int, add_pooling: bool = False
    ) -> None:
        super().__init__()
        self.conv_a = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_b = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if add_pooling else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.relu(self.conv_a(hidden_states))
        hidden_states = self.relu(self.conv_b(hidden_states))
        if self.pool is not None:
            hidden_states = self.pool(hidden_states)
        return hidden_states


class SuperPointEncoder(nn.Module):
    """
    SuperPoint encoder module. It is made of 4 convolutional layers with ReLU activation and max pooling, reducing the
     dimensionality of the image.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()
        # SuperPoint uses 1 channel images
        self.input_dim = 1

        conv_blocks = []
        conv_blocks.append(
            SuperPointConvBlock(config, self.input_dim, config.encoder_hidden_sizes[0], add_pooling=True)
        )
        for i in range(1, len(config.encoder_hidden_sizes) - 1):
            conv_blocks.append(
                SuperPointConvBlock(
                    config, config.encoder_hidden_sizes[i - 1], config.encoder_hidden_sizes[i], add_pooling=True
                )
            )
        conv_blocks.append(
            SuperPointConvBlock(
                config, config.encoder_hidden_sizes[-2], config.encoder_hidden_sizes[-1], add_pooling=False
            )
        )
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(
        self,
        input,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        for conv_block in self.conv_blocks:
            input = conv_block(input)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (input,)
        output = input
        if not return_dict:
            return tuple(v for v in [output, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=output,
            hidden_states=all_hidden_states,
        )


class SuperPointInterestPointDecoder(nn.Module):
    """
    The SuperPointInterestPointDecoder uses the output of the SuperPointEncoder to compute the keypoint with scores.
    The scores are first computed by a convolutional layer, then a softmax is applied to get a probability distribution
    over the 65 possible keypoint classes. The keypoints are then extracted from the scores by thresholding and
    non-maximum suppression. Post-processing is then applied to remove keypoints too close to the image borders as well
    as to keep only the k keypoints with highest score.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()
        self.keypoint_threshold = config.keypoint_threshold
        self.max_keypoints = config.max_keypoints
        self.nms_radius = config.nms_radius
        self.border_removal_distance = config.border_removal_distance

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_score_a = nn.Conv2d(
            config.encoder_hidden_sizes[-1],
            config.decoder_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_score_b = nn.Conv2d(
            config.decoder_hidden_size, config.keypoint_decoder_dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self._get_pixel_scores(encoded)
        keypoints, scores = self._extract_keypoints(scores)

        return keypoints, scores

    def _get_pixel_scores(self, encoded: torch.Tensor) -> torch.Tensor:
        """Based on the encoder output, compute the scores for each pixel of the image"""
        scores = self.relu(self.conv_score_a(encoded))
        scores = self.conv_score_b(scores)
        scores = nn.functional.softmax(scores, 1)[:, :-1]
        batch_size, _, height, width = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(batch_size, height, width, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(batch_size, height * 8, width * 8)
        scores = simple_nms(scores, self.nms_radius)
        return scores

    def _extract_keypoints(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on their scores, extract the pixels that represent the keypoints that will be used for descriptors computation.
        The keypoints are in the form of relative (x, y) coordinates.
        """
        _, height, width = scores.shape

        # Threshold keypoints by score value
        keypoints = torch.nonzero(scores[0] > self.keypoint_threshold)
        scores = scores[0][tuple(keypoints.t())]

        # Discard keypoints near the image borders
        keypoints, scores = remove_keypoints_from_borders(
            keypoints, scores, self.border_removal_distance, height * 8, width * 8
        )

        # Keep the k keypoints with highest score
        if self.max_keypoints >= 0:
            keypoints, scores = top_k_keypoints(keypoints, scores, self.max_keypoints)

        # Convert (y, x) to (x, y)
        keypoints = torch.flip(keypoints, [1]).float()

        return keypoints, scores


class SuperPointDescriptorDecoder(nn.Module):
    """
    The SuperPointDescriptorDecoder uses the outputs of both the SuperPointEncoder and the
    SuperPointInterestPointDecoder to compute the descriptors at the keypoints locations.

    The descriptors are first computed by a convolutional layer, then normalized to have a norm of 1. The descriptors
    are then interpolated at the keypoints locations.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_descriptor_a = nn.Conv2d(
            config.encoder_hidden_sizes[-1],
            config.decoder_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_descriptor_b = nn.Conv2d(
            config.decoder_hidden_size,
            config.descriptor_decoder_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, encoded: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """Based on the encoder output and the keypoints, compute the descriptors for each keypoint"""
        descriptors = self.conv_descriptor_b(self.relu(self.conv_descriptor_a(encoded)))
        descriptors = nn.functional.normalize(descriptors, p=2, dim=1)

        descriptors = self._sample_descriptors(keypoints[None], descriptors[0][None], 8)[0]

        # [descriptor_dim, num_keypoints] -> [num_keypoints, descriptor_dim]
        descriptors = torch.transpose(descriptors, 0, 1)

        return descriptors

    @staticmethod
    def _sample_descriptors(keypoints, descriptors, scale: int = 8) -> torch.Tensor:
        """Interpolate descriptors at keypoint locations"""
        batch_size, num_channels, height, width = descriptors.shape
        keypoints = keypoints - scale / 2 + 0.5
        divisor = torch.tensor([[(width * scale - scale / 2 - 0.5), (height * scale - scale / 2 - 0.5)]])
        divisor = divisor.to(keypoints)
        keypoints /= divisor
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        kwargs = {"align_corners": True}
        # [batch_size, num_channels, num_keypoints, 2] -> [batch_size, num_channels, num_keypoints, 2]
        keypoints = keypoints.view(batch_size, 1, -1, 2)
        descriptors = nn.functional.grid_sample(descriptors, keypoints, mode="bilinear", **kwargs)
        # [batch_size, descriptor_decoder_dim, num_channels, num_keypoints] -> [batch_size, descriptor_decoder_dim, num_keypoints]
        descriptors = descriptors.reshape(batch_size, num_channels, -1)
        descriptors = nn.functional.normalize(descriptors, p=2, dim=1)
        return descriptors


class SuperPointPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SuperPointConfig
    base_model_prefix = "superpoint"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def extract_one_channel_pixel_values(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """
        Assuming pixel_values has shape (batch_size, 3, height, width), and that all channels values are the same,
        extract the first channel value to get a tensor of shape (batch_size, 1, height, width) for SuperPoint. This is
        a workaround for the issue discussed in :
        https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

        Args:
            pixel_values: torch.FloatTensor of shape (batch_size, 3, height, width)

        Returns:
            pixel_values: torch.FloatTensor of shape (batch_size, 1, height, width)

        """
        return pixel_values[:, 0, :, :][:, None, :, :]


SUPERPOINT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SuperPointConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

SUPERPOINT_INPUTS_DOCSTRING = r"""
Args:
    pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        Pixel values. Pixel values can be obtained using [`SuperPointImageProcessor`]. See
        [`SuperPointImageProcessor.__call__`] for details.
    output_hidden_states (`bool`, *optional*):
        Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more
        detail.
    return_dict (`bool`, *optional*):
        Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """


@add_start_docstrings(
    "SuperPoint model outputting keypoints and descriptors.",
    SUPERPOINT_START_DOCSTRING,
)
class SuperPointForKeypointDetection(SuperPointPreTrainedModel):
    """
    SuperPoint model. It consists of a SuperPointEncoder, a SuperPointInterestPointDecoder and a
    SuperPointDescriptorDecoder. SuperPoint was proposed in `SuperPoint: Self-Supervised Interest Point Detection and
    Description <https://arxiv.org/abs/1712.07629>`__ by Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. It
    is a fully convolutional neural network that extracts keypoints and descriptors from an image. It is trained in a
    self-supervised manner, using a combination of a photometric loss and a loss based on the homographic adaptation of
    keypoints. It is made of a convolutional encoder and two decoders: one for keypoints and one for descriptors.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__(config)

        self.config = config

        self.encoder = SuperPointEncoder(config)
        self.keypoint_decoder = SuperPointInterestPointDecoder(config)
        self.descriptor_decoder = SuperPointDescriptorDecoder(config)

        self.post_init()

    @add_start_docstrings_to_model_forward(SUPERPOINT_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SuperPointKeypointDescriptionOutput]:
        """
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, SuperPointForKeypointDetection
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        >>> model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        loss = None
        if labels is not None:
            raise ValueError("SuperPoint does not support training for now.")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pixel_values = self.extract_one_channel_pixel_values(pixel_values)

        batch_size, _, height, width = pixel_values.shape

        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        list_keypoints_scores = [
            self.keypoint_decoder(last_hidden_state[None, ...]) for last_hidden_state in last_hidden_state
        ]

        list_keypoints = [keypoints_scores[0] for keypoints_scores in list_keypoints_scores]
        list_scores = [keypoints_scores[1] for keypoints_scores in list_keypoints_scores]

        list_descriptors = [
            self.descriptor_decoder(last_hidden_state[None, ...], keypoints[None, ...])
            for last_hidden_state, keypoints in zip(last_hidden_state, list_keypoints)
        ]

        maximum_num_keypoints = max(keypoints.shape[0] for keypoints in list_keypoints)

        keypoints = torch.zeros((batch_size, maximum_num_keypoints, 2), device=pixel_values.device)
        scores = torch.zeros((batch_size, maximum_num_keypoints), device=pixel_values.device)
        descriptors = torch.zeros(
            (batch_size, maximum_num_keypoints, self.config.descriptor_decoder_dim),
            device=pixel_values.device,
        )
        mask = torch.zeros((batch_size, maximum_num_keypoints), device=pixel_values.device, dtype=torch.int)

        for i, (_keypoints, _scores, _descriptors) in enumerate(zip(list_keypoints, list_scores, list_descriptors)):
            keypoints[i, : _keypoints.shape[0]] = _keypoints
            scores[i, : _scores.shape[0]] = _scores
            descriptors[i, : _descriptors.shape[0]] = _descriptors
            mask[i, : _scores.shape[0]] = 1

        # Convert to relative coordinates
        keypoints = keypoints / torch.tensor([width, height], device=keypoints.device)

        hidden_states = encoder_outputs[1] if output_hidden_states else None
        if not return_dict:
            return tuple(v for v in [loss, keypoints, scores, descriptors, mask, hidden_states] if v is not None)

        return SuperPointKeypointDescriptionOutput(
            loss=loss,
            keypoints=keypoints,
            scores=scores,
            descriptors=descriptors,
            mask=mask,
            hidden_states=hidden_states,
        )


__all__ = ["SuperPointForKeypointDetection", "SuperPointPreTrainedModel"]

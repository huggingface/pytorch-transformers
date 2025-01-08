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
"""Image processor class for SuperPoint."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from ... import is_torch_available, is_vision_available
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, logging, requires_backends


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from .modeling_superpoint import SuperPointKeypointDescriptionOutput

if is_vision_available():
    import PIL

logger = logging.get_logger(__name__)


def is_grayscale(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    if input_data_format == ChannelDimension.FIRST:
        return np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...])
    elif input_data_format == ChannelDimension.LAST:
        return np.all(image[..., 0] == image[..., 1]) and np.all(image[..., 1] == image[..., 2])


def convert_to_grayscale(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> ImageInput:
    """
    Converts an image to grayscale format using the NTSC formula. Only support numpy and PIL Image. TODO support torch
    and tensorflow grayscale conversion

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (Image):
            The image to convert.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
    """
    requires_backends(convert_to_grayscale, ["vision"])

    if isinstance(image, np.ndarray):
        if input_data_format == ChannelDimension.FIRST:
            gray_image = image[0, ...] * 0.2989 + image[1, ...] * 0.5870 + image[2, ...] * 0.1140
            gray_image = np.stack([gray_image] * 3, axis=0)
        elif input_data_format == ChannelDimension.LAST:
            gray_image = image[..., 0] * 0.2989 + image[..., 1] * 0.5870 + image[..., 2] * 0.1140
            gray_image = np.stack([gray_image] * 3, axis=-1)
        return gray_image

    if not isinstance(image, PIL.Image.Image):
        return image

    image = image.convert("L")
    return image


class SuperPointImageProcessor(BaseImageProcessor):
    r"""
    Constructs a SuperPoint image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be overriden
            by `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 480, "width": 640}`):
            Resolution of the output image after `resize` is applied. Only has an effect if `do_resize` is set to
            `True`. Can be overriden by `size` in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overriden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overriden by `rescale_factor` in the `preprocess`
            method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 480, "width": 640}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary of the form `{"height": int, "width": int}`, specifying the size of the output image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the output image. If not provided, it will be inferred from the input
                image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        size = get_size_dict(size, default_to_square=False)

        return resize(
            image,
            size=(size["height"], size["width"]),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the output image after `resize` has been applied. If `size["shortest_edge"]` >= 384, the image
                is resized to `(size["shortest_edge"], size["shortest_edge"])`. Otherwise, the smaller edge of the
                image will be matched to `int(size["shortest_edge"]/ crop_pct)`, after which the image is cropped to
                `(size["shortest_edge"], size["shortest_edge"])`. Only has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """

        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None:
            raise ValueError("Size must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_resize:
            images = [self.resize(image=image, size=size, input_data_format=input_data_format) for image in images]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        # Checking if image is RGB or grayscale
        for i in range(len(images)):
            if not is_grayscale(images[i], input_data_format):
                images[i] = convert_to_grayscale(images[i], input_data_format=input_data_format)

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_keypoint_detection(
        self, outputs: "SuperPointKeypointDescriptionOutput", target_sizes: Union[TensorType, List[Tuple]]
    ) -> List[Dict[str, "torch.Tensor"]]:
        """
        Converts the raw output of [`SuperPointForKeypointDetection`] into lists of keypoints, scores and descriptors
        with coordinates absolute to the original image sizes.

        Args:
            outputs ([`SuperPointKeypointDescriptionOutput`]):
                Raw outputs of the model containing keypoints in a relative (x, y) format, with scores and descriptors.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. This must be the original
                image size (before any processing).
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the keypoints in absolute format according
            to target_sizes, scores and descriptors for an image in the batch as predicted by the model.
        """
        if len(outputs.mask) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the mask")

        if isinstance(target_sizes, List):
            image_sizes = torch.tensor(target_sizes)
        else:
            if target_sizes.shape[1] != 2:
                raise ValueError(
                    "Each element of target_sizes must contain the size (h, w) of each image of the batch"
                )
            image_sizes = target_sizes

        # Flip the image sizes to (width, height) and convert keypoints to absolute coordinates
        image_sizes = torch.flip(image_sizes, [1])
        masked_keypoints = outputs.keypoints * image_sizes[:, None]

        # Convert masked_keypoints to int
        masked_keypoints = masked_keypoints.to(torch.int32)

        results = []
        for image_mask, keypoints, scores, descriptors in zip(
            outputs.mask, masked_keypoints, outputs.scores, outputs.descriptors
        ):
            indices = torch.nonzero(image_mask).squeeze(1)
            keypoints = keypoints[indices]
            scores = scores[indices]
            descriptors = descriptors[indices]
            results.append({"keypoints": keypoints, "scores": scores, "descriptors": descriptors})

        return results


__all__ = ["SuperPointImageProcessor"]

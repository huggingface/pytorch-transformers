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
"""Image processor class for LightGlue."""

from typing import Dict, Optional, Union

import numpy as np

from ... import is_vision_available
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, logging, requires_backends


if is_vision_available():
    import PIL

logger = logging.get_logger(__name__)


# Copied from transformers.models.superpoint.image_processing_superpoint.is_grayscale
def is_grayscale(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    if input_data_format == ChannelDimension.FIRST:
        if image.shape[0] == 1:
            return True
        return np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...])
    elif input_data_format == ChannelDimension.LAST:
        if image.shape[-1] == 1:
            return True
        return np.all(image[..., 0] == image[..., 1]) and np.all(image[..., 1] == image[..., 2])


# Copied from transformers.models.superpoint.image_processing_superpoint.convert_to_grayscale
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


def pad_images(images, pad_value=0, data_format: Optional[Union[str, ChannelDimension]] = None):
    """
    Given a list of images, pads them to the same height and width by adding `pad_value` around the edges.
    Args:
        images (`List[np.ndarray]`):
            List of images to pad.
        pad_value (`int`, *optional*, defaults to `0`):
            Value to use for padding.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
    Returns:
        padded_images (`List[np.ndarray]`):
            List of padded images.
    """
    if data_format == ChannelDimension.FIRST:
        max_height = max(image.shape[1] for image in images)
        max_width = max(image.shape[2] for image in images)
    else:  # channels_last
        max_height = max(image.shape[0] for image in images)
        max_width = max(image.shape[1] for image in images)

    padded_images = []

    for image in images:
        if data_format == ChannelDimension.FIRST:
            channels, height, width = image.shape
        else:
            height, width, channels = image.shape

        top_pad = (max_height - height) // 2
        left_pad = (max_width - width) // 2

        if data_format == ChannelDimension.FIRST:
            padded_image = np.full((channels, max_height, max_width), pad_value, dtype=image.dtype)
            padded_image[:, top_pad : top_pad + height, left_pad : left_pad + width] = image
        else:
            padded_image = np.full((max_height, max_width, channels), pad_value, dtype=image.dtype)
            padded_image[top_pad : top_pad + height, left_pad : left_pad + width, :] = image

        padded_images.append(padded_image)

    return padded_images


class LightGlueImageProcessor(BaseImageProcessor):
    r"""
    Constructs a LightGlue image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be overriden
            by `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 480, "width": 640}`):
            Resolution of the output image after `resize` is applied. Only has an effect if `do_resize` is set to
            `True`. Can be overriden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overriden by `resample` in the `preprocess` method.
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
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 480, "width": 640}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
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
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
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
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
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
                Image pairs to preprocess. Expects either a list of 2 images or a list of list of 2 images list with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the output image after `resize` has been applied. If `size["shortest_edge"]` >= 384, the image
                is resized to `(size["shortest_edge"], size["shortest_edge"])`. Otherwise, the smaller edge of the
                image will be matched to `int(size["shortest_edge"]/ crop_pct)`, after which the image is cropped to
                `(size["shortest_edge"], size["shortest_edge"])`. Only has an effect if `do_resize` is set to `True`.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of `PILImageResampling`, filters. Only
                has an effect if `do_resize` is set to `True`.
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
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)

        image_pairs = images

        if not isinstance(image_pairs, list):
            raise ValueError(
                "Input images must be a list containing at least 2 images because SuperGlue takes pairs of images."
            )
        elif len(image_pairs) == 2 and not isinstance(image_pairs[0], list):
            images = image_pairs
        else:
            for pair in image_pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(
                        "Input images must be a list of pairs of images because SuperGlue takes pairs of images."
                    )
            images = [image for pair in image_pairs for image in pair]

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

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        # Checking if image is RGB or grayscale
        for i in range(len(images)):
            if not is_grayscale(images[i], input_data_format):
                images[i] = convert_to_grayscale(images[i], input_data_format=input_data_format)

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        if not do_resize:
            images = pad_images(images, data_format=data_format)

        image_pairs = [images[i : i + 2] for i in range(0, len(images), 2)]

        data = {"pixel_values": image_pairs}

        return BatchFeature(data=data, tensor_type=return_tensors)

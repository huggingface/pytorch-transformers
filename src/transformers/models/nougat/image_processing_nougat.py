# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Nougat."""

from typing import Dict, List, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,
    pad,
    resize,
    to_channel_dimension_format,
    to_pil_image,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, filter_out_non_signature_kwargs, logging
from ...utils.import_utils import is_cv2_available, is_vision_available


logger = logging.get_logger(__name__)


if is_cv2_available():
    pass


if is_vision_available():
    import PIL


class NougatImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Nougat image processor.

    Args:
        do_crop_margin (`bool`, *optional*, defaults to `True`):
            Whether to crop the image margins.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 896, "width": 672}`):
            Size of the image after resizing. Can be overridden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to resize the image using thumbnail method.
        do_align_long_axis (`bool`, *optional*, defaults to `False`):
            Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the images to the largest image size in the batch.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Image standard deviation.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_crop_margin: bool = True,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_thumbnail: bool = True,
        do_align_long_axis: bool = False,
        do_pad: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        size = size if size is not None else {"height": 896, "width": 672}
        size = get_size_dict(size)

        self.do_crop_margin = do_crop_margin
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_thumbnail = do_thumbnail
        self.do_align_long_axis = do_align_long_axis
        self.do_pad = do_pad
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    def python_find_non_zero(self, image: np.array):
        """This is a reimplementation of a findNonZero function equivalent to cv2."""
        non_zero_indices = np.column_stack(np.nonzero(image))
        idxvec = non_zero_indices[:, [1, 0]]
        idxvec = idxvec.reshape(-1, 1, 2)
        return idxvec

    def python_bounding_rect(self, coordinates):
        """This is a reimplementation of a BoundingRect function equivalent to cv2."""
        min_values = np.min(coordinates, axis=(0, 1)).astype(int)
        max_values = np.max(coordinates, axis=(0, 1)).astype(int)
        x_min, y_min = min_values[0], min_values[1]
        width = max_values[0] - x_min + 1
        height = max_values[1] - y_min + 1
        return x_min, y_min, width, height

    def crop_margin(
        self,
        image: np.array,
        gray_threshold: int = 200,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.array:
        """
        Crops the margin of the image. Gray pixels are considered margin (i.e., pixels with a value below the
        threshold).

        Args:
            image (`np.array`):
                The image to be cropped.
            gray_threshold (`int`, *optional*, defaults to `200`)
                Value below which pixels are considered to be gray.
            data_format (`ChannelDimension`, *optional*):
                The channel dimension format of the output image. If unset, will use the inferred format from the
                input.
            input_data_format (`ChannelDimension`, *optional*):
                The channel dimension format of the input image. If unset, will use the inferred format from the input.
        """
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        image = to_pil_image(image, input_data_format=input_data_format)
        data = np.array(image.convert("L")).astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            image = np.array(image)
            image = (
                to_channel_dimension_format(image, data_format, input_data_format)
                if data_format is not None
                else image
            )
            return image
        data = (data - min_val) / (max_val - min_val) * 255
        gray = data < gray_threshold
        coords = self.python_find_non_zero(gray)
        x_min, y_min, width, height = self.python_bounding_rect(coords)
        image = image.crop((x_min, y_min, x_min + width, y_min + height))
        image = np.array(image).astype(np.uint8)
        image = to_channel_dimension_format(image, input_data_format, ChannelDimension.LAST)

        image = (
            to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        )

        return image

    # Copied from transformers.models.donut.image_processing_donut.DonutImageProcessor.align_long_axis
    def align_long_axis(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Align the long axis of the image to the longest axis of the specified size.

        Args:
            image (`np.ndarray`):
                The image to be aligned.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to align the long axis to.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `np.ndarray`: The aligned image.
        """
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = size["height"], size["width"]

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = np.rot90(image, 3)

        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        return image

    def pad_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad the image to the specified size at the top, bottom, left and right.

        Args:
            image (`np.ndarray`):
                The image to be padded.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to pad the image to.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        output_height, output_width = size["height"], size["width"]
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        pad_top = delta_height // 2
        pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        return pad(image, padding, data_format=data_format, input_data_format=input_data_format)

    # Copied from transformers.models.donut.image_processing_donut.DonutImageProcessor.thumbnail
    def thumbnail(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any
        corresponding dimension of the specified size.

        Args:
            image (`np.ndarray`):
                The image to be resized.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to resize the image to.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                The resampling filter to use.
            data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = size["height"], size["width"]

        # We always resize to the smallest of either the input or output size.
        height = min(input_height, output_height)
        width = min(input_width, output_width)

        if height == input_height and width == input_width:
            return image

        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        return resize(
            image,
            size=(height, width),
            resample=resample,
            reducing_gap=2.0,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # Copied from transformers.models.donut.image_processing_donut.DonutImageProcessor.resize
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resizes `image` to `(height, width)` specified by `size` using the PIL library.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        size = get_size_dict(size)
        shortest_edge = min(size["height"], size["width"])
        output_size = get_resize_output_image_size(
            image, size=shortest_edge, default_to_square=False, input_data_format=input_data_format
        )
        resized_image = resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        return resized_image

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_crop_margin: bool = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        do_rescale: bool = None,
        rescale_factor: Union[int, float] = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255.
            do_crop_margin (`bool`, *optional*, defaults to `self.do_crop_margin`):
                Whether to crop the image margins.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to min(size["height"],
                size["width"]) with the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_thumbnail (`bool`, *optional*, defaults to `self.do_thumbnail`):
                Whether to resize the image using thumbnail method.
            do_align_long_axis (`bool`, *optional*, defaults to `self.do_align_long_axis`):
                Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the images to the largest image size in the batch.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image by the specified scale `rescale_factor`.
            rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: defaults to the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_crop_margin = do_crop_margin if do_crop_margin is not None else self.do_crop_margin
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_thumbnail = do_thumbnail if do_thumbnail is not None else self.do_thumbnail
        do_align_long_axis = do_align_long_axis if do_align_long_axis is not None else self.do_align_long_axis
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            size_divisibility=size,  # There is no pad divisibility in this processor, but pad requires the size arg.
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

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

        if do_crop_margin:
            images = [self.crop_margin(image, input_data_format=input_data_format) for image in images]

        if do_align_long_axis:
            images = [self.align_long_axis(image, size=size, input_data_format=input_data_format) for image in images]

        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        if do_thumbnail:
            images = [self.thumbnail(image=image, size=size, input_data_format=input_data_format) for image in images]

        if do_pad:
            images = [self.pad_image(image=image, size=size, input_data_format=input_data_format) for image in images]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["NougatImageProcessor"]

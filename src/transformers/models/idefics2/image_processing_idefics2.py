# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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


from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import PaddingMode, pad, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    import PIL
    from PIL import Image


def get_resize_output_image_size(image, size, input_data_format) -> Tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.

    Args:
        image (`np.ndarray`):
            Image to resize.
        size (`Dict[str, int]`):
            Size of the output image containing the keys "shortest_edge" and "longest_edge".
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        The output size of the image after resizing.
    """
    height, width = get_image_size(image, channel_dim=input_data_format)

    min_len = size["shortest_edge"]
    max_len = size["longest_edge"]
    aspect_ratio = width / height

    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)
    height = max(height, min_len)
    width = max(width, min_len)
    return height, width


def make_list_of_images(images: ImageInput) -> List[List[np.ndarray]]:
    """
    Convert a single image or a list of images to a list of numpy arrays.

    Args:
        images (`ImageInput`):
            A single image or a list of images.

    Returns:
        A list of numpy arrays.
    """
    # If it's a single image, convert it to a list of lists
    if is_valid_image(images):
        images = [[images]]
    # If it's a list of images, it's a single batch, so convert it to a list of lists
    elif isinstance(images, (list, tuple)) and len(images) > 0 and is_valid_image(images[0]):
        images = [images]
    # If it's a list of batches, it's already in the right format
    elif (
        isinstance(images, (list, tuple))
        and len(images) > 0
        and isinstance(images[0], (list, tuple))
        and len(images[0]) > 0
        and is_valid_image(images[0][0])
    ):
        pass
    else:
        raise ValueError(
            "Invalid input type. Must be a single image, a list of images, or a list of batches of images."
        )
    return images


# Copied from transformers.models.detr.image_processing_detr.max_across_indices
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


def get_max_height_width(
    images_list: List[List[np.ndarray]], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images_list[0][0])

    image_sizes = []
    for images in images_list:
        for image in images:
            image_sizes.append(get_image_size(image, channel_dim=input_data_format))

    max_height, max_width = max_across_indices(image_sizes)
    return (max_height, max_width)


# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


# FIXME Amy: merge this function with the one in image_transforms.py
def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.
    Args:
        image (Image):
            The image to convert.
    """
    if not isinstance(image, PIL.Image.Image):
        return image

    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


class Idefics2ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Idefics image processor.

    Args:
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB. This is useful if the input image is of a different format e.g. RGBA.
            Only has an effect if the input image is in the PIL format.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image. The longest edge of the image is resized to  be <= `size["longest_edge"]`, with the
            shortest edge resized to keep the input aspect ratio, with a minimum size of `size["shortest_edge"]`.
        size (`Dict`, *optional*):
            Controls the size of the output image. This is a dictionary containing the keys "shortest_edge" and "longest_edge".
        resample (`Resampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image. If set to `True`, the image is rescaled to have pixel values between 0 and 1.
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            Rescale factor to rescale the image by if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. If set to `True`, the image is normalized to have a mean of `image_mean` and
            a standard deviation of `image_std`.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether or not to pad the images to the largest height and width in the batch and number of images per
            sample in the batch, such that the returned tensor is of shape (batch_size, max_num_images, num_channels, max_height, max_width).
        do_image_splitting (`bool`, *optional*, defaults to `False`):
            Whether to split the image into a sequence 4 equal sub-images concatenated with the original image. That
            strategy was first introduced in https://arxiv.org/abs/2311.06607.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        do_image_splitting: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_convert_rgb = do_convert_rgb
        self.do_resize = do_resize
        self.size = size if size is not None else {"shortest_edge": 378, "longest_edge": 980}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_pad = do_pad
        self.do_image_splitting = do_image_splitting

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

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
        if "shortest_edge" in size and "longest_edge" in size:
            size = get_resize_output_image_size(image, size, input_data_format)
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError(
                "size must be a dictionary with keys 'shortest_edge' and 'longest_edge' or 'height' and 'width'."
            )
        return resize(
            image, size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )

    # Copied from transformers.models.vilt.image_processing_vilt.ViltImageProcessor._pad_image
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image with zeros to the given size.
        """
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return padded_image

    def pad(
        self,
        images: List[np.ndarray],
        constant_values: Union[float, Iterable[float]] = 0,
        return_pixel_mask: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        For a list of images, for each images, pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width.
        For each sample in the batch, pads the sample with empty images to the max_number of images per sample in the batch. Optionally returns a pixel mask.

        Args:
            images (`np.ndarray`):
                List of list of images to pad. Pads to the largest height and width in the batch.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a pixel mask.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        batch_size = len(images)
        max_num_images = max(len(images_) for images_ in images)
        input_data_format = (
            infer_channel_dimension_format(images[0][0]) if input_data_format is None else input_data_format
        )
        data_format = input_data_format if data_format is None else data_format

        def empty_image(size, input_data_format):
            if input_data_format == ChannelDimension.FIRST:
                return np.zeros((3, *size), dtype=np.uint8)
            elif input_data_format == ChannelDimension.LAST:
                return np.zeros((*size, 3), dtype=np.uint8)
            raise ValueError("Invalid channel dimension format.")

        padded_images_list = [
            [empty_image(pad_size, data_format) for _ in range(max_num_images)] for _ in range(batch_size)
        ]
        padded_masks = [[np.zeros(pad_size) for _ in range(max_num_images)] for _ in range(batch_size)]

        for batch_idx in range(batch_size):
            for sample_idx, image in enumerate(images[batch_idx]):
                padded_images_list[batch_idx][sample_idx] = self._pad_image(
                    image,
                    pad_size,
                    constant_values=constant_values,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                padded_masks[batch_idx][sample_idx] = make_pixel_mask(
                    image, output_size=pad_size, input_data_format=input_data_format
                )

        padded_masks = padded_masks if return_pixel_mask else None
        return padded_images_list, padded_masks

    def _crop(
        self,
        im: np.ndarray,
        w1: int,
        h1: int,
        w2: int,
        h2: int,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        if input_data_format == ChannelDimension.FIRST:
            return im[:, h1:h2, w1:w2]
        elif input_data_format == ChannelDimension.LAST:
            return im[h1:h2, w1:w2, :]

    def split_image(
        self,
        image: np.ndarray,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Split an image into 4 equal sub-images, and the concatenate that sequence with the original image.
        That means that a single image becomes a sequence of 5 images.
        This is a "trick" to spend more compute on each image with no changes in the vision encoder.

        Args:
            image (`np.ndarray`):
                Images to split.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        height, width = get_image_size(image, input_data_format)

        mid_width = width // 2
        mid_height = height // 2
        return [
            self._crop(image, 0, 0, mid_width, mid_height, input_data_format),
            self._crop(image, mid_width, 0, width, mid_height, input_data_format),
            self._crop(image, 0, mid_height, mid_width, height, input_data_format),
            self._crop(image, mid_width, mid_height, width, height, input_data_format),
            image,
        ]

    def preprocess(
        self,
        images: ImageInput,
        do_convert_rgb: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        do_image_splitting: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[ChannelDimension] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
    ):
        """
        Preprocess a batch of images.

        Args:
            images (`ImageInput`):
                A list of images to preprocess.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether or not to pad the images to the largest height and width in the batch.
            do_image_splitting (`bool`, *optional*, defaults to `self.do_image_splitting`):
                Whether to split the image into a sequence 4 equal sub-images concatenated with the original image. That
                strategy was first introduced in https://arxiv.org/abs/2311.06607.
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
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_image_splitting = do_image_splitting if do_image_splitting is not None else self.do_image_splitting

        images_list = make_list_of_images(images)

        if not valid_images(images_list[0]):
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
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images_list = [[convert_to_rgb(image) for image in images] for images in images_list]

        # All transformations expect numpy arrays.
        images_list = [[to_numpy_array(image) for image in images] for images in images_list]

        if do_rescale and is_scaled_image(images_list[0][0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images_list[0][0])

        if do_image_splitting:
            new_images_list = []
            for images in images_list:
                new_images = []
                for image in images:
                    new_images.extend(self.split_image(image, input_data_format))
                new_images_list.append(new_images)
            images_list = new_images_list

        if do_resize:
            images_list = [
                [
                    self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                    for image in images
                ]
                for images in images_list
            ]

        if do_rescale:
            images_list = [
                [
                    self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                    for image in images
                ]
                for images in images_list
            ]

        if do_normalize:
            images_list = [
                [
                    self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                    for image in images
                ]
                for images in images_list
            ]

        pixel_attention_mask = None
        if do_pad:
            images_list, pixel_attention_mask = self.pad(
                images_list, return_pixel_mask=True, return_tensors=return_tensors, input_data_format=input_data_format
            )

        if data_format is not None:
            images_list = [
                [
                    to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
                    for image in images
                ]
                for images in images_list
            ]

        data = {"pixel_values": np.array(images_list) if do_pad else images_list}  # Faster tensor conversion
        if pixel_attention_mask is not None:
            data["pixel_attention_mask"] = np.array(pixel_attention_mask) if do_pad else pixel_attention_mask

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Idefics2ImageProcessor"]

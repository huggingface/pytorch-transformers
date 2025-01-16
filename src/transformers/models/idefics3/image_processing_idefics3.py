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

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import PaddingMode, pad, to_channel_dimension_format, to_pil_image
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_nested_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)
MAX_IMAGE_SIZE = 4096  # 4k resolution as absolute maximum


if is_vision_available():
    import PIL
    from PIL import Image


def _resize_output_size_rescale_to_max_len(
    height: int, width: int, min_len: Optional[int] = 1, max_len: Optional[int] = None
) -> Tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.
    Args:
        height (`int`):
            Height of the input image.
        width (`int`):
            Width of the input image.
        min_len (`int`, *optional*, defaults to 1):
            Minimum size of the output image.
        max_len (`int`, *optional*, defaults to the maximum size of the image):
            Maximum size of the output image.
    Returns:
        The output size of the image after resizing.
    """
    max_len = max(height, width) if max_len is None else max_len
    aspect_ratio = width / height

    if width >= height:
        width = max_len
        height = int(width / aspect_ratio)
        if height % 2 != 0:
            height += 1
    elif height > width:
        height = max_len
        width = int(height * aspect_ratio)
        if width % 2 != 0:
            width += 1

    # Avoid resizing to a size smaller than min_len
    height = max(height, min_len)
    width = max(width, min_len)
    return height, width


def _resize_output_size_scale_below_upper_bound(
    height: int, width: int, max_len: Optional[Dict[str, int]] = None
) -> Tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.
    Args:
        height (`int`):
            Height of the input image.
        width (`int`):
            Width of the input image.
        max_len (`Dict[str, int]`, *optional*, defaults to the maximum size of the image):
            Defines the maximum dimensions of the image.
    Returns:
        The output size of the image after resizing.
    """
    max_len = max(height, width) if max_len is None else max_len

    aspect_ratio = width / height
    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)

    # Avoid resizing to a size smaller than 1
    height = max(height, 1)
    width = max(width, 1)
    return height, width


def get_resize_output_image_size(
    image,
    resolution_max_side: int,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.
    Args:
        image (`np.ndarray`):
            Image to resize.
        resolution_max_side (`int`):
            The longest edge of the image will be resized to this value. The shortest edge will be resized to keep the
            input aspect ratio.
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.
    Returns:
        The output size of the image after resizing.
    """
    height, width = get_image_size(image, channel_dim=input_data_format)

    # Find the output size, when rescaling the longest edge to max_len and preserving the aspect ratio
    height, width = _resize_output_size_rescale_to_max_len(height, width, max_len=resolution_max_side)
    # Find the output size when scaling the image to be below the MAX_IMAGE_SIZE
    height, width = _resize_output_size_scale_below_upper_bound(height, width, max_len=MAX_IMAGE_SIZE)
    return height, width


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
        input_data_format = infer_channel_dimension_format(images_list[0][0], num_channels=(1, 3, 4))

    max_height = max_width = float("-inf")
    for images in images_list:
        for image in images:
            height, width = get_image_size(image, channel_dim=input_data_format)
            max_height = max(height, max_height)
            max_width = max(width, max_width)
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


def convert_to_rgb(
    image: np.ndarray,
    palette: Optional[PIL.ImagePalette.ImagePalette] = None,
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> ImageInput:
    """
    Converts an image to RGB format.
    Args:
        image (`np.ndarray`):
            The image to convert.
        palette (List[int], *optional*):
            The palette to use if given.
        data_format (ChannelDimension or str, *optional*):
            The channel dimension format for the output image. If not provided, it will be the same as the input image.
        input_data_format (ChannelDimension or str, *optional*):
            The channel dimension format of the input image.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image, num_channels=(1, 3, 4))

    # For all transformations, we want to keep the same data format as the input image unless otherwise specified.
    # The resized image from PIL will always have channels last, so find the input format first.
    data_format = input_data_format if data_format is None else data_format

    mode = "P" if palette is not None else None
    image = to_pil_image(image, image_mode=mode, input_data_format=input_data_format)
    if image.mode == "P" and palette is not None:
        image.putpalette(palette)

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")

    output_array = np.array(alpha_composite)
    # The image is always in channels last format after converting from a PIL image
    output_array = to_channel_dimension_format(output_array, data_format, input_channel_dim=ChannelDimension.LAST)
    return output_array


# FIXME Amy: make a more general crop function that isn't just centre crop
def _crop(
    image: np.ndarray,
    w1: int,
    h1: int,
    w2: int,
    h2: int,
    data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    if data_format is None:
        data_format = infer_channel_dimension_format(image, num_channels=(1, 3, 4))

    if data_format == ChannelDimension.FIRST:
        image = image[:, h1:h2, w1:w2]
    elif data_format == ChannelDimension.LAST:
        image = image[h1:h2, w1:w2, :]
    else:
        raise ValueError("Invalid channel dimension format.")

    return image


class Idefics3ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Idefics3 image processor.
    Args:
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB. This is useful if the input image is of a different format e.g. RGBA.
            Only has an effect if the input image is in the PIL format.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image. The longest edge of the image is resized to  be <= `size["longest_edge"]`, with the
            shortest edge resized to keep the input aspect ratio.
        size (`Dict`, *optional*, defaults to `{"longest_edge": 4 * 364}`):
            Controls the size of the output image. This is a dictionary containing the key "longest_edge".
            The image will be resized such that the longest edge is <= `size["longest_edge"]` and the shortest edge is resized
            to keep the input aspect ratio.
        resample (`Resampling`, *optional*, defaults to `Resampling.LANCZOS`):
            Resampling filter to use when resizing the image.
        do_image_splitting (`bool`, *optional*, defaults to `True`):
            Whether to split the image into sub-images concatenated with the original image. They are split into patches
            such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.
        max_image_size (`Dict`, *optional*, defaults to `{"longest_edge": 364}`):
            Maximum resolution of the patches of images accepted by the model. This is a dictionary containing the key "longest_edge".
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
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.LANCZOS,
        do_image_splitting: bool = True,
        max_image_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_convert_rgb = do_convert_rgb
        self.do_resize = do_resize
        self.size = size if size is not None else {"longest_edge": 4 * 364}
        self.resample = resample
        self.do_image_splitting = do_image_splitting
        self.max_image_size = max_image_size if max_image_size is not None else {"longest_edge": 364}
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_pad = do_pad

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.LANCZOS,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The longest edge of the image is resized to size["longest_edge"], with the shortest edge
        resized to keep the input aspect ratio. Can also be used with size["height"] and size["width"].
        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.LANCZOS`):
                Resampling filter to use when resizing the image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the output image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image, num_channels=(1, 3, 4))

        # For all transformations, we want to keep the same data format as the input image unless otherwise specified.
        # The resized image from PIL will always have channels last, so find the input format first.
        data_format = input_data_format if data_format is None else data_format

        if "longest_edge" in size:
            size = get_resize_output_image_size(
                image, resolution_max_side=size["longest_edge"], input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("size must be a dictionary with key 'longest_edge' or 'height' and 'width'.")

        image_mode = None
        if image.ndim == 2 or image.shape[-1] == 1:
            image_mode = "P"
        image = to_pil_image(image, image_mode=image_mode, input_data_format=input_data_format)

        resized_image = image.resize((size[1], size[0]), resample=resample)
        resized_image = np.array(resized_image)

        # If the input image channel dimension was of size 1, then it is dropped when converting to a PIL image
        # so we need to add it back if necessary.
        resized_image = np.expand_dims(resized_image, axis=-1) if resized_image.ndim == 2 else resized_image
        # The image is always in channels last format after converting from a PIL image
        resized_image = to_channel_dimension_format(
            resized_image, data_format, input_channel_dim=ChannelDimension.LAST
        )
        return resized_image

    def split_image(
        self,
        image,
        max_image_size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.LANCZOS,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Split an image into squares of side max_image_size and the original image resized to max_image_size.
        That means that a single image becomes a sequence of images.
        This is a "trick" to spend more compute on each image with no changes in the vision encoder.
        1) If one side of the original image is larger than `max_image_size`, resize it to `max_image_size` while preserving the aspect ratio.
        2) Divide the resulting image into `ceil(height / max_image_size)` x `ceil(width / max_image_size)`
        sub-images of the same size each (image_size, image_size). Typically, 364x364.
        3) Returns the list of the crops and the original image, in addition to the number of splits for the height and the width.
        Args:
            image (`np.ndarray`):
                Images to split.
            max_image_size (`Dict[str, int]`):
                Maximum size of the output image. If the image is larger than this size, it will be split into
                patches of this size, and the original image will be concatenated with the patches, resized to max_size.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.LANCZOS`):
                Resampling filter to use when resizing the image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the output image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        height, width = get_image_size(image, channel_dim=input_data_format)
        max_height = max_width = max_image_size["longest_edge"]

        frames = []
        if height > max_height or width > max_width:
            # Calculate the number of splits
            num_splits_h = math.ceil(height / max_height)
            num_splits_w = math.ceil(width / max_width)
            # Calculate the optimal width and height for the sub-images
            optimal_height = math.ceil(height / num_splits_h)
            optimal_width = math.ceil(width / num_splits_w)

            # Iterate through each row and column
            for r in range(num_splits_h):
                for c in range(num_splits_w):
                    # Calculate the starting point of the crop
                    start_x = c * optimal_width
                    start_y = r * optimal_height

                    # Calculate the ending point of the crop
                    end_x = min(start_x + optimal_width, width)
                    end_y = min(start_y + optimal_height, height)

                    # Crop the image
                    cropped_image = _crop(
                        image,
                        start_x,
                        start_y,
                        end_x,
                        end_y,
                        data_format=data_format,
                    )
                    frames.append(cropped_image)

            # For the global image at the end, we resize it to match the max_image_size, for cpu memory efficiency
            global_image_height, global_image_width = max_height, max_width
            if height != global_image_height or width != global_image_width:
                image = self.resize(
                    image,
                    {"height": global_image_height, "width": global_image_width},
                    resample=resample,
                    input_data_format=data_format,
                )
        else:
            num_splits_h, num_splits_w = 0, 0

        frames.append(image)

        return frames, num_splits_h, num_splits_w

    def resize_for_vision_encoder(
        self,
        image: np.ndarray,
        vision_encoder_max_size: int,
        resample: PILImageResampling = PILImageResampling.LANCZOS,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Resize images to be multiples of `vision_encoder_max_size` while preserving the aspect ratio.
        Args:
            image (`np.ndarray`):
                Images to resize.
            vision_encoder_max_size (`int`):
                Maximum size of the output image. If the image is larger than this size, it will be split into
                patches of this size, and the original image will be concatenated with the patches, resized to max_size.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.LANCZOS`):
                Resampling filter to use when resizing the image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the output image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred
        """
        height, width = get_image_size(image, channel_dim=input_data_format)

        aspect_ratio = width / height
        if width >= height:
            width = math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size
            height = int(width / aspect_ratio)
            height = math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size
        elif height > width:
            height = math.ceil(height / vision_encoder_max_size) * vision_encoder_max_size
            width = int(height * aspect_ratio)
            width = math.ceil(width / vision_encoder_max_size) * vision_encoder_max_size
        new_size = {"height": height, "width": width}
        return self.resize(
            image, size=new_size, resample=resample, input_data_format=input_data_format, data_format=data_format
        )

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
            images (`List[np.ndarray]`):
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
            infer_channel_dimension_format(images[0][0], num_channels=(1, 3, 4))
            if input_data_format is None
            else input_data_format
        )
        data_format = input_data_format if data_format is None else data_format

        if input_data_format == ChannelDimension.FIRST:
            n_channels = images[0][0].shape[0]
        elif input_data_format == ChannelDimension.LAST:
            n_channels = images[0][0].shape[-1]
        else:
            raise ValueError("Invalid channel dimension format.")

        def empty_image(size, input_data_format):
            if input_data_format == ChannelDimension.FIRST:
                return np.zeros((n_channels, *size), dtype=np.uint8)
            elif input_data_format == ChannelDimension.LAST:
                return np.zeros((*size, n_channels), dtype=np.uint8)

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

    def preprocess(
        self,
        images: ImageInput,
        do_convert_rgb: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_image_splitting: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        max_image_size: Optional[Dict[str, int]] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_row_col_info: bool = False,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
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
                Size of the image after resizing. With the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_image_splitting (`bool`, *optional*, defaults to `self.do_image_splitting`):
                Whether to split the image into sub-images concatenated with the original image. They are split into patches
                such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.
            max_image_size (`Dict`, *optional*, defaults to `self.max_image_size`):
                Maximum resolution of the images. If the image is larger than this size, the image is split into patches.
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
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            return_row_col_info (`bool`, *optional*, default to `False`):
                Whether to return the number of rows and columns of the split images. This is used for the
                `Idefics3Processor` to generate prompt strings based on the number of rows and columns.
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
        do_image_splitting = do_image_splitting if do_image_splitting is not None else self.do_image_splitting
        max_image_size = max_image_size if max_image_size is not None else self.max_image_size
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_pad = do_pad if do_pad is not None else self.do_pad

        images_list = make_nested_list_of_images(images)

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

        # save the palettes for conversion to RGB
        palettes_list = [
            [im.getpalette() if isinstance(im, Image.Image) and im.mode == "P" else None for im in images]
            for images in images_list
        ]

        # All transformations expect numpy arrays.
        images_list = [[to_numpy_array(image) for image in images] for images in images_list]

        # Extra channel dimension for grayscale images
        if input_data_format in [ChannelDimension.LAST, None]:
            images_list = [
                [np.expand_dims(img, axis=-1) if img.ndim == 2 else img for img in images] for images in images_list
            ]
        elif input_data_format == ChannelDimension.FIRST:
            images_list = [
                [np.expand_dims(img, axis=0) if img.ndim == 2 else img for img in images] for images in images_list
            ]

        if do_rescale and is_scaled_image(images_list[0][0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        # We assume that all images have the same channel dimension format.
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images_list[0][0], num_channels=(1, 3, 4))

        if do_resize:
            images_list = [
                [
                    self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                    for image in images
                ]
                for images in images_list
            ]

        if do_image_splitting:
            # We first resize both height and width of each image to the nearest max_image_size multiple, disregarding the aspect ratio
            # for size=(10, max_image_size) -> rescaled_size=(max_image_size, max_image_size)
            # for size=(11, max_image_size+1) -> rescaled_size=(max_image_size, max_image_size*2)
            images_list = [
                [
                    self.resize_for_vision_encoder(
                        image, max_image_size["longest_edge"], resample=resample, input_data_format=input_data_format
                    )
                    for image in images
                ]
                for images in images_list
            ]
            images_list_split_arrays = []
            palettes_list_split_arrays = []
            images_list_rows = []
            images_list_cols = []
            for images, palettes in zip(images_list, palettes_list):
                split_image_arrays = []
                split_palettes_arrays = []
                image_rows = []
                image_cols = []
                for image, palette in zip(images, palettes):
                    split_image_array, rows, cols = self.split_image(
                        image,
                        max_image_size=max_image_size,
                        input_data_format=input_data_format,
                    )
                    split_image_arrays.extend(split_image_array)
                    split_palettes_arrays.extend([palette] * len(split_image_array))
                    image_rows.append(rows)
                    image_cols.append(cols)
                images_list_split_arrays.append(split_image_arrays)
                palettes_list_split_arrays.append(split_palettes_arrays)
                images_list_rows.append(image_rows)
                images_list_cols.append(image_cols)
            images_list = images_list_split_arrays
            palettes_list = palettes_list_split_arrays
        else:
            # We square the images to max_image_size
            images_list = [
                [
                    self.resize(
                        image=image,
                        size={"height": max_image_size["longest_edge"], "width": max_image_size["longest_edge"]},
                        resample=resample,
                        input_data_format=input_data_format,
                    )
                    for image in images
                ]
                for images in images_list
            ]
            images_list_rows = [[0] * len(images) for images in images_list]
            images_list_cols = [[0] * len(images) for images in images_list]

        if do_convert_rgb:
            images_list = [
                [convert_to_rgb(img, palette) for img, palette in zip(images, palettes)]
                for images, palettes in zip(images_list, palettes_list)
            ]

        if do_rescale:
            images_list = [
                [self.rescale(image, rescale_factor, input_data_format=input_data_format) for image in images]
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

        # Faster tensor conversion
        data = {"pixel_values": np.array(images_list) if do_pad and return_tensors is not None else images_list}
        if pixel_attention_mask is not None:
            data["pixel_attention_mask"] = (
                np.array(pixel_attention_mask) if do_pad and return_tensors is not None else pixel_attention_mask
            )

        encoding = BatchFeature(data=data, tensor_type=return_tensors)

        # This is needed for generating correct text inputs in the processor - we don't pad to the max number of images
        if return_row_col_info:
            encoding["rows"] = images_list_rows
            encoding["cols"] = images_list_cols

        return encoding


__all__ = ["Idefics3ImageProcessor"]

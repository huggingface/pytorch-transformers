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
"""Image processor class for Pixtral."""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_torch_device, is_torch_dtype, is_torch_tensor, is_vision_available, logging
from ...utils.import_utils import requires_backends


logger = logging.get_logger(__name__)


if is_vision_available():
    import PIL


class BatchMixFeature(BatchFeature):
    def to(self, *args, **kwargs) -> "BatchMixFeature":
        """
        Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
        different `dtypes` and sending the `BatchFeature` to a different `device`.

        Args:
            args (`Tuple`):
                Will be passed to the `to(...)` function of the tensors.
            kwargs (`Dict`, *optional*):
                Will be passed to the `to(...)` function of the tensors.

        Returns:
            [`BatchFeature`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])
        import torch  # noqa

        new_data = {}
        device = kwargs.get("device")
        # Check if the args are a device or a dtype
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # The first argument is a dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            # check if v is a floating point
            if isinstance(v, list):
                new_data[k] = [
                    element.to(*args, **kwargs) for sample in v for element in sample if is_torch_tensor(element)
                ]
            elif isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                # cast and send to device
                new_data[k] = v.to(*args, **kwargs)
            elif isinstance(v, torch.Tensor) and device is not None:
                new_data[k] = v.to(device=device)
            else:
                new_data[k] = v
        self.data = new_data
        return self


# Copied from transformers.models.idefics2.image_processing_idefics2.make_list_of_images
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


# Adapted from function in image_transforms.py to ensure any transparent pixels are converted to white.
def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.
    Args:
        image (Image):
            The image to convert.
    """
    requires_backends(convert_to_rgb, ["vision"])

    if not isinstance(image, PIL.Image.Image):
        return image

    if image.mode == "RGB":
        return image

    # First we convert to RGBA to set background to white.
    image = image.convert("RGBA")

    # Create a new image with a white background.
    new_image = PIL.Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image = new_image.convert("RGB")
    return new_image


def _num_image_tokens(image_size: Tuple[int, int], patch_size: Tuple[int, int]) -> int:
    """
    Calculate the number of image tokens given the image size and patch size.

    Args:
        image_size (`Tuple[int, int]`):
            The size of the image as `(height, width)`.
        patch_size (`Tuple[int, int]`):
            The patch size as `(height, width)`.

    Returns:
        `int`: The number of image tokens.
    """
    height, width = image_size
    patch_height, patch_width = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
    num_width_tokens = (width - 1) // patch_width + 1
    num_height_tokens = (height - 1) // patch_height + 1
    return num_height_tokens, num_width_tokens


def get_resize_output_image_size(
    input_image: ImageInput,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    patch_size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> tuple:
    """
    Find the target (height, width) dimension of the output image after resizing given the input image and the desired
    size.

    Args:
        input_image (`ImageInput`):
            The image to resize.
        size (`int` or `Tuple[int, int]`):
            Max image size an input image can be. Must be a dictionary with the key "longest_edge".
        patch_size (`int` or `Tuple[int, int]`):
            The patch_size as `(height, width)` to use for resizing the image. If patch_size is an integer, `(patch_size, patch_size)`
            will be used
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `tuple`: The target (height, width) dimension of the output image after resizing.
    """
    max_height, max_width = size if isinstance(size, (tuple, list)) else (size, size)
    patch_height, patch_width = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
    height, width = get_image_size(input_image, input_data_format)

    ratio = max(height / max_height, width / max_width)

    if ratio > 1:
        # Orgiginal implementation uses `round` which utilises bankers rounding, which can lead to surprising results
        height = int(math.ceil(height / ratio))
        width = int(math.ceil(width / ratio))

    num_height_tokens, num_width_tokens = _num_image_tokens((height, width), (patch_height, patch_width))
    return num_height_tokens * patch_height, num_width_tokens * patch_width


# Hack to get tensor conversion used in BatchFeature without batching the images
def _get_is_as_tensor_fns(tensor_type: Union[str, TensorType]) -> Tuple[Callable, Callable]:
    return BatchFeature()._get_is_as_tensor_fns(tensor_type)


def convert_to_tensor(array, tensor_type: Union[str, TensorType]) -> Any:
    is_tensor, as_tensor = _get_is_as_tensor_fns(tensor_type)
    if is_tensor(array):
        return array
    return as_tensor(array)


class PixtralImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Pixtral image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"longest_edge": 1024}`):
            Size of the maximum dimension of either the height or width dimension of the image. Used to control how
            images are resized. If either the height or width are greater than `size["longest_edge"]` then both the height and width are rescaled by `height / ratio`, `width /ratio` where `ratio = max(height / longest_edge, width / longest_edge)`
        patch_size (`Dict[str, int]` *optional*, defaults to `{"height": 16, "width": 16}`):
            Size of the patches in the model, used to calculate the output image size. Can be overridden by `patch_size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        patch_size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"longest_edge": 1024}
        patch_size = patch_size if patch_size is not None else {"height": 16, "width": 16}
        patch_size = get_size_dict(patch_size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.patch_size = patch_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]
        self.do_convert_rgb = do_convert_rgb
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "patch_size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        patch_size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
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
                Dict containing the longest possible edge of the image.
            patch_size (`Dict[str, int]`):
                Patch size used to calculate the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        if "longest_edge" in size:
            size = (size["longest_edge"], size["longest_edge"])
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("size must contain either 'longest_edge' or 'height' and 'width'.")

        if "height" in patch_size and "width" in patch_size:
            patch_size = (patch_size["height"], patch_size["width"])
        else:
            raise ValueError("patch_size must contain either 'shortest_edge' or 'height' and 'width'.")

        output_size = get_resize_output_image_size(
            image,
            size=size,
            patch_size=patch_size,
            input_data_format=input_data_format,
        )
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        patch_size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Describes the maximum input dimensions to the model.
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Patch size in the model. Used to calculate the image after resizing.
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
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
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
        patch_size = patch_size if patch_size is not None else self.patch_size
        patch_size = get_size_dict(patch_size, default_to_square=True)

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

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

        if is_scaled_image(images_list[0][0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images_list[0][0])

        batch_images = []
        batch_image_sizes = []
        for sample_images in images_list:
            images = []
            image_sizes = []
            for image in sample_images:
                if do_resize:
                    image = self.resize(
                        image=image,
                        size=size,
                        patch_size=patch_size,
                        resample=resample,
                        input_data_format=input_data_format,
                    )

                if do_rescale:
                    image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

                if do_normalize:
                    image = self.normalize(
                        image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                    )

                images.append(image)
                image_sizes.append(get_image_size(image, input_data_format))
            batch_images.append(images)
            batch_image_sizes.append(image_sizes)

        images_list = [
            [to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images]
            for images in batch_images
        ]

        # Convert to tensor type outside of BatchFeature to avoid batching the images of different sizes
        images_list = [[convert_to_tensor(image, return_tensors) for image in images] for images in images_list]
        return BatchMixFeature(data={"pixel_values": images_list, "image_sizes": batch_image_sizes}, tensor_type=None)


__all__ = ["PixtralImageProcessor"]

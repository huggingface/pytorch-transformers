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
"""Image processor class for ViT."""

import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ...image_processing_utils import get_size_dict
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
)
from ...utils import TensorType, logging
from ...utils.generic import ExplicitEnum
from ...utils.import_utils import is_torch_available, is_torchvision_available, is_vision_available


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

if is_torchvision_available():
    from torchvision.transforms import Compose, InterpolationMode, Lambda, Normalize, Resize, ToTensor


pil_torch_interpolation_mapping = {
    PILImageResampling.NEAREST: InterpolationMode.NEAREST,
    PILImageResampling.BOX: InterpolationMode.BOX,
    PILImageResampling.BILINEAR: InterpolationMode.BILINEAR,
    PILImageResampling.HAMMING: InterpolationMode.HAMMING,
    PILImageResampling.BICUBIC: InterpolationMode.BICUBIC,
    PILImageResampling.LANCZOS: InterpolationMode.LANCZOS,
    PILImageResampling.NEAREST: InterpolationMode.NEAREST,
}


@dataclass(frozen=True)
class SizeDict:
    height: int = None
    width: int = None
    longest_edge: int = None
    shortest_edge: int = None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in SizeDict.")


class ImageType(ExplicitEnum):
    PIL = "pillow"
    TORCH = "torch"
    NUMPY = "numpy"


def get_image_type(image):
    if is_vision_available() and isinstance(image, Image.Image):
        return ImageType.PIL
    if is_torch_available() and isinstance(image, torch.Tensor):
        return ImageType.TORCH
    if isinstance(image, np.ndarray):
        return ImageType.NUMPY
    raise ValueError(f"Unrecognised image type {type(image)}")


class ViTImageProcessorFast(BaseImageProcessorFast):
    r"""
    Constructs a ViT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]
    _transform_params = [
        "do_resize",
        "do_rescale",
        "do_normalize",
        "size",
        "resample",
        "rescale_factor",
        "image_mean",
        "image_std",
        "image_type",
    ]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self._transform_settings = {}

    def _build_transforms(
        self,
        do_resize: bool,
        size: Dict[str, int],
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,  # dummy
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
        image_type: ImageType,
    ) -> Compose:
        """
        Given the input settings build the image transforms using `torchvision.transforms.Compose`.
        """

        def rescale_image(image, rescale_factor):
            return image * rescale_factor

        transforms = []
        if do_resize:
            transforms.append(
                Resize((size["height"], size["width"]), interpolation=pil_torch_interpolation_mapping[resample])
            )

        # Regardless of whether we rescale, all PIL and numpy values need to be converted to a torch tensor
        # to keep cross compatibility with slow image processors
        convert_to_tensor = image_type in (ImageType.PIL, ImageType.NUMPY)
        if convert_to_tensor:
            transforms.append(ToTensor())

        if do_rescale:
            if convert_to_tensor:
                # ToTensor scales the pixel values to [0, 1] by dividing by the largest value in the image.
                # By default, the rescale factor for the image processor is 1 / 255, i.e. assuming the maximum
                # possible value is 255. Here, if it's different, we need to undo the (assumed) 1/255 scaling
                # and then rescale again
                #
                # NB: This means that the final pixel values will be different in the torchvision transform
                # depending on the pixels in the image as they become [min_val / max_value, max_value / max_value]
                # whereas in the image processors they are [min_value * rescale_factor, max_value * rescale_factor]
                if rescale_factor != 1 / 255:
                    rescale_factor = rescale_factor * 255
                    transforms.append(Lambda(functools.partial(rescale_image, rescale_factor=rescale_factor)))
            else:
                # If do_rescale is `True`, we should still respect it
                transforms.append(Lambda(functools.partial(rescale_image, rescale_factor=rescale_factor)))
        elif convert_to_tensor:
            # If we've converted to a tensor and do_rescale=False, then we need to unscale.
            # As with do_scale=True, we assume that the pixel values were rescaled by 1/255
            rescale_factor = 255
            transforms.append(Lambda(functools.partial(rescale_image, rescale_factor=rescale_factor)))

        if do_normalize:
            transforms.append(Normalize(image_mean, image_std))
        return Compose(transforms)

    @functools.lru_cache(maxsize=1)
    def _validate_input_arguments(
        self,
        return_tensors: Union[str, TensorType],
        do_resize: bool,
        size: Dict[str, int],
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
        data_format: Union[str, ChannelDimension],
        image_type: ImageType,
    ):
        if return_tensors != "pt":
            raise ValueError("Only returning PyTorch tensors is currently supported.")

        if data_format != ChannelDimension.FIRST:
            raise ValueError("Only channel first data format is currently supported.")

        if do_resize and None in (size, resample):
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and None in (image_mean, image_std):
            raise ValueError("Image mean and standard deviation must be specified if do_normalize is True.")

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Only "pt" is supported
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. The following formats are currently supported:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        size = size if size is not None else self.size
        # Make hashable for cache
        size = SizeDict(**size)
        image_mean = tuple(image_mean) if isinstance(image_mean, list) else image_mean
        image_std = tuple(image_std) if isinstance(image_std, list) else image_std

        images = make_list_of_images(images)
        image_type = get_image_type(images[0])

        self._validate_input_arguments(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            return_tensors=return_tensors,
            data_format=data_format,
            image_type=image_type,
        )

        self._maybe_update_transforms(
            do_resize=do_resize,
            do_rescale=do_rescale,
            do_normalize=do_normalize,
            size=size,
            resample=resample,
            rescale_factor=rescale_factor,
            image_mean=image_mean,
            image_std=image_std,
            image_type=image_type,
        )
        transformed_images = [self._transforms(image) for image in images]

        data = {"pixel_values": torch.vstack(transformed_images)}
        return data

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.pop("_transforms", None)
        result.pop("_transform_settings", None)
        return result

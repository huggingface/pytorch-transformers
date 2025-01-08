# coding=utf-8
# Copyright 2023 The Intel AIA Team Authors, and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License=, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing=, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for TVP."""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    PaddingMode,
    flip_channel_order,
    pad,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, filter_out_non_signature_kwargs, is_vision_available, logging


if is_vision_available():
    import PIL


logger = logging.get_logger(__name__)


# Copied from transformers.models.vivit.image_processing_vivit.make_batched
def make_batched(videos) -> List[List[ImageInput]]:
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        return [videos]

    elif is_valid_image(videos):
        return [[videos]]

    raise ValueError(f"Could not make batched video from {videos}")


def get_resize_output_image_size(
    input_image: np.ndarray,
    max_size: int = 448,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    height, width = get_image_size(input_image, input_data_format)
    if height >= width:
        ratio = width * 1.0 / height
        new_height = max_size
        new_width = new_height * ratio
    else:
        ratio = height * 1.0 / width
        new_width = max_size
        new_height = new_width * ratio
    size = (int(new_height), int(new_width))

    return size


class TvpImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Tvp image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"longest_edge": 448}`):
            Size of the output image after resizing. The longest edge of the image will be resized to
            `size["longest_edge"]` while maintaining the aspect ratio of the original image. Can be overriden by
            `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by the `do_center_crop`
            parameter in the `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 448, "width": 448}`):
            Size of the image after applying the center crop. Can be overridden by the `crop_size` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
            in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess` method.
        pad_size (`Dict[str, int]`, *optional*, defaults to `{"height": 448, "width": 448}`):
            Size of the image after applying the padding. Can be overridden by the `pad_size` parameter in the
            `preprocess` method.
        constant_values (`Union[float, Iterable[float]]`, *optional*, defaults to 0):
            The fill value to use when padding the image.
        pad_mode (`PaddingMode`, *optional*, defaults to `PaddingMode.CONSTANT`):
            Use what kind of mode in padding.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        do_flip_channel_order (`bool`, *optional*, defaults to `True`):
            Whether to flip the color channels from RGB to BGR. Can be overridden by the `do_flip_channel_order`
            parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_pad: bool = True,
        pad_size: Dict[str, int] = None,
        constant_values: Union[float, Iterable[float]] = 0,
        pad_mode: PaddingMode = PaddingMode.CONSTANT,
        do_normalize: bool = True,
        do_flip_channel_order: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"longest_edge": 448}
        crop_size = crop_size if crop_size is not None else {"height": 448, "width": 448}
        pad_size = pad_size if pad_size is not None else {"height": 448, "width": 448}

        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.constant_values = constant_values
        self.pad_mode = pad_mode
        self.do_normalize = do_normalize
        self.do_flip_channel_order = do_flip_channel_order
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

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
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image. If `size` is of the form `{"height": h, "width": w}`, the output image will
                have the size `(h, w)`. If `size` is of the form `{"longest_edge": s}`, the output image will have its
                longest edge of length `s` while keeping the aspect ratio of the original image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        size = get_size_dict(size, default_to_square=False)
        if "height" in size and "width" in size:
            output_size = (size["height"], size["width"])
        elif "longest_edge" in size:
            output_size = get_resize_output_image_size(image, size["longest_edge"], input_data_format)
        else:
            raise ValueError(f"Size must have 'height' and 'width' or 'longest_edge' as keys. Got {size.keys()}")

        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def pad_image(
        self,
        image: np.ndarray,
        pad_size: Dict[str, int] = None,
        constant_values: Union[float, Iterable[float]] = 0,
        pad_mode: PaddingMode = PaddingMode.CONSTANT,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Pad an image with zeros to the given size.

        Args:
            image (`np.ndarray`):
                Image to pad.
            pad_size (`Dict[str, int]`)
                Size of the output image with pad.
            constant_values (`Union[float, Iterable[float]]`)
                The fill value to use when padding the image.
            pad_mode (`PaddingMode`)
                The pad mode, default to PaddingMode.CONSTANT
            data_format (`ChannelDimension` or `str`, *optional*)
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        height, width = get_image_size(image, channel_dim=input_data_format)
        max_height = pad_size.get("height", height)
        max_width = pad_size.get("width", width)

        pad_right, pad_bottom = max_width - width, max_height - height
        if pad_right < 0 or pad_bottom < 0:
            raise ValueError("The padding size must be greater than image size")

        padding = ((0, pad_bottom), (0, pad_right))
        padded_image = pad(
            image,
            padding,
            mode=pad_mode,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        return padded_image

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_pad: bool = True,
        pad_size: Dict[str, int] = None,
        constant_values: Union[float, Iterable[float]] = None,
        pad_mode: PaddingMode = None,
        do_normalize: bool = None,
        do_flip_channel_order: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Preprocesses a single image."""

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            size_divisibility=pad_size,  # here the pad() method simply requires the pad_size argument.
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # All transformations expect numpy arrays.
        image = to_numpy_array(image)

        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        if do_center_crop:
            image = self.center_crop(image, size=crop_size, input_data_format=input_data_format)

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            image = self.normalize(
                image=image.astype(np.float32), mean=image_mean, std=image_std, input_data_format=input_data_format
            )

        if do_pad:
            image = self.pad_image(
                image=image,
                pad_size=pad_size,
                constant_values=constant_values,
                pad_mode=pad_mode,
                input_data_format=input_data_format,
            )

        # the pretrained checkpoints assume images are BGR, not RGB
        if do_flip_channel_order:
            image = flip_channel_order(image=image, input_data_format=input_data_format)

        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        return image

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        videos: Union[ImageInput, List[ImageInput], List[List[ImageInput]]],
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_pad: bool = None,
        pad_size: Dict[str, int] = None,
        constant_values: Union[float, Iterable[float]] = None,
        pad_mode: PaddingMode = None,
        do_normalize: bool = None,
        do_flip_channel_order: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            videos (`ImageInput` or `List[ImageInput]` or `List[List[ImageInput]]`):
                Frames to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after applying resize.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_centre_crop`):
                Whether to centre crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the image after applying the centre crop.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_pad (`bool`, *optional*, defaults to `True`):
                Whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess` method.
            pad_size (`Dict[str, int]`, *optional*, defaults to `{"height": 448, "width": 448}`):
                Size of the image after applying the padding. Can be overridden by the `pad_size` parameter in the
                `preprocess` method.
            constant_values (`Union[float, Iterable[float]]`, *optional*, defaults to 0):
                The fill value to use when padding the image.
            pad_mode (`PaddingMode`, *optional*, defaults to "PaddingMode.CONSTANT"):
                Use what kind of mode in padding.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            do_flip_channel_order (`bool`, *optional*, defaults to `self.do_flip_channel_order`):
                Whether to flip the channel order of the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
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
                    - Unset: Use the inferred channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_pad = do_pad if do_pad is not None else self.do_pad
        pad_size = pad_size if pad_size is not None else self.pad_size
        constant_values = constant_values if constant_values is not None else self.constant_values
        pad_mode = pad_mode if pad_mode else self.pad_mode
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_flip_channel_order = (
            do_flip_channel_order if do_flip_channel_order is not None else self.do_flip_channel_order
        )
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        if not valid_images(videos):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        videos = make_batched(videos)

        videos = [
            np.array(
                [
                    self._preprocess_image(
                        image=img,
                        do_resize=do_resize,
                        size=size,
                        resample=resample,
                        do_center_crop=do_center_crop,
                        crop_size=crop_size,
                        do_rescale=do_rescale,
                        rescale_factor=rescale_factor,
                        do_pad=do_pad,
                        pad_size=pad_size,
                        constant_values=constant_values,
                        pad_mode=pad_mode,
                        do_normalize=do_normalize,
                        do_flip_channel_order=do_flip_channel_order,
                        image_mean=image_mean,
                        image_std=image_std,
                        data_format=data_format,
                        input_data_format=input_data_format,
                    )
                    for img in video
                ]
            )
            for video in videos
        ]

        data = {"pixel_values": videos}
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["TvpImageProcessor"]

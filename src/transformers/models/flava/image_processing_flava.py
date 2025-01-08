# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Flava."""

import math
import random
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, filter_out_non_signature_kwargs, is_vision_available, logging


if is_vision_available():
    import PIL


logger = logging.get_logger(__name__)


# These values are taken from CLIP
FLAVA_IMAGE_MEAN = OPENAI_CLIP_MEAN
FLAVA_IMAGE_STD = OPENAI_CLIP_STD
FLAVA_CODEBOOK_MEAN = [0.0, 0.0, 0.0]
FLAVA_CODEBOOK_STD = [1.0, 1.0, 1.0]
LOGIT_LAPLACE_EPS: float = 0.1


# Inspired from https://github.com/microsoft/unilm/blob/master/beit/masking_generator.py
class FlavaMaskingGenerator:
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 14,
        total_mask_patches: int = 75,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_patches: int = 16,
        mask_group_min_aspect_ratio: Optional[float] = 0.3,
        mask_group_max_aspect_ratio: float = None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.total_mask_patches = total_mask_patches

        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_max_patches = total_mask_patches if mask_group_max_patches is None else mask_group_max_patches

        mask_group_max_aspect_ratio = mask_group_max_aspect_ratio or 1 / mask_group_min_aspect_ratio
        self.log_aspect_ratio = (math.log(mask_group_min_aspect_ratio), math.log(mask_group_max_aspect_ratio))

    def __repr__(self):
        repr_str = "MaskingGenerator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.mask_group_min_patches,
            self.mask_group_max_patches,
            self.total_mask_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _attempt in range(10):
            target_area = random.uniform(self.mask_group_min_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))
            if width < self.width and height < self.height:
                top = random.randint(0, self.height - height)
                left = random.randint(0, self.width - width)

                num_masked = mask[top : top + height, left : left + width].sum()
                # Overlap
                if 0 < height * width - num_masked <= max_mask_patches:
                    for i in range(top, top + height):
                        for j in range(left, left + width):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=int)
        mask_count = 0
        while mask_count < self.total_mask_patches:
            max_mask_patches = self.total_mask_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.mask_group_max_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


class FlavaImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Flava image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by the `size` parameter in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in
            `preprocess`.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the images. Can be overridden by the `do_center_crop` parameter in `preprocess`.
        crop_size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of image after the center crop `(crop_size["height"], crop_size["width"])`. Can be overridden by the
            `crop_size` parameter in `preprocess`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in `preprocess`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in
            `preprocess`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in `preprocess`.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        return_image_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the image mask. Can be overridden by the `return_image_mask` parameter in `preprocess`.
        input_size_patches (`int`, *optional*, defaults to 14):
            Number of patches in the image in height and width direction. 14x14 = 196 total patches. Can be overridden
            by the `input_size_patches` parameter in `preprocess`.
        total_mask_patches (`int`, *optional*, defaults to 75):
            Total number of patches that should be masked. Can be overridden by the `total_mask_patches` parameter in
            `preprocess`.
        mask_group_min_patches (`int`, *optional*, defaults to 16):
            Minimum number of patches that should be masked. Can be overridden by the `mask_group_min_patches`
            parameter in `preprocess`.
        mask_group_max_patches (`int`, *optional*):
            Maximum number of patches that should be masked. Can be overridden by the `mask_group_max_patches`
            parameter in `preprocess`.
        mask_group_min_aspect_ratio (`float`, *optional*, defaults to 0.3):
            Minimum aspect ratio of the mask window. Can be overridden by the `mask_group_min_aspect_ratio` parameter
            in `preprocess`.
        mask_group_max_aspect_ratio (`float`, *optional*):
            Maximum aspect ratio of the mask window. Can be overridden by the `mask_group_max_aspect_ratio` parameter
            in `preprocess`.
        codebook_do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input for codebook to a certain. Can be overridden by the `codebook_do_resize`
            parameter in `preprocess`. `codebook_size`.
        codebook_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Resize the input for codebook to the given size. Can be overridden by the `codebook_size` parameter in
            `preprocess`.
        codebook_resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.LANCZOS`):
            Resampling filter to use if resizing the codebook image. Can be overridden by the `codebook_resample`
            parameter in `preprocess`.
        codebook_do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input for codebook at the center. If the input size is smaller than
            `codebook_crop_size` along any edge, the image is padded with 0's and then center cropped. Can be
            overridden by the `codebook_do_center_crop` parameter in `preprocess`.
        codebook_crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Desired output size for codebook input when applying center-cropping. Can be overridden by the
            `codebook_crop_size` parameter in `preprocess`.
        codebook_do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input for codebook by the specified scale `codebook_rescale_factor`. Can be
            overridden by the `codebook_do_rescale` parameter in `preprocess`.
        codebook_rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Defines the scale factor to use if rescaling the codebook image. Can be overridden by the
            `codebook_rescale_factor` parameter in `preprocess`.
        codebook_do_map_pixels (`bool`, *optional*, defaults to `True`):
            Whether to map the pixel values of the codebook input to (1 - 2e)x + e. Can be overridden by the
            `codebook_do_map_pixels` parameter in `preprocess`.
        codebook_do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input for codebook with `codebook_image_mean` and `codebook_image_std`. Can
            be overridden by the `codebook_do_normalize` parameter in `preprocess`.
        codebook_image_mean (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0, 0, 0]`):
            The sequence of means for each channel, to be used when normalizing images for codebook. Can be overridden
            by the `codebook_image_mean` parameter in `preprocess`.
        codebook_image_std (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of standard deviations for each channel, to be used when normalizing images for codebook. Can
            be overridden by the `codebook_image_std` parameter in `preprocess`.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, Iterable[float]]] = None,
        image_std: Optional[Union[float, Iterable[float]]] = None,
        # Mask related params
        return_image_mask: bool = False,
        input_size_patches: int = 14,
        total_mask_patches: int = 75,
        mask_group_min_patches: int = 16,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_aspect_ratio: float = 0.3,
        mask_group_max_aspect_ratio: Optional[float] = None,
        # Codebook related params
        return_codebook_pixels: bool = False,
        codebook_do_resize: bool = True,
        codebook_size: bool = None,
        codebook_resample: int = PILImageResampling.LANCZOS,
        codebook_do_center_crop: bool = True,
        codebook_crop_size: int = None,
        codebook_do_rescale: bool = True,
        codebook_rescale_factor: Union[int, float] = 1 / 255,
        codebook_do_map_pixels: bool = True,
        codebook_do_normalize: bool = True,
        codebook_image_mean: Optional[Union[float, Iterable[float]]] = None,
        codebook_image_std: Optional[Union[float, Iterable[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size)
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        codebook_size = codebook_size if codebook_size is not None else {"height": 112, "width": 112}
        codebook_size = get_size_dict(codebook_size, param_name="codebook_size")
        codebook_crop_size = codebook_crop_size if codebook_crop_size is not None else {"height": 112, "width": 112}
        codebook_crop_size = get_size_dict(codebook_crop_size, param_name="codebook_crop_size")

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else FLAVA_IMAGE_MEAN
        self.image_std = image_std if image_std is not None else FLAVA_IMAGE_STD

        self.return_image_mask = return_image_mask
        self.input_size_patches = input_size_patches
        self.total_mask_patches = total_mask_patches
        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_max_patches = mask_group_max_patches
        self.mask_group_min_aspect_ratio = mask_group_min_aspect_ratio
        self.mask_group_max_aspect_ratio = mask_group_max_aspect_ratio

        self.return_codebook_pixels = return_codebook_pixels
        self.codebook_do_resize = codebook_do_resize
        self.codebook_size = codebook_size
        self.codebook_resample = codebook_resample
        self.codebook_do_center_crop = codebook_do_center_crop
        self.codebook_crop_size = codebook_crop_size
        self.codebook_do_rescale = codebook_do_rescale
        self.codebook_rescale_factor = codebook_rescale_factor
        self.codebook_do_map_pixels = codebook_do_map_pixels
        self.codebook_do_normalize = codebook_do_normalize
        self.codebook_image_mean = codebook_image_mean
        self.codebook_image_mean = codebook_image_mean if codebook_image_mean is not None else FLAVA_CODEBOOK_MEAN
        self.codebook_image_std = codebook_image_std if codebook_image_std is not None else FLAVA_CODEBOOK_STD

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `FlavaImageProcessor.from_pretrained(checkpoint, codebook_size=600)`
        """
        image_processor_dict = image_processor_dict.copy()
        if "codebook_size" in kwargs:
            image_processor_dict["codebook_size"] = kwargs.pop("codebook_size")
        if "codebook_crop_size" in kwargs:
            image_processor_dict["codebook_crop_size"] = kwargs.pop("codebook_crop_size")
        return super().from_dict(image_processor_dict, **kwargs)

    @lru_cache()
    def masking_generator(
        self,
        input_size_patches,
        total_mask_patches,
        mask_group_min_patches,
        mask_group_max_patches,
        mask_group_min_aspect_ratio,
        mask_group_max_aspect_ratio,
    ) -> FlavaMaskingGenerator:
        return FlavaMaskingGenerator(
            input_size=input_size_patches,
            total_mask_patches=total_mask_patches,
            mask_group_min_patches=mask_group_min_patches,
            mask_group_max_patches=mask_group_max_patches,
            mask_group_min_aspect_ratio=mask_group_min_aspect_ratio,
            mask_group_max_aspect_ratio=mask_group_max_aspect_ratio,
        )

    # Copied from transformers.models.vit.image_processing_vit.ViTImageProcessor.resize with PILImageResampling.BILINEAR->PILImageResampling.BICUBIC
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
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def map_pixels(self, image: np.ndarray) -> np.ndarray:
        return (1 - 2 * LOGIT_LAPLACE_EPS) * image + LOGIT_LAPLACE_EPS

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
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_map_pixels: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[ChannelDimension] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # All transformations expect numpy arrays.
        image = to_numpy_array(image)

        if do_rescale and is_scaled_image(image):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(image)

        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        if do_center_crop:
            image = self.center_crop(image=image, size=crop_size, input_data_format=input_data_format)

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        if do_map_pixels:
            image = self.map_pixels(image)

        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        # Mask related params
        return_image_mask: Optional[bool] = None,
        input_size_patches: Optional[int] = None,
        total_mask_patches: Optional[int] = None,
        mask_group_min_patches: Optional[int] = None,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_aspect_ratio: Optional[float] = None,
        mask_group_max_aspect_ratio: Optional[float] = None,
        # Codebook related params
        return_codebook_pixels: Optional[bool] = None,
        codebook_do_resize: Optional[bool] = None,
        codebook_size: Optional[Dict[str, int]] = None,
        codebook_resample: Optional[int] = None,
        codebook_do_center_crop: Optional[bool] = None,
        codebook_crop_size: Optional[Dict[str, int]] = None,
        codebook_do_rescale: Optional[bool] = None,
        codebook_rescale_factor: Optional[float] = None,
        codebook_do_map_pixels: Optional[bool] = None,
        codebook_do_normalize: Optional[bool] = None,
        codebook_image_mean: Optional[Iterable[float]] = None,
        codebook_image_std: Optional[Iterable[float]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
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
                Size of the image.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            return_image_mask (`bool`, *optional*, defaults to `self.return_image_mask`):
                Whether to return the image mask.
            input_size_patches (`int`, *optional*, defaults to `self.input_size_patches`):
                Size of the patches to extract from the image.
            total_mask_patches (`int`, *optional*, defaults to `self.total_mask_patches`):
                Total number of patches to extract from the image.
            mask_group_min_patches (`int`, *optional*, defaults to `self.mask_group_min_patches`):
                Minimum number of patches to extract from the image.
            mask_group_max_patches (`int`, *optional*, defaults to `self.mask_group_max_patches`):
                Maximum number of patches to extract from the image.
            mask_group_min_aspect_ratio (`float`, *optional*, defaults to `self.mask_group_min_aspect_ratio`):
                Minimum aspect ratio of the patches to extract from the image.
            mask_group_max_aspect_ratio (`float`, *optional*, defaults to `self.mask_group_max_aspect_ratio`):
                Maximum aspect ratio of the patches to extract from the image.
            return_codebook_pixels (`bool`, *optional*, defaults to `self.return_codebook_pixels`):
                Whether to return the codebook pixels.
            codebook_do_resize (`bool`, *optional*, defaults to `self.codebook_do_resize`):
                Whether to resize the codebook pixels.
            codebook_size (`Dict[str, int]`, *optional*, defaults to `self.codebook_size`):
                Size of the codebook pixels.
            codebook_resample (`int`, *optional*, defaults to `self.codebook_resample`):
                Resampling filter to use if resizing the codebook pixels. This can be one of the enum
                `PILImageResampling`, Only has an effect if `codebook_do_resize` is set to `True`.
            codebook_do_center_crop (`bool`, *optional*, defaults to `self.codebook_do_center_crop`):
                Whether to center crop the codebook pixels.
            codebook_crop_size (`Dict[str, int]`, *optional*, defaults to `self.codebook_crop_size`):
                Size of the center crop of the codebook pixels. Only has an effect if `codebook_do_center_crop` is set
                to `True`.
            codebook_do_rescale (`bool`, *optional*, defaults to `self.codebook_do_rescale`):
                Whether to rescale the codebook pixels values between [0 - 1].
            codebook_rescale_factor (`float`, *optional*, defaults to `self.codebook_rescale_factor`):
                Rescale factor to rescale the codebook pixels by if `codebook_do_rescale` is set to `True`.
            codebook_do_map_pixels (`bool`, *optional*, defaults to `self.codebook_do_map_pixels`):
                Whether to map the codebook pixels values.
            codebook_do_normalize (`bool`, *optional*, defaults to `self.codebook_do_normalize`):
                Whether to normalize the codebook pixels.
            codebook_image_mean (`float` or `List[float]`, *optional*, defaults to `self.codebook_image_mean`):
                Codebook pixels mean to normalize the codebook pixels by if `codebook_do_normalize` is set to `True`.
            codebook_image_std (`float` or `List[float]`, *optional*, defaults to `self.codebook_image_std`):
                Codebook pixels standard deviation to normalize the codebook pixels by if `codebook_do_normalize` is
                set to `True`.
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
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size)
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size")
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        return_image_mask = return_image_mask if return_image_mask is not None else self.return_image_mask
        input_size_patches = input_size_patches if input_size_patches is not None else self.input_size_patches
        total_mask_patches = total_mask_patches if total_mask_patches is not None else self.total_mask_patches
        mask_group_min_patches = (
            mask_group_min_patches if mask_group_min_patches is not None else self.mask_group_min_patches
        )
        mask_group_max_patches = (
            mask_group_max_patches if mask_group_max_patches is not None else self.mask_group_max_patches
        )
        mask_group_min_aspect_ratio = (
            mask_group_min_aspect_ratio
            if mask_group_min_aspect_ratio is not None
            else self.mask_group_min_aspect_ratio
        )
        mask_group_max_aspect_ratio = (
            mask_group_max_aspect_ratio
            if mask_group_max_aspect_ratio is not None
            else self.mask_group_max_aspect_ratio
        )

        return_codebook_pixels = (
            return_codebook_pixels if return_codebook_pixels is not None else self.return_codebook_pixels
        )
        codebook_do_resize = codebook_do_resize if codebook_do_resize is not None else self.codebook_do_resize
        codebook_size = codebook_size if codebook_size is not None else self.codebook_size
        codebook_size = get_size_dict(codebook_size, param_name="codebook_size")
        codebook_resample = codebook_resample if codebook_resample is not None else self.codebook_resample
        codebook_do_rescale = codebook_do_rescale if codebook_do_rescale is not None else self.codebook_do_rescale
        codebook_rescale_factor = (
            codebook_rescale_factor if codebook_rescale_factor is not None else self.codebook_rescale_factor
        )
        codebook_do_center_crop = (
            codebook_do_center_crop if codebook_do_center_crop is not None else self.codebook_do_center_crop
        )
        codebook_crop_size = codebook_crop_size if codebook_crop_size is not None else self.codebook_crop_size
        codebook_crop_size = get_size_dict(codebook_crop_size, param_name="codebook_crop_size")
        codebook_do_map_pixels = (
            codebook_do_map_pixels if codebook_do_map_pixels is not None else self.codebook_do_map_pixels
        )
        codebook_do_normalize = (
            codebook_do_normalize if codebook_do_normalize is not None else self.codebook_do_normalize
        )
        codebook_image_mean = codebook_image_mean if codebook_image_mean is not None else self.codebook_image_mean
        codebook_image_std = codebook_image_std if codebook_image_std is not None else self.codebook_image_std

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        processed_images = [
            self._preprocess_image(
                image=img,
                do_resize=do_resize,
                size=size,
                resample=resample,
                do_center_crop=do_center_crop,
                crop_size=crop_size,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                do_map_pixels=False,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for img in images
        ]
        data = {"pixel_values": processed_images}

        if return_codebook_pixels:
            codebook_images = [
                self._preprocess_image(
                    image=img,
                    do_resize=codebook_do_resize,
                    size=codebook_size,
                    resample=codebook_resample,
                    do_center_crop=codebook_do_center_crop,
                    crop_size=codebook_crop_size,
                    do_rescale=codebook_do_rescale,
                    rescale_factor=codebook_rescale_factor,
                    do_normalize=codebook_do_normalize,
                    image_mean=codebook_image_mean,
                    image_std=codebook_image_std,
                    do_map_pixels=codebook_do_map_pixels,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for img in images
            ]
            data["codebook_pixel_values"] = codebook_images

        if return_image_mask:
            mask_generator = self.masking_generator(
                input_size_patches=input_size_patches,
                total_mask_patches=total_mask_patches,
                mask_group_min_patches=mask_group_min_patches,
                mask_group_max_patches=mask_group_max_patches,
                mask_group_min_aspect_ratio=mask_group_min_aspect_ratio,
                mask_group_max_aspect_ratio=mask_group_max_aspect_ratio,
            )
            masks = [mask_generator() for _ in images]
            data["bool_masked_pos"] = masks

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["FlavaImageProcessor"]

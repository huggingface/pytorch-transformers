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
"""Fast Image processor class for LLaVa-Onevision."""

import functools
from typing import Dict, List, Optional, Union

from ...image_processing_utils import BatchFeature, get_patch_output_size, get_size_dict, select_best_resolution
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    SizeDict,
    divide_to_patches,
)
from ...image_transforms import CenterCrop, GroupByShape, Normalize, ReorderImages, Rescale, Resize
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    make_batched_images,
    validate_kwargs,
)
from ...utils import (
    TensorType,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
)


if is_torch_available():
    import torch
    from torch.nn import Sequential

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


class LlavaOnevisionImageProcessorFast(BaseImageProcessorFast):
    r"""
    Constructs a fast LLaVa-Onevision image processor. Based on [`SiglipImageProcessor`] with incorporation of processing each video frame.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        image_grid_pinpoints (`List` *optional*, defaults to `[[672, 336], [336, 672], [672, 672], [336, 1008], [1008, 336]]`):
            A list of possible resolutions to use for processing high resolution images. The best resolution is selected
            based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
            method. Not used for processinf videos.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
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
        do_pad (`bool`, *optional*, defaults to `True`):
                Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    # To be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    default_to_square = False
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    image_grid_pinpoints = [
        [384, 384],
        [384, 768],
        [384, 1152],
        [384, 1536],
        [384, 1920],
        [384, 2304],
        [768, 384],
        [768, 768],
        [768, 1152],
        [768, 1536],
        [768, 1920],
        [768, 2304],
        [1152, 384],
        [1152, 768],
        [1152, 1152],
        [1152, 1536],
        [1152, 1920],
        [1152, 2304],
        [1536, 384],
        [1536, 768],
        [1536, 1152],
        [1536, 1536],
        [1536, 1920],
        [1536, 2304],
        [1920, 384],
        [1920, 768],
        [1920, 1152],
        [1920, 1536],
        [1920, 1920],
        [1920, 2304],
        [2304, 384],
        [2304, 768],
        [2304, 1152],
        [2304, 1536],
        [2304, 1920],
        [2304, 2304],
    ]

    # Copied from transformers.models.llava_next.image_processing_llava_next_fast.LlavaNextImageProcessorFast.__init__
    def __init__(
        self,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        image_grid_pinpoints: List = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        do_convert_rgb: bool = None,
        **kwargs,
    ) -> None:
        super().__init__(
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
            do_convert_rgb=do_convert_rgb,
            **kwargs,
        )
        self.image_grid_pinpoints = (
            image_grid_pinpoints if image_grid_pinpoints is not None else self.image_grid_pinpoints
        )
        self.do_pad = do_pad if do_pad is not None else self.do_pad

    # Copied from transformers.models.llava_next.image_processing_llava_next_fast.LlavaNextImageProcessorFast._prepare_images_structure
    def _prepare_images_structure(
        self,
        images: ImageInput,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        return make_batched_images(images)

    # Copied from transformers.models.llava_next.image_processing_llava_next_fast.LlavaNextImageProcessorFast._resize_for_patching
    def _resize_for_patching(
        self,
        image: "torch.Tensor",
        target_resolution: tuple,
        interpolation: "F.InterpolationMode",
        input_data_format: ChannelDimension,
    ) -> "torch.Tensor":
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image ("torch.Tensor"):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            interpolation (`InterpolationMode`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            "torch.Tensor": The resized and padded image.
        """
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)

        # Resize the image
        resized_image = F.resize(image, (new_height, new_width), interpolation=interpolation)

        return resized_image

    # Copied from transformers.models.llava_next.image_processing_llava_next_fast.LlavaNextImageProcessorFast._pad_for_patching
    def _pad_for_patching(
        self, image: "torch.Tensor", target_resolution: tuple, input_data_format: ChannelDimension
    ) -> "torch.Tensor":
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        new_height, new_width = get_patch_output_size(image, target_resolution, input_data_format)

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = F.pad(image, padding=[paste_x, paste_y, paste_x, paste_y])

        return padded_image

    # Copied from transformers.models.llava_next.image_processing_llava_next_fast.LlavaNextImageProcessorFast._get_image_patches
    def _get_image_patches(
        self,
        image: "torch.Tensor",
        grid_pinpoints,
        size: tuple,
        patch_size: int,
        interpolation: "F.InterpolationMode",
    ) -> List["torch.Tensor"]:
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image ("torch.Tensor"):
                The input image to be processed.
            grid_pinpoints (List):
                A string representation of a list of possible resolutions.
            size (`tuple`):
                Size to resize the original image to.
            patch_size (`int`):
                Size of the patches to divide the image into.
            interpolation (`"InterpolationMode"`):
                Resampling filter to use if resizing the image.

        Returns:
            List["torch.Tensor"]: A list of NumPy arrays containing the processed image patches.
        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, interpolation=interpolation, input_data_format=ChannelDimension.FIRST
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=ChannelDimension.FIRST)
        patches = divide_to_patches(padded_image, patch_size=patch_size)
        resized_original_image = F.resize(image, size=size, interpolation=interpolation)

        image_patches = [resized_original_image] + patches

        return image_patches

    # Copied from transformers.models.llava_next.image_processing_llava_next_fast.LlavaNextImageProcessorFast._pad_for_batching
    def _pad_for_batching(
        self,
        pixel_values: List["torch.Tensor"],
    ) -> List["torch.Tensor"]:
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`List[torch.Tensor]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)

        Returns:
            List[`torch.Tensor`]: The padded images.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            torch.nn.functional.pad(image, pad=[0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[0]])
            for image in pixel_values
        ]

        return pixel_values

    def _build_transforms(
        self,
        do_resize: bool,
        size: SizeDict,
        interpolation: "F.InterpolationMode",
        do_center_crop: bool,
        crop_size: int,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
    ) -> "Sequential":
        """
        Given the input settings build the image transforms using a `Sequential` module.
        """
        transforms = []

        transforms.append(GroupByShape())
        if do_resize:
            transforms.append(Resize(size, interpolation=interpolation))
            # Since the size was changed, we need to group the images by shape again
            transforms.append(ReorderImages())
            transforms.append(GroupByShape())
        if do_center_crop:
            transforms.append(CenterCrop(crop_size))
            # Since the size was changed, we need to group the images by shape again
            transforms.append(ReorderImages())
            transforms.append(GroupByShape())
        # We can combine rescale and normalize into a single operation for speed
        if do_rescale and do_normalize:
            # image_mean and image_std have already been adjusted for rescaling
            transforms.append(Normalize(image_mean, image_std))
        elif do_rescale:
            transforms.append(Rescale(rescale_factor=rescale_factor))
        elif do_normalize:
            transforms.append(Normalize(image_mean, image_std))

        if isinstance(transforms[-1], GroupByShape):
            # No added transforms, so we can remove the last GroupByShape
            transforms.pop()
        else:
            # We necessarily have grouped images, so we need to reorder them back to the original order
            transforms.append(ReorderImages())

        return Sequential(*transforms)

    @functools.lru_cache(maxsize=1)
    def get_transforms(self, **kwargs) -> "Sequential":
        return self._build_transforms(**kwargs)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        image_grid_pinpoints: List = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
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
                Describes the maximum input dimensions to the model.
            resample (`PILImageResampling` or `InterpolationMode`, *optional*, defaults to self.resample):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the output image after applying `center_crop`.
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
                The type of tensors to return. Default to `"pt"` for PyTorch tensors if unset.
                Fast image processors only support PyTorch tensors.
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
            device (`torch.device`, *optional*):
                The device to process the images on. If unset, the device is inferred from the input images.
        """
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_extra_kwargs)

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        default_to_square = kwargs.pop(
            "default_to_square", self.default_to_square if self.default_to_square is not None else True
        )
        size = get_size_dict(size=size, default_to_square=default_to_square) if size is not None else None
        image_grid_pinpoints = image_grid_pinpoints if image_grid_pinpoints is not None else self.image_grid_pinpoints
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size") if crop_size is not None else None
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = self._prepare_input_images(
            images=images,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )

        image_mean, image_std, size, crop_size, interpolation = self._prepare_process_arguments(
            do_resize=do_resize,
            size=size,
            resample=resample,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            return_tensors=return_tensors,
            data_format=data_format,
            device=images[0].device,
            **kwargs,
        )

        patches_transforms = self.get_transforms(
            do_resize=do_resize,
            size=size,
            interpolation=interpolation,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )

        processed_images = []
        image_sizes = []
        for image in images:
            size_tuple = (
                (size.height, size.width) if size.height and size.width else (size.shortest_edge, size.shortest_edge)
            )
            patch_size = (
                crop_size.height
                if crop_size is not None and crop_size.height
                else size.height
                if size.height
                else size.shortest_edge
            )
            image_patches = self._get_image_patches(
                image,
                image_grid_pinpoints,
                size=size_tuple,
                patch_size=patch_size,
                interpolation=interpolation,
            )

            # apply torchvision transforms to patches
            processed_image_patches = patches_transforms(image_patches)

            processed_image_patches = (
                torch.stack(processed_image_patches, dim=0) if return_tensors else processed_image_patches
            )
            processed_images.append(processed_image_patches)
            image_sizes.append(get_image_size(image, input_data_format))

        if do_pad:
            processed_images = self._pad_for_batching(processed_images)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes}, tensor_type=return_tensors
        )


__all__ = ["LlavaOnevisionImageProcessorFast"]

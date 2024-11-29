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

from typing import Dict, List, Optional, Union

from ...image_processing_utils import get_size_dict
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    ImageType,
    PILImageResampling,
    get_image_size,
    get_image_type,
    infer_channel_dimension_format,
    validate_fast_preprocess_arguments,
    validate_kwargs,
)
from ...utils import (
    TensorType,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)
from .image_processing_pixtral import (
    BatchMixFeature,
    convert_to_rgb,
    get_resize_output_image_size,
    make_list_of_images,
)


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torchvision_available():
    if is_vision_available():
        from ...image_utils import pil_torch_interpolation_mapping

    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


class PixtralImageProcessorFast(BaseImageProcessorFast):
    r"""
    Constructs a fast Pixtral image processor that leverages torchvision.

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
        resample: Union[PILImageResampling, "F.InterpolationMode"] = PILImageResampling.BICUBIC,
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
        image: torch.Tensor,
        size: Dict[str, int],
        patch_size: Dict[str, int],
        interpolation: "F.InterpolationMode" = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`Dict[str, int]`):
                Dict containing the longest possible edge of the image.
            patch_size (`Dict[str, int]`):
                Patch size used to calculate the size of the output image.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                Resampling filter to use when resiizing the image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
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
        )
        return F.resize(
            image,
            size=output_size,
            interpolation=interpolation,
            **kwargs,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        patch_size: Dict[str, int] = None,
        resample: Optional[Union[PILImageResampling, "F.InterpolationMode"]] = None,
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
    ) -> BatchMixFeature:
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
            resample (`PILImageResampling` or `InterpolationMode`, *optional*, defaults to self.resample):
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
        device = kwargs.pop("device", None)

        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

        images_list = make_list_of_images(images)
        image_type = get_image_type(images_list[0][0])

        if image_type not in [ImageType.PIL, ImageType.TORCH, ImageType.NUMPY]:
            raise ValueError(f"Unsupported input image type {image_type}")

        validate_fast_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
            return_tensors=return_tensors,
            data_format=data_format,
        )

        if do_convert_rgb:
            images_list = [[convert_to_rgb(image) for image in images] for images in images_list]

        if image_type == ImageType.PIL:
            images_list = [[F.pil_to_tensor(image) for image in images] for images in images_list]
        elif image_type == ImageType.NUMPY:
            # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
            images_list = [[torch.from_numpy(image).contiguous() for image in images] for images in images_list]

        if device is not None:
            images_list = [[image.to(device) for image in images] for images in images_list]

        # We assume that all images have the same channel dimension format.
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images_list[0][0])
        if input_data_format == ChannelDimension.LAST:
            images_list = [[image.permute(2, 0, 1).contiguous() for image in images] for images in images_list]
            input_data_format = ChannelDimension.FIRST

        if do_rescale and do_normalize:
            # fused rescale and normalize
            new_mean = torch.tensor(image_mean, device=images_list[0][0].device) * (1.0 / rescale_factor)
            new_std = torch.tensor(image_std, device=images_list[0][0].device) * (1.0 / rescale_factor)

        batch_images = []
        batch_image_sizes = []
        for sample_images in images_list:
            images = []
            image_sizes = []
            for image in sample_images:
                if do_resize:
                    interpolation = (
                        pil_torch_interpolation_mapping[resample]
                        if isinstance(resample, (PILImageResampling, int))
                        else resample
                    )
                    image = self.resize(
                        image=image,
                        size=size,
                        patch_size=patch_size,
                        interpolation=interpolation,
                    )

                if do_rescale and do_normalize:
                    # fused rescale and normalize
                    image = F.normalize(image.to(dtype=torch.float32), new_mean, new_std)
                elif do_rescale:
                    image = image * rescale_factor
                elif do_normalize:
                    image = F.normalize(image, image_mean, image_std)

                images.append(image)
                image_sizes.append(get_image_size(image, input_data_format))
            batch_images.append(images)
            batch_image_sizes.append(image_sizes)

        return BatchMixFeature(data={"pixel_values": batch_images, "image_sizes": batch_image_sizes}, tensor_type=None)

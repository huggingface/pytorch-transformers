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
"""Image processor class for SegGPT."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available, is_vision_available, logging, requires_backends


if is_torch_available():
    import torch

if is_vision_available():
    pass


logger = logging.get_logger(__name__)


# See https://arxiv.org/pdf/2212.02499.pdf  at 3.1 Redefining Output Spaces as "Images" - Semantic Segmentation from PAINTER paper
# Taken from https://github.com/Abdullah-Meda/Painter/blob/main/Painter/data/coco_semseg/gen_color_coco_panoptic_segm.py#L31
def build_palette(num_labels: int) -> List[Tuple[int, int]]:
    base = int(num_labels ** (1 / 3)) + 1
    margin = 256 // base

    # we assume that class_idx 0 is the background which is mapped to black
    color_list = [(0, 0, 0)]
    for location in range(num_labels):
        num_seq_r = location // base**2
        num_seq_g = (location % base**2) // base
        num_seq_b = location % base

        R = 255 - num_seq_r * margin
        G = 255 - num_seq_g * margin
        B = 255 - num_seq_b * margin

        color_list.append((R, G, B))

    return color_list


def mask_to_rgb(
    mask: np.ndarray, palette: Optional[List[Tuple[int, int]]] = None, data_format: Optional[ChannelDimension] = None
) -> np.ndarray:
    data_format = data_format if data_format is not None else ChannelDimension.FIRST

    if palette is not None:
        height, width = mask.shape

        rgb_mask = np.zeros((3, height, width), dtype=np.uint8)

        classes_in_mask = np.unique(mask)

        for class_idx in classes_in_mask:
            rgb_value = palette[class_idx]
            class_mask = (mask == class_idx).astype(np.uint8)
            class_mask = np.expand_dims(class_mask, axis=-1)
            class_rgb_mask = class_mask * np.array(rgb_value)
            class_rgb_mask = np.moveaxis(class_rgb_mask, -1, 0)
            rgb_mask += class_rgb_mask.astype(np.uint8)

        rgb_mask = np.clip(rgb_mask, 0, 255).astype(np.uint8)

    else:
        rgb_mask = np.repeat(mask[None, ...], 3, axis=0)

    return to_channel_dimension_format(rgb_mask, data_format)


class SegGptImageProcessor(BaseImageProcessor):
    r"""
    Constructs a SegGpt image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 448, "width": 448}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
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
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the prompt mask to RGB format. Can be overridden by the `do_convert_rgb` parameter in the
            `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
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
        size = size if size is not None else {"height": 448, "width": 448}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_convert_rgb = do_convert_rgb

    def get_palette(self, num_labels: int) -> List[Tuple[int, int]]:
        """Build a palette to map the prompt mask from a single channel to a 3 channel RGB.

        Args:
            num_labels (`int`):
                Number of classes in the segmentation task (excluding the background).

        Returns:
            `List[Tuple[int, int]]`: Palette to map the prompt mask from a single channel to a 3 channel RGB.
        """
        return build_palette(num_labels)

    def mask_to_rgb(
        self,
        image: np.ndarray,
        palette: Optional[List[Tuple[int, int]]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Converts a segmentation map to RGB format.

        Args:
            image (`np.ndarray`):
                Segmentation map with dimensions (height, width) where pixel values represent the class index.
            palette (`List[Tuple[int, int]]`, *optional*, defaults to `None`):
                Palette to use to convert the mask to RGB format. If unset, the mask is duplicated across the channel
                dimension.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The mask in RGB format.
        """
        return mask_to_rgb(image, palette=palette, data_format=data_format)

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

    def _preprocess_step(
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
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        do_convert_rgb: Optional[bool] = None,
        num_labels: Optional[int] = None,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to _preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BICUBIC`. Only has
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
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the prompt mask to RGB format. If `num_labels` is specified, a palette will be built
                to map the prompt mask from a single channel to a 3 channel RGB. If unset, the prompt mask is duplicated
                across the channel dimension. Must be set to `False` if the prompt mask is already in RGB format.
            num_labels: (`int`, *optional*):
                Number of classes in the segmentation task (excluding the background). If specified, a palette will be
                built, assuming that class_idx 0 is the background, to map the prompt mask from a single class_idx
                channel to a 3 channel RGB. Not specifying this will result in the prompt mask either being passed
                through as is if it is already in RGB format or being duplicated across the channel dimension.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        resample = resample if resample is not None else self.resample
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size
        size_dict = get_size_dict(size)

        # If segmentation map is passed we expect 2D images
        images = make_list_of_images(images, expected_ndims=2 if do_convert_rgb else 3)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None:
            raise ValueError("Size must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None and not do_convert_rgb:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_convert_rgb:
            palette = self.get_palette(num_labels) if num_labels is not None else None
            # Since this is the input for the next transformations its format should be the same as the input_data_format
            images = [
                self.mask_to_rgb(image=image, palette=palette, data_format=ChannelDimension.FIRST) for image in images
            ]
            input_data_format = ChannelDimension.FIRST

        if do_resize:
            images = [
                self.resize(image=image, size=size_dict, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

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

        return images

    def preprocess(
        self,
        images: Optional[ImageInput] = None,
        prompt_images: Optional[ImageInput] = None,
        prompt_masks: Optional[ImageInput] = None,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        num_labels: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to _preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            prompt_images (`ImageInput`):
                Prompt image to _preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            prompt_masks (`ImageInput`):
                Prompt mask from prompt image to _preprocess that specify prompt_masks value in the preprocessed output.
                Can either be in the format of segmentation maps (no channels) or RGB images. If in the format of
                RGB images, `do_convert_rgb` should be set to `False`. If in the format of segmentation maps, `num_labels`
                specifying `num_labels` is recommended to build a palette to map the prompt mask from a single channel to
                a 3 channel RGB. If `num_labels` is not specified, the prompt mask will be duplicated across the channel
                dimension.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BICUBIC`. Only has
                an effect if `do_resize` is set to `True`. Doesn't apply to prompt mask as it is resized using nearest.
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
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the prompt mask to RGB format. If `num_labels` is specified, a palette will be built
                to map the prompt mask from a single channel to a 3 channel RGB. If unset, the prompt mask is duplicated
                across the channel dimension. Must be set to `False` if the prompt mask is already in RGB format.
            num_labels: (`int`, *optional*):
                Number of classes in the segmentation task (excluding the background). If specified, a palette will be
                built, assuming that class_idx 0 is the background, to map the prompt mask from a plain segmentation map
                with no channels to a 3 channel RGB. Not specifying this will result in the prompt mask either being passed
                through as is if it is already in RGB format (if `do_convert_rgb` is false) or being duplicated
                across the channel dimension.
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
        if all(v is None for v in [images, prompt_images, prompt_masks]):
            raise ValueError("At least one of images, prompt_images, prompt_masks must be specified.")

        data = {}

        if images is not None:
            images = self._preprocess_step(
                images,
                is_mask=False,
                do_resize=do_resize,
                size=size,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                do_convert_rgb=False,
                data_format=data_format,
                input_data_format=input_data_format,
                **kwargs,
            )

            data["pixel_values"] = images

        if prompt_images is not None:
            prompt_images = self._preprocess_step(
                prompt_images,
                is_mask=False,
                do_resize=do_resize,
                size=size,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                do_convert_rgb=False,
                data_format=data_format,
                input_data_format=input_data_format,
                **kwargs,
            )

            data["prompt_pixel_values"] = prompt_images

        if prompt_masks is not None:
            prompt_masks = self._preprocess_step(
                prompt_masks,
                do_resize=do_resize,
                size=size,
                resample=PILImageResampling.NEAREST,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                do_convert_rgb=do_convert_rgb,
                num_labels=num_labels,
                data_format=data_format,
                input_data_format=input_data_format,
                **kwargs,
            )

            data["prompt_masks"] = prompt_masks

        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = None, num_labels: Optional[int] = None
    ):
        """
        Converts the output of [`SegGptImageSegmentationOutput`] into segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`SegGptImageSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
            num_labels (`int`, *optional*):
                Number of classes in the segmentation task (excluding the background). If specified, a palette will be
                built, assuming that class_idx 0 is the background, to map prediction masks from RGB values to class
                indices. This value should be the same used when preprocessing inputs.
        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        requires_backends(self, ["torch"])
        # batch_size x num_channels x 2*height x width
        masks = outputs.pred_masks

        # Predicted mask and prompt are concatenated in the height dimension
        # batch_size x num_channels x height x width
        masks = masks[:, :, masks.shape[2] // 2 :, :]

        # To unnormalize we need to permute to channel last
        # batch_size x height x width x num_channels
        std = torch.tensor(self.image_std).to(masks.device)
        mean = torch.tensor(self.image_mean).to(masks.device)

        masks = masks.permute(0, 2, 3, 1) * std + mean

        # batch_size x num_channels x height x width
        masks = masks.permute(0, 3, 1, 2)

        # Clip to match with palette if specified
        masks = torch.clip(masks * 255, 0, 255)

        semantic_segmentation = []
        palette_tensor = None
        palette = self.get_palette(num_labels) if num_labels is not None else None
        if palette is not None:
            palette_tensor = torch.tensor(palette).float().to(masks.device)
            _, num_channels, _, _ = masks.shape
            palette_tensor = palette_tensor.view(1, 1, num_labels + 1, num_channels)

        for idx, mask in enumerate(masks):
            if target_sizes is not None:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0),
                    size=target_sizes[idx],
                    mode="nearest",
                )[0]

            if num_labels is not None:
                channels, height, width = mask.shape
                dist = mask.permute(1, 2, 0).view(height, width, 1, channels)
                dist = dist - palette_tensor
                dist = torch.pow(dist, 2)
                dist = torch.sum(dist, dim=-1)
                pred = dist.argmin(dim=-1)

            else:
                # If no palette is specified SegGpt will try to paint using the mask class idx as RGB
                pred = mask.mean(dim=0).int()

            semantic_segmentation.append(pred)

        return semantic_segmentation


__all__ = ["SegGptImageProcessor"]

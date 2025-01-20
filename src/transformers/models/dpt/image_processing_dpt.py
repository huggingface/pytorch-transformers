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
"""Image processor class for DPT."""

import math
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union


if TYPE_CHECKING:
    from ...modeling_outputs import DepthEstimatorOutput

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import pad, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_torch_available,
    is_torch_tensor,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import (
    TensorType,
    filter_out_non_signature_kwargs,
    is_vision_available,
    logging,
    requires_backends,
)


if is_torch_available():
    import torch

if is_vision_available():
    import PIL


logger = logging.get_logger(__name__)


def get_resize_output_image_size(
    input_image: np.ndarray,
    output_size: Union[int, Iterable[int]],
    keep_aspect_ratio: bool,
    multiple: int,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    def constrain_to_multiple_of(val, multiple, min_val=0, max_val=None):
        x = round(val / multiple) * multiple

        if max_val is not None and x > max_val:
            x = math.floor(val / multiple) * multiple

        if x < min_val:
            x = math.ceil(val / multiple) * multiple

        return x

    output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    input_height, input_width = get_image_size(input_image, input_data_format)
    output_height, output_width = output_size

    # determine new height and width
    scale_height = output_height / input_height
    scale_width = output_width / input_width

    if keep_aspect_ratio:
        # scale as little as possible
        if abs(1 - scale_width) < abs(1 - scale_height):
            # fit width
            scale_height = scale_width
        else:
            # fit height
            scale_width = scale_height

    new_height = constrain_to_multiple_of(scale_height * input_height, multiple=multiple)
    new_width = constrain_to_multiple_of(scale_width * input_width, multiple=multiple)

    return (new_height, new_width)


class DPTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a DPT image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions. Can be overidden by `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the image after resizing. Can be overidden by `size` in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Defines the resampling filter to use if resizing the image. Can be overidden by `resample` in `preprocess`.
        keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
            If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved. Can
            be overidden by `keep_aspect_ratio` in `preprocess`.
        ensure_multiple_of (`int`, *optional*, defaults to 1):
            If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Can be overidden
            by `ensure_multiple_of` in `preprocess`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overidden by `do_rescale` in
            `preprocess`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overidden by `rescale_factor` in `preprocess`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `False`):
            Whether to apply center padding. This was introduced in the DINOv2 paper, which uses the model in
            combination with DPT.
        size_divisor (`int`, *optional*):
            If `do_pad` is `True`, pads the image dimensions to be divisible by this value. This was introduced in the
            DINOv2 paper, which uses the model in combination with DPT.
        do_reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is
            used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The
            background label will be replaced by 255. Can be overridden by the `do_reduce_labels` parameter in the
            `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = False,
        size_divisor: int = None,
        do_reduce_labels: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 384, "width": 384}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.size = size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.ensure_multiple_of = ensure_multiple_of
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_pad = do_pad
        self.size_divisor = size_divisor
        self.do_reduce_labels = do_reduce_labels

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to target size `(size["height"], size["width"])`. If `keep_aspect_ratio` is `True`, the image
        is resized to the largest possible size such that the aspect ratio is preserved. If `ensure_multiple_of` is
        set, the image is resized to a size that is a multiple of this value.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Target size of the output image.
            keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
                If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved.
            ensure_multiple_of (`int`, *optional*, defaults to 1):
                The image is resized to a size that is a multiple of this value.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Defines the resampling filter to use if resizing the image. Otherwise, the image is resized to size
                specified in `size`.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The size dictionary must contain the keys 'height' and 'width'. Got {size.keys()}")

        output_size = get_resize_output_image_size(
            image,
            output_size=(size["height"], size["width"]),
            keep_aspect_ratio=keep_aspect_ratio,
            multiple=ensure_multiple_of,
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

    def pad_image(
        self,
        image: np.array,
        size_divisor: int,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Center pad an image to be a multiple of `multiple`.

        Args:
            image (`np.ndarray`):
                Image to pad.
            size_divisor (`int`):
                The width and height of the image will be padded to a multiple of this number.
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

        def _get_pad(size, size_divisor):
            new_size = math.ceil(size / size_divisor) * size_divisor
            pad_size = new_size - size
            pad_size_left = pad_size // 2
            pad_size_right = pad_size - pad_size_left
            return pad_size_left, pad_size_right

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        height, width = get_image_size(image, input_data_format)

        pad_size_left, pad_size_right = _get_pad(height, size_divisor)
        pad_size_top, pad_size_bottom = _get_pad(width, size_divisor)

        return pad(image, ((pad_size_left, pad_size_right), (pad_size_top, pad_size_bottom)), data_format=data_format)

    # Copied from transformers.models.beit.image_processing_beit.BeitImageProcessor.reduce_label
    def reduce_label(self, label: ImageInput) -> np.ndarray:
        label = to_numpy_array(label)
        # Avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
        return label

    def _preprocess(
        self,
        image: ImageInput,
        do_reduce_labels: bool = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        keep_aspect_ratio: bool = None,
        ensure_multiple_of: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = None,
        size_divisor: int = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        if do_reduce_labels:
            image = self.reduce_label(image)

        if do_resize:
            image = self.resize(
                image=image,
                size=size,
                resample=resample,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=ensure_multiple_of,
                input_data_format=input_data_format,
            )

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        if do_pad:
            image = self.pad_image(image=image, size_divisor=size_divisor, input_data_format=input_data_format)

        return image

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        keep_aspect_ratio: bool = None,
        ensure_multiple_of: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = None,
        size_divisor: int = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""
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

        image = self._preprocess(
            image,
            do_reduce_labels=False,
            do_resize=do_resize,
            size=size,
            resample=resample,
            keep_aspect_ratio=keep_aspect_ratio,
            ensure_multiple_of=ensure_multiple_of,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            size_divisor=size_divisor,
            input_data_format=input_data_format,
        )
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    def _preprocess_segmentation_map(
        self,
        segmentation_map: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        keep_aspect_ratio: bool = None,
        ensure_multiple_of: int = None,
        do_reduce_labels: bool = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """Preprocesses a single segmentation map."""
        # All transformations expect numpy arrays.
        segmentation_map = to_numpy_array(segmentation_map)
        # Add an axis to the segmentation maps for transformations.
        if segmentation_map.ndim == 2:
            segmentation_map = segmentation_map[None, ...]
            added_dimension = True
            input_data_format = ChannelDimension.FIRST
        else:
            added_dimension = False
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
        segmentation_map = self._preprocess(
            image=segmentation_map,
            do_reduce_labels=do_reduce_labels,
            do_resize=do_resize,
            size=size,
            resample=resample,
            keep_aspect_ratio=keep_aspect_ratio,
            ensure_multiple_of=ensure_multiple_of,
            do_normalize=False,
            do_rescale=False,
            input_data_format=input_data_format,
        )
        # Remove extra axis if added
        if added_dimension:
            segmentation_map = np.squeeze(segmentation_map, axis=0)
        segmentation_map = segmentation_map.astype(np.int64)
        return segmentation_map

    # Copied from transformers.models.beit.image_processing_beit.BeitImageProcessor.__call__
    def __call__(self, images, segmentation_maps=None, **kwargs):
        # Overrides the `__call__` method of the `Preprocessor` class such that the images and segmentation maps can both
        # be passed in as positional arguments.
        return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        do_resize: bool = None,
        size: int = None,
        keep_aspect_ratio: bool = None,
        ensure_multiple_of: int = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = None,
        size_divisor: int = None,
        do_reduce_labels: Optional[bool] = None,
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
            segmentation_maps (`ImageInput`, *optional*):
                Segmentation map to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after reszing. If `keep_aspect_ratio` is `True`, the image is resized to the largest
                possible size such that the aspect ratio is preserved. If `ensure_multiple_of` is set, the image is
                resized to a size that is a multiple of this value.
            keep_aspect_ratio (`bool`, *optional*, defaults to `self.keep_aspect_ratio`):
                Whether to keep the aspect ratio of the image. If False, the image will be resized to (size, size). If
                True, the image will be resized to keep the aspect ratio and the size will be the maximum possible.
            ensure_multiple_of (`int`, *optional*, defaults to `self.ensure_multiple_of`):
                Ensure that the image size is a multiple of this value.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
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
            do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
                Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
                is used for background, and background itself is not included in all classes of a dataset (e.g.
                ADE20k). The background label will be replaced by 255.
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
        keep_aspect_ratio = keep_aspect_ratio if keep_aspect_ratio is not None else self.keep_aspect_ratio
        ensure_multiple_of = ensure_multiple_of if ensure_multiple_of is not None else self.ensure_multiple_of
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels

        images = make_list_of_images(images)

        if segmentation_maps is not None:
            segmentation_maps = make_list_of_images(segmentation_maps, expected_ndims=2)

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
            size_divisibility=size_divisor,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        images = [
            self._preprocess_image(
                image=img,
                do_resize=do_resize,
                do_rescale=do_rescale,
                do_normalize=do_normalize,
                do_pad=do_pad,
                size=size,
                resample=resample,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=ensure_multiple_of,
                rescale_factor=rescale_factor,
                image_mean=image_mean,
                image_std=image_std,
                size_divisor=size_divisor,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for img in images
        ]

        data = {"pixel_values": images}

        if segmentation_maps is not None:
            segmentation_maps = [
                self._preprocess_segmentation_map(
                    segmentation_map=segmentation_map,
                    do_reduce_labels=do_reduce_labels,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    keep_aspect_ratio=keep_aspect_ratio,
                    ensure_multiple_of=ensure_multiple_of,
                    input_data_format=input_data_format,
                )
                for segmentation_map in segmentation_maps
            ]

            data["labels"] = segmentation_maps

        return BatchFeature(data=data, tensor_type=return_tensors)

    # Copied from transformers.models.beit.image_processing_beit.BeitImageProcessor.post_process_semantic_segmentation with Beit->DPT
    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple] = None):
        """
        Converts the output of [`DPTForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`DPTForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        # TODO: add support for other frameworks
        logits = outputs.logits

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            if is_torch_tensor(target_sizes):
                target_sizes = target_sizes.numpy()

            semantic_segmentation = []

            for idx in range(len(logits)):
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = logits.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_depth_estimation(
        self,
        outputs: "DepthEstimatorOutput",
        target_sizes: Optional[Union[TensorType, List[Tuple[int, int]], None]] = None,
    ) -> List[Dict[str, TensorType]]:
        """
        Converts the raw output of [`DepthEstimatorOutput`] into final depth predictions and depth PIL images.
        Only supports PyTorch.

        Args:
            outputs ([`DepthEstimatorOutput`]):
                Raw outputs of the model.
            target_sizes (`TensorType` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.

        Returns:
            `List[Dict[str, TensorType]]`: A list of dictionaries of tensors representing the processed depth
            predictions.
        """
        requires_backends(self, "torch")

        predicted_depth = outputs.predicted_depth

        if (target_sizes is not None) and (len(predicted_depth) != len(target_sizes)):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
            )

        results = []
        target_sizes = [None] * len(predicted_depth) if target_sizes is None else target_sizes
        for depth, target_size in zip(predicted_depth, target_sizes):
            if target_size is not None:
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(0).unsqueeze(1), size=target_size, mode="bicubic", align_corners=False
                ).squeeze()

            results.append({"predicted_depth": depth})

        return results


__all__ = ["DPTImageProcessor"]

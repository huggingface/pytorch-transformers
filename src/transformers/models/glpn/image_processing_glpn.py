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
"""Image processor class for GLPN."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union


if TYPE_CHECKING:
    from ...modeling_outputs import DepthEstimatorOutput

import numpy as np
import PIL.Image

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_torch_available,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, filter_out_non_signature_kwargs, logging, requires_backends


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class GLPNImageProcessor(BaseImageProcessor):
    r"""
    Constructs a GLPN image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions, rounding them down to the closest multiple of
            `size_divisor`. Can be overridden by `do_resize` in `preprocess`.
        size_divisor (`int`, *optional*, defaults to 32):
            When `do_resize` is `True`, images are resized so their height and width are rounded down to the closest
            multiple of `size_divisor`. Can be overridden by `size_divisor` in `preprocess`.
        resample (`PIL.Image` resampling filter, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Can be
            overridden by `do_rescale` in `preprocess`.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size_divisor: int = 32,
        resample=PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        **kwargs,
    ) -> None:
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.size_divisor = size_divisor
        self.resample = resample
        super().__init__(**kwargs)

    def resize(
        self,
        image: np.ndarray,
        size_divisor: int,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize the image, rounding the (height, width) dimensions down to the closest multiple of size_divisor.

        If the image is of dimension (3, 260, 170) and size_divisor is 32, the image will be resized to (3, 256, 160).

        Args:
            image (`np.ndarray`):
                The image to resize.
            size_divisor (`int`):
                The image is resized so its height and width are rounded down to the closest multiple of
                `size_divisor`.
            resample:
                `PIL.Image` resampling filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If `None`, the channel dimension format of the input
                image is used. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not set, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        height, width = get_image_size(image, channel_dim=input_data_format)
        # Rounds the height and width down to the closest multiple of size_divisor
        new_h = height // size_divisor * size_divisor
        new_w = width // size_divisor * size_divisor
        image = resize(
            image,
            (new_h, new_w),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        return image

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: Union["PIL.Image.Image", TensorType, List["PIL.Image.Image"], List[TensorType]],
        do_resize: Optional[bool] = None,
        size_divisor: Optional[int] = None,
        resample=None,
        do_rescale: Optional[bool] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Preprocess the given images.

        Args:
            images (`PIL.Image.Image` or `TensorType` or `List[np.ndarray]` or `List[TensorType]`):
                Images to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_normalize=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the input such that the (height, width) dimensions are a multiple of `size_divisor`.
            size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
                When `do_resize` is `True`, images are resized so their height and width are rounded down to the
                closest multiple of `size_divisor`.
            resample (`PIL.Image` resampling filter, *optional*, defaults to `self.resample`):
                `PIL.Image` resampling filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.).
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - `None`: Return a list of `np.ndarray`.
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
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        resample = resample if resample is not None else self.resample

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # Here, the rescale() method uses a constant rescale_factor. It does not need to be validated
        # with a rescale_factor.
        validate_preprocess_arguments(
            do_resize=do_resize,
            size=size_divisor,  # Here, size_divisor is used as a parameter for optimal resizing instead of size.
            resample=resample,
        )

        # All transformations expect numpy arrays.
        images = [to_numpy_array(img) for img in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_resize:
            images = [
                self.resize(image, size_divisor=size_divisor, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        if do_rescale:
            images = [self.rescale(image, scale=1 / 255, input_data_format=input_data_format) for image in images]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

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
                depth = depth[None, None, ...]
                depth = torch.nn.functional.interpolate(depth, size=target_size, mode="bicubic", align_corners=False)
                depth = depth.squeeze()

            results.append({"predicted_depth": depth})

        return results


__all__ = ["GLPNImageProcessor"]

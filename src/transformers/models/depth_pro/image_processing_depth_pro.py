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
"""Image processor class for DepthPro."""

from typing import Dict, List, Optional, Union
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from icecream import ic


from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    pil_torch_interpolation_mapping,
)
from ...utils import TensorType, filter_out_non_signature_kwargs, logging

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


class DepthProImageProcessor(BaseImageProcessor):
    r"""
    Constructs a DepthPro image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 1536, "width": 1536}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        antialias (`bool`, *optional*, defaults to `False`):
            Whether to apply an anti-aliasing filter when resizing the image. It only affects tensors with
            bilinear or bicubic modes and it is ignored otherwise.
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

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        antialias: bool = False,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 1536, "width": 1536}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.antialias = antialias
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def resize(
        self,
        images: List[np.ndarray],
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        antialias: bool = False,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            images (`List[np.ndarray]`):
                Images to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            antialias (`bool`, *optional*, defaults to `False`):
                Whether to apply an anti-aliasing filter when resizing the image. It only affects tensors with
                bilinear or bicubic modes and it is ignored otherwise.
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
            `np.ndarray`: The resized images.
        """
        requires_backends(self, "torch")

        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])

        images = np.stack(images)
        images = torch.from_numpy(images)

        return torch.nn.functional.interpolate(
            # input should be (B, C, H, W)
            input=images,
            size=output_size,
            mode=pil_torch_interpolation_mapping[resample].value,
            antialias=antialias,
        )

    def _validate_input_arguments(
        self,
        do_resize: bool,
        size: Dict[str, int],
        resample: PILImageResampling,
        antialias: bool,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
        data_format: Union[str, ChannelDimension],
    ):
        if data_format != ChannelDimension.FIRST:
            raise ValueError("Only channel first data format is currently supported.")

        if do_resize and None in (size, resample, antialias):
            raise ValueError("Size, resample and antialias must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and None in (image_mean, image_std):
            raise ValueError("Image mean and standard deviation must be specified if do_normalize is True.")

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        antialias: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
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
            antialias (`bool`, *optional*, defaults to `False`):
                Whether to apply an anti-aliasing filter when resizing the image. It only affects tensors with
                bilinear or bicubic modes and it is ignored otherwise.
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
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        antialias = antialias if antialias is not None else self.antialias
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size
        size_dict = get_size_dict(size)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        self._validate_input_arguments(
            do_resize=do_resize,
            size=size,
            resample=resample,
            antialias=antialias,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            data_format=data_format,
        )

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

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

        # depth-pro scales the image before resizing it
        # uses torch interpolation which requires ChannelDimension.FIRST
        if do_resize:
            images = self.resize(images, size=size_dict, resample=resample, antialias=antialias)
            images = images.numpy()

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_depth_estimation(
        self,
        predicted_depths: Union[TensorType, List[TensorType]],
        fovs: Optional[Union[TensorType, List[TensorType], None]] = None,
        target_sizes: Optional[Union[TensorType, List[Tuple[int, int]], None]] = None,
    ) -> Dict[str, List[TensorType]]:
        """
        Post-processes the raw depth predictions from the model to generate final depth predictions and optionally
        resizes them to specified target sizes. This function supports scaling based on the field of view (FoV)
        and adjusts depth values accordingly.

        Args:
            predicted_depths (`Union[TensorType, List[TensorType]]`):
                Raw depth predictions output by the model. Can be a single tensor or a list of tensors, each
                corresponding to an image in the batch.
            fovs (`Optional[Union[TensorType, List[TensorType], None]]`, *optional*, defaults to `None`):
                Field of view (FoV) values corresponding to each depth prediction. Should have the same length
                as `predicted_depths` if provided. If `None`, FoV scaling is skipped.
            target_sizes (`Optional[Union[TensorType, List[Tuple[int, int]], None]]`, *optional*, defaults to `None`):
                Target sizes to resize the depth predictions. Can be a tensor of shape `(batch_size, 2)` 
                or a list of tuples `(height, width)` for each image in the batch. If `None`, no resizing
                is performed.

        Returns:
            `Dict[str, List[TensorType]]`:
                A dictionary containing:
                    - `"predicted_depth"`: A list of processed depth tensors.
                    - `"fov"`: A list of processed FoV values if provided, otherwise `None`.

        Raises:
            `ValueError`:
                If the lengths of `predicted_depths`, `fovs`, or `target_sizes` are mismatched.
        """
        requires_backends(self, "torch")

        if (fovs is not None) and (len(predicted_depths) != len(fovs)):
            raise ValueError(
                "Make sure that you pass in as many fov values as the batch dimension of the predicted depth"
            )
        if (target_sizes is not None) and (len(predicted_depths) != len(target_sizes)):
            raise ValueError(
                "Make sure that you pass in as many fov values as the batch dimension of the predicted depth"
            )

        outputs = {
            "predicted_depth": [],
            "fov": [] if fovs is not None else None
        }

        fovs = [None] * len(predicted_depths) if fovs is None else fovs
        target_sizes = [None] * len(predicted_depths) if target_sizes is None else target_sizes

        for predicted_depth, fov, target_size in zip(predicted_depths, fovs, target_sizes):

            if target_size is not None:

                # scale image w.r.t fov
                if fov is not None:
                    width = target_size[1]
                    fov = 0.5 * width / torch.tan(0.5 * torch.deg2rad(fov))
                    predicted_depth = predicted_depth * width / fov
                    outputs["fov"].append(fov)

                # interpolate
                predicted_depth = self.resize(
                    predicted_depth.unsqueeze(0).unsqueeze(1),
                    size=target_size,
                    resample=self.resample,
                    antialias=self.antialias
                ).squeeze()

            # inverse the depth
            predicted_depth = 1.0 / torch.clamp(predicted_depth, min=1e-4, max=1e4)

            outputs["predicted_depth"].append(predicted_depth)

        return outputs

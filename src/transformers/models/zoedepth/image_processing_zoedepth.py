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
"""Image processor class for ZoeDepth."""

import math
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


if TYPE_CHECKING:
    from .modeling_zoedepth import ZoeDepthDepthEstimatorOutput

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import PaddingMode, pad, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import (
    TensorType,
    filter_out_non_signature_kwargs,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)


if is_vision_available():
    import PIL

if is_torch_available():
    import torch
    from torch import nn


logger = logging.get_logger(__name__)


def get_resize_output_image_size(
    input_image: np.ndarray,
    output_size: Union[int, Iterable[int]],
    keep_aspect_ratio: bool,
    multiple: int,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    def constrain_to_multiple_of(val, multiple, min_val=0):
        x = (np.round(val / multiple) * multiple).astype(int)

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


class ZoeDepthImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ZoeDepth image processor.

    Args:
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to apply pad the input.
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
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions. Can be overidden by `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 384, "width": 512}`):
            Size of the image after resizing. Size of the image after resizing. If `keep_aspect_ratio` is `True`,
            the image is resized by choosing the smaller of the height and width scaling factors and using it for both dimensions.
            If `ensure_multiple_of` is also set, the image is further resized to a size that is a multiple of this value.
            Can be overidden by `size` in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Defines the resampling filter to use if resizing the image. Can be overidden by `resample` in `preprocess`.
        keep_aspect_ratio (`bool`, *optional*, defaults to `True`):
            If `True`, the image is resized by choosing the smaller of the height and width scaling factors and using it
            for both dimensions. This ensures that the image is scaled down as little as possible while still fitting
            within the desired output size. In case `ensure_multiple_of` is also set, the image is further resized to a
            size that is a multiple of this value by flooring the height and width to the nearest multiple of this value.
            Can be overidden by `keep_aspect_ratio` in `preprocess`.
        ensure_multiple_of (`int`, *optional*, defaults to 32):
            If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Works by flooring
            the height and width to the nearest multiple of this value.

            Works both with and without `keep_aspect_ratio` being set to `True`. Can be overidden by `ensure_multiple_of`
            in `preprocess`.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_pad: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        keep_aspect_ratio: bool = True,
        ensure_multiple_of: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        size = size if size is not None else {"height": 384, "width": 512}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.size = size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.ensure_multiple_of = ensure_multiple_of
        self.resample = resample

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        keep_aspect_ratio: bool = False,
        ensure_multiple_of: int = 1,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
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
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Defines the resampling filter to use if resizing the image. Otherwise, the image is resized to size
                specified in `size`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        data_format = data_format if data_format is not None else input_data_format

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

        height, width = output_size

        torch_image = torch.from_numpy(image).unsqueeze(0)
        torch_image = torch_image.permute(0, 3, 1, 2) if input_data_format == "channels_last" else torch_image

        # TODO support align_corners=True in image_transforms.resize
        requires_backends(self, "torch")
        resample_to_mode = {PILImageResampling.BILINEAR: "bilinear", PILImageResampling.BICUBIC: "bicubic"}
        mode = resample_to_mode[resample]
        resized_image = nn.functional.interpolate(
            torch_image, (int(height), int(width)), mode=mode, align_corners=True
        )
        resized_image = resized_image.squeeze().numpy()

        resized_image = to_channel_dimension_format(
            resized_image, data_format, input_channel_dim=ChannelDimension.FIRST
        )

        return resized_image

    def pad_image(
        self,
        image: np.array,
        mode: PaddingMode = PaddingMode.REFLECT,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pad an image as done in the original ZoeDepth implementation.

        Padding fixes the boundary artifacts in the output depth map.
        Boundary artifacts are sometimes caused by the fact that the model is trained on NYU raw dataset
        which has a black or white border around the image. This function pads the input image and crops
        the prediction back to the original size / view.

        Args:
            image (`np.ndarray`):
                Image to pad.
            mode (`PaddingMode`):
                The padding mode to use. Can be one of:
                    - `"constant"`: pads with a constant value.
                    - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
                    vector along each axis.
                    - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
                    - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
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
        height, width = get_image_size(image, input_data_format)

        pad_height = int(np.sqrt(height / 2) * 3)
        pad_width = int(np.sqrt(width / 2) * 3)

        return pad(
            image,
            padding=((pad_height, pad_height), (pad_width, pad_width)),
            mode=mode,
            data_format=data_format,
            input_data_format=input_data_format,
        )

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_pad: bool = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_resize: bool = None,
        size: int = None,
        keep_aspect_ratio: bool = None,
        ensure_multiple_of: int = None,
        resample: PILImageResampling = None,
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
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the input image.
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
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. If `keep_aspect_ratio` is `True`, he image is resized by choosing the
                smaller of the height and width scaling factors and using it for both dimensions. If `ensure_multiple_of`
                is also set, the image is further resized to a size that is a multiple of this value.
            keep_aspect_ratio (`bool`, *optional*, defaults to `self.keep_aspect_ratio`):
                If `True` and `do_resize=True`, the image is resized by choosing the smaller of the height and width
                scaling factors and using it for both dimensions. This ensures that the image is scaled down as little
                as possible while still fitting within the desired output size. In case `ensure_multiple_of` is also
                set, the image is further resized to a size that is a multiple of this value by flooring the height and
                width to the nearest multiple of this value.
            ensure_multiple_of (`int`, *optional*, defaults to `self.ensure_multiple_of`):
                If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Works by
                flooring the height and width to the nearest multiple of this value.

                Works both with and without `keep_aspect_ratio` being set to `True`. Can be overidden by
                `ensure_multiple_of` in `preprocess`.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
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

        images = make_list_of_images(images)

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
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
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

        if do_pad:
            images = [self.pad_image(image=image, input_data_format=input_data_format) for image in images]

        if do_resize:
            images = [
                self.resize(
                    image=image,
                    size=size,
                    resample=resample,
                    keep_aspect_ratio=keep_aspect_ratio,
                    ensure_multiple_of=ensure_multiple_of,
                    input_data_format=input_data_format,
                )
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

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_depth_estimation(
        self,
        outputs: "ZoeDepthDepthEstimatorOutput",
        source_sizes: Optional[Union[TensorType, List[Tuple[int, int]], None]] = None,
        target_sizes: Optional[Union[TensorType, List[Tuple[int, int]], None]] = None,
        outputs_flipped: Optional[Union["ZoeDepthDepthEstimatorOutput", None]] = None,
        do_remove_padding: Optional[Union[bool, None]] = None,
    ) -> List[Dict[str, TensorType]]:
        """
        Converts the raw output of [`ZoeDepthDepthEstimatorOutput`] into final depth predictions and depth PIL images.
        Only supports PyTorch.

        Args:
            outputs ([`ZoeDepthDepthEstimatorOutput`]):
                Raw outputs of the model.
            source_sizes (`TensorType` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the source size
                (height, width) of each image in the batch before preprocessing. This argument should be dealt as
                "required" unless the user passes `do_remove_padding=False` as input to this function.
            target_sizes (`TensorType` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.
            outputs_flipped ([`ZoeDepthDepthEstimatorOutput`], *optional*):
                Raw outputs of the model from flipped input (averaged out in the end).
            do_remove_padding (`bool`, *optional*):
                By default ZoeDepth addes padding equal to `int(√(height / 2) * 3)` (and similarly for width) to fix the
                boundary artifacts in the output depth map, so we need remove this padding during post_processing. The
                parameter exists here in case the user changed the image preprocessing to not include padding.

        Returns:
            `List[Dict[str, TensorType]]`: A list of dictionaries of tensors representing the processed depth
            predictions.
        """
        requires_backends(self, "torch")

        predicted_depth = outputs.predicted_depth

        if (outputs_flipped is not None) and (predicted_depth.shape != outputs_flipped.predicted_depth.shape):
            raise ValueError("Make sure that `outputs` and `outputs_flipped` have the same shape")

        if (target_sizes is not None) and (len(predicted_depth) != len(target_sizes)):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
            )

        if do_remove_padding is None:
            do_remove_padding = self.do_pad

        if source_sizes is None and do_remove_padding:
            raise ValueError(
                "Either `source_sizes` should be passed in, or `do_remove_padding` should be set to False"
            )

        if (source_sizes is not None) and (len(predicted_depth) != len(source_sizes)):
            raise ValueError(
                "Make sure that you pass in as many source image sizes as the batch dimension of the logits"
            )

        if outputs_flipped is not None:
            predicted_depth = (predicted_depth + torch.flip(outputs_flipped.predicted_depth, dims=[-1])) / 2

        predicted_depth = predicted_depth.unsqueeze(1)

        # Zoe Depth model adds padding around the images to fix the boundary artifacts in the output depth map
        # The padding length is `int(np.sqrt(img_h/2) * fh)` for the height and similar for the width
        # fh (and fw respectively) are equal to '3' by default
        # Check [here](https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/models/depth_model.py#L57)
        # for the original implementation.
        # In this section, we remove this padding to get the final depth image and depth prediction
        padding_factor_h = padding_factor_w = 3

        results = []
        target_sizes = [None] * len(predicted_depth) if target_sizes is None else target_sizes
        source_sizes = [None] * len(predicted_depth) if source_sizes is None else source_sizes
        for depth, target_size, source_size in zip(predicted_depth, target_sizes, source_sizes):
            # depth.shape = [1, H, W]
            if source_size is not None:
                pad_h = pad_w = 0

                if do_remove_padding:
                    pad_h = int(np.sqrt(source_size[0] / 2) * padding_factor_h)
                    pad_w = int(np.sqrt(source_size[1] / 2) * padding_factor_w)

                depth = nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=[source_size[0] + 2 * pad_h, source_size[1] + 2 * pad_w],
                    mode="bicubic",
                    align_corners=False,
                )

                if pad_h > 0:
                    depth = depth[:, :, pad_h:-pad_h, :]
                if pad_w > 0:
                    depth = depth[:, :, :, pad_w:-pad_w]

                depth = depth.squeeze(1)
            # depth.shape = [1, H, W]
            if target_size is not None:
                target_size = [target_size[0], target_size[1]]
                depth = nn.functional.interpolate(
                    depth.unsqueeze(1), size=target_size, mode="bicubic", align_corners=False
                )
            depth = depth.squeeze()
            # depth.shape = [H, W]
            results.append({"predicted_depth": depth})

        return results


__all__ = ["ZoeDepthImageProcessor"]

# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the S-Lab License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/sczhou/ProPainter/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Video processor class for ProPainter."""

from typing import Dict, List, Optional, Union

import numpy as np

from transformers.utils import is_vision_available
from transformers.utils.generic import TensorType

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    VideoInput,
    get_channel_dimension_axis,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import (
    filter_out_non_signature_kwargs,
    is_scipy_available,
    logging,
    requires_backends,
)


if is_scipy_available():
    from scipy.ndimage import binary_dilation

if is_vision_available():
    import PIL

logger = logging.get_logger(__name__)

# Adapted from original code at https://github.com/sczhou/ProPainter


def make_batched(videos) -> List[List[VideoInput]]:
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        if isinstance(videos[0], PIL.Image.Image) or len(videos[0].shape) == 3:
            return [videos]
        elif len(videos[0].shape) == 4:
            return [list(video) for video in videos]

    elif is_valid_image(videos) and len(videos.shape) == 4:
        return [list(videos)]

    raise ValueError(f"Could not make batched video from {videos}")


def convert_to_grayscale_and_dilation(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    mask_dilation: int = 4,
) -> ImageInput:
    """
    Converts image(video frame) to grayscale format using the NTSC formula and performs binary dilation on an image. Only support numpy and PIL image. TODO support torch
    and tensorflow grayscale conversion

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (Image):
            The image to convert.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
        mask_dilation (`int`, *optional*, defaults to `4`):
            The number of iterations for binary dilation the mask used in video processing tasks.
    """
    requires_backends(convert_to_grayscale_and_dilation, ["vision"])
    if isinstance(image, np.ndarray):
        if input_data_format == ChannelDimension.FIRST:
            if image.shape[0] == 1:
                gray_image = image
            else:
                gray_image = image[0, ...] * 0.2989 + image[1, ...] * 0.5870 + image[2, ...] * 0.1140
                gray_image = np.stack([gray_image] * 1, axis=0)
            gray_dilated_image = binary_dilation(gray_image, iterations=mask_dilation).astype(np.float32)
        elif input_data_format == ChannelDimension.LAST:
            if image.shape[-1] == 1:
                gray_image = image
            else:
                gray_image = image[..., 0] * 0.2989 + image[..., 1] * 0.5870 + image[..., 2] * 0.1140
                gray_image = np.stack([gray_image] * 1, axis=-1)
            gray_dilated_image = binary_dilation(gray_image, iterations=mask_dilation).astype(np.float32)
        return gray_dilated_image

    if not isinstance(image, PIL.Image.Image):
        return image

    image = np.array(image.convert("L"))
    image = np.stack([image] * 1, axis=0)
    image = binary_dilation(image, iterations=mask_dilation).astype(np.float32)

    return image


def extrapolation(
    image: ImageInput,
    scale_size: Optional[tuple[float, float]] = None,
    data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    """
    Prepares video frames for the outpainting process by extrapolating the field of view (FOV) and generating masks.

    This function performs the following tasks:
    (a) Scaling: If the `scale_size` parameter is provided(necesaary to provide for outpainting), it resizes the dimensions of the video frames based on
        the scaling factors for height  and width. This step is crucial for `"video_outpainting"` mode. If `scale_size` is `None`, no resizing is applied.
    (b) Field of View Expansion: The function calculates new dimensions for the frames to accommodate the expanded FOV.
        The new dimensions are adjusted to be divisible by 8 to meet processing requirements.
    (c) Frame Adjustment: The original frames are placed at the center of the new, larger frames. The rest of the frame is filled with zeros.
    (d) Mask Generation:
        - Flow Masks: Creates masks indicating the missing regions in the expanded FOV. These masks are used for flow-based propagation.
        - Dilated Masks: Generates additional masks with dilated borders to account for edge effects and improve the robustness of the process.
    (e) Format Conversion: Converts the image and masks to the specified channel dimension format, if needed.

    Args:
        image (Image):
            The video frames to convert.
        scale_size (`tuple[float, float]`, *optional*, defaults to `None`):
            Tuple containing scaling factors for the video's height and width dimensions during `"video_outpainting"` mode.
            It is only applicable during `"video_outpainting"` mode. If `None`, no scaling is applied and code execution will end.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format of the image. If not provided, it will be the same as the input image.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred.
    Returns:
        image (`Image`): A list of video frames with expanded FOV, adjusted to the specified channel dimension format.
        flow_masks (`Image`): A list of masks for the missing regions, intended for flow-based applications. Each mask is scaled to fit the expanded FOV.
        masks_dilated (`Image`): A list of dilated masks for the missing regions, also scaled to fit the expanded FOV.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)

    height, width = get_image_size(image, input_data_format)

    num_channels = image.shape[get_channel_dimension_axis(image, input_data_format)]

    # Defines new FOV.
    height_extr = int(scale_size[0] * height)
    width_extr = int(scale_size[1] * width)
    height_extr = height_extr - height_extr % 8
    width_extr = width_extr - width_extr % 8
    height_start = int((height_extr - height) / 2)
    width_start = int((width_extr - width) / 2)

    # Extrapolates the FOV for video.

    if input_data_format == ChannelDimension.LAST:
        frame = np.zeros(((height_extr, width_extr, num_channels)), dtype=np.float32)
        frame[
            height_start : height_start + height,
            width_start : width_start + width,
            :,
        ] = image
        image = frame
    elif input_data_format == ChannelDimension.FIRST:
        frame = np.zeros((num_channels, height_extr, width_extr), dtype=np.float32)  # Adjusted shape
        frame[
            :,
            height_start : height_start + height,
            width_start : width_start + width,
        ] = image
        image = frame

    # Generates the mask for missing region.

    dilate_h = 4 if height_start > 10 else 0
    dilate_w = 4 if width_start > 10 else 0
    mask = np.ones(((height_extr, width_extr)), dtype=np.float32)

    mask[
        height_start + dilate_h : height_start + height - dilate_h,
        width_start + dilate_w : width_start + width - dilate_w,
    ] = 0
    flow_mask = mask

    mask[height_start : height_start + height, width_start : width_start + width] = 0
    mask_dilated = mask

    if input_data_format == ChannelDimension.FIRST:
        # Expand dimensions as (1, height, width)
        flow_mask = np.expand_dims(flow_mask, axis=0)
        mask_dilated = np.expand_dims(mask_dilated, axis=0)
    elif input_data_format == ChannelDimension.LAST:
        # Expand dimensions as (height, width, 1)
        flow_mask = np.expand_dims(flow_mask, axis=-1)
        mask_dilated = np.expand_dims(mask_dilated, axis=-1)

    image = (
        to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        if data_format is not None
        else image
    )

    flow_mask = (
        to_channel_dimension_format(flow_mask, data_format, input_channel_dim=input_data_format)
        if data_format is not None
        else image
    )

    mask_dilated = (
        to_channel_dimension_format(mask_dilated, data_format, input_channel_dim=input_data_format)
        if data_format is not None
        else image
    )

    return image, flow_mask, mask_dilated


class ProPainterVideoProcessor(BaseImageProcessor):
    r"""
    Constructs a ProPainter video processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `False`):
            Whether to resize the video's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 256}`):
            Size of the output video after resizing. The shortest edge of the video will be resized to
            `size["shortest_edge"]` while maintaining the aspect ratio of the original video. Can be overriden by
            `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.NEAREST`):
            Resampling filter to use if resizing the video. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `False`):
            Whether to center crop the video to the specified `crop_size`. Can be overridden by the `do_center_crop`
            parameter in the `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the video after applying the center crop. Can be overridden by the `crop_size` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the video by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255.0`):
            Defines the scale factor to use if rescaling the video. Can be overridden by the `rescale_factor` parameter
            in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the video. This is a float or list of floats the length of the number of
            channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
            number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
        video_painting_mode (`str`, *optional*, defaults to `"video_inpainting"`):
            Specifies the mode for video reconstruction tasks, such as object removal, video completion, video outpainting.
            choices=['video_inpainting', 'video_outpainting']
        scale_size (`tuple[float, float]`, *optional*):
            Tuple containing scaling factors for the video's height and width dimensions during `"video_outpainting"` mode.
            It is only applicable during `"video_outpainting"` mode. If `None`, no scaling is applied and code execution will end.
        mask_dilation (`int`, *optional*, defaults to 4):
            The number of iterations for binary dilation the mask used in video processing tasks.
    """

    model_input_names = ["pixel_values_videos", "flow_masks", "masks_dilated"]

    def __init__(
        self,
        do_resize: bool = False,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.NEAREST,
        do_center_crop: bool = False,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255.0,
        do_normalize: bool = False,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        video_painting_mode: str = "video_inpainting",
        scale_size: Optional[tuple[float, float]] = None,
        mask_dilation: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"shortest_edge": 256}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.video_painting_mode = video_painting_mode
        self.scale_size = scale_size
        self.mask_dilation = mask_dilation

    # Adapted from transformers.models.vivit.image_processing_vivit.VivitImageProcessor.resize
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.NEAREST,
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
                have the size `(h, w)`. If `size` is of the form `{"shortest_edge": s}`, the output image will have its
                shortest edge of length `s` while keeping the aspect ratio of the original image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.NEAREST`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        size = get_size_dict(size, default_to_square=False)
        if "shortest_edge" in size:
            output_size = get_resize_output_image_size(
                image,
                size["shortest_edge"],
                default_to_square=False,
                input_data_format=input_data_format,
            )
        elif "height" in size and "width" in size:
            output_size = (size["height"], size["width"])
        else:
            raise ValueError(f"Size must have 'height' and 'width' or 'shortest_edge' as keys. Got {size.keys()}")
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _extrapolation(
        self,
        images: ImageInput,
        scale_size: Optional[tuple[float, float]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Preprocess the video for `video_outpainting` mode.

        Args:
        images (Image):
            The video frames to convert.
        scale_size (`tuple[float, float]`, *optional*, defaults to `None`):
            Tuple containing scaling factors for the video's height and width dimensions during `"video_outpainting"` mode.
            It is only applicable during `"video_outpainting"` mode. If `None`, no scaling is applied and code execution will end.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format of the image. If not provided, it will be the same as the input image.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred.
        """
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        images, flow_masks, masks_dilated = zip(
            *[
                extrapolation(
                    image=image,
                    scale_size=scale_size,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for image in images
            ]
        )

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        flow_masks = [
            to_channel_dimension_format(flow_mask, data_format, input_channel_dim=input_data_format)
            for flow_mask in flow_masks
        ]

        masks_dilated = [
            to_channel_dimension_format(mask_dilated, data_format, input_channel_dim=input_data_format)
            for mask_dilated in masks_dilated
        ]

        return images, flow_masks, masks_dilated

    def _preprocess(
        self,
        images: ImageInput,
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
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        is_mask_frame: bool = None,
        mask_dilation: int = None,
    ) -> np.ndarray:
        """Preprocesses a single image (one video frame)."""

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
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled videos. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_resize:
            images = [
                self.resize(
                    image=image,
                    size=size,
                    resample=resample,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        if is_mask_frame:
            images = [
                convert_to_grayscale_and_dilation(
                    image,
                    input_data_format=input_data_format,
                    mask_dilation=mask_dilation,
                )
                for image in images
            ]

        if do_center_crop:
            images = [self.center_crop(image, size=crop_size, input_data_format=input_data_format) for image in images]

        if do_rescale:
            images = [
                self.rescale(
                    image=image,
                    scale=rescale_factor,
                    dtype=np.float32,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        # If the mask frames even consisted of 0s and 255s, they are already rescaled and normally masks are not normalised as well
        if do_normalize and not (is_mask_frame):
            images = [
                self.normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        return images

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        videos: VideoInput,
        masks: VideoInput,
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
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        video_painting_mode: str = None,
        scale_size: Optional[tuple[float, float]] = None,
        mask_dilation: int = None,
    ):
        """
        Preprocess an video or batch of videos.

        Args:
            videos (`VideoInput`):
                Video frames to preprocess. Expects a single or batch of video frames with pixel values ranging from 0
                to 255. If passing in frames with pixel values between 0 and 1, set `do_rescale=False`.
            masks (`VideoInput`):
                masks for each frames to preprocess. Expects a single or batch of masks frames with pixel values ranging from 0
                to 255. If passing in frames with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the video.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the video after applying resize.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the video. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to centre crop the video.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the video after applying the centre crop.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the video values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the video by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the video.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                video mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                video standard deviation.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output video. Can be one of:
                    - `ChannelDimension.FIRST`: video in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: video in (height, width, num_channels) format.
                    - Unset: Use the inferred channel dimension format of the input video.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input video. If unset, the channel dimension format is inferred
                from the input video. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: video in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: video in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: video in (height, width) format.
            video_painting_mode (`str`, *optional*, defaults to `self.video_inpainting`):
                Specifies the mode for video reconstruction tasks, such as object removal, video completion, video outpainting.
                choices=['video_inpainting', 'video_outpainting']
            scale_size (`tuple[float, float]`, *optional*, defaults to `self.scale_size`):
                Tuple containing scaling factors for the video's height and width dimensions during `"video_outpainting"` mode.
                It is only applicable during `"video_outpainting"` mode. If `None`, no scaling is applied and code execution will end.
            mask_dilation (`int`, *optional*, defaults to `self.mask_dilation`):
                The number of iterations for binary dilation the mask used in video processing tasks.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        video_painting_mode = video_painting_mode if video_painting_mode is not None else self.video_painting_mode
        mask_dilation = mask_dilation if mask_dilation is not None else self.mask_dilation

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        if video_painting_mode == "video_outpainting":
            assert scale_size is not None, "Please provide a outpainting scale (scale_height, scale_width)."

        if not valid_images(videos):
            raise ValueError(
                "Invalid video type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        if not valid_images(masks):
            raise ValueError(
                "Invalid video type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        videos = make_batched(videos)
        masks = make_batched(masks)

        video_size = get_image_size(to_numpy_array(videos[0][0]), input_data_format)
        video_size = (
            video_size[0] - video_size[0] % 8,
            video_size[1] - video_size[1] % 8,
        )

        pixel_values = [
            self._preprocess(
                images=video,
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
                data_format=data_format,
                input_data_format=input_data_format,
                is_mask_frame=False,
            )
            for video in videos
        ]

        if video_painting_mode == "video_inpainting":
            pixel_values_masks = [
                (
                    self._preprocess(
                        images=mask,
                        do_resize=do_resize,
                        size=size,
                        resample=resample,
                        do_center_crop=do_center_crop,
                        crop_size=crop_size,
                        do_normalize=do_normalize,
                        image_mean=image_mean,
                        image_std=image_std,
                        data_format=data_format,
                        input_data_format=input_data_format,
                        is_mask_frame=True,
                        mask_dilation=mask_dilation,
                    )
                    * len(pixel_values[0])
                    if len(mask) == 1
                    else self._preprocess(
                        images=mask,
                        do_resize=do_resize,
                        size=size,
                        resample=resample,
                        do_center_crop=do_center_crop,
                        crop_size=crop_size,
                        do_normalize=do_normalize,
                        image_mean=image_mean,
                        image_std=image_std,
                        data_format=data_format,
                        input_data_format=input_data_format,
                        is_mask_frame=True,
                        mask_dilation=mask_dilation,
                    )
                )
                for mask in masks
            ]
        elif video_painting_mode == "video_outpainting":
            # for outpainting of videos
            pixel_values, flow_masks, masks_dilated = [
                list(pixels)
                for pixels in zip(
                    *[
                        self._extrapolation(
                            video,
                            scale_size=scale_size,
                            data_format=data_format,
                            input_data_format=input_data_format,
                        )
                        for video in pixel_values
                    ]
                )
            ]
        else:
            raise ValueError(f"Unsupported video painting mode: {video_painting_mode}")

        if video_painting_mode == "video_inpainting":
            # masks is for both flow_masks, masks_dilated, just add the same data to both variables in case of inpainting
            flow_masks = masks_dilated = pixel_values_masks

        data = {
            "pixel_values_videos": pixel_values,
            "flow_masks": flow_masks,
            "masks_dilated": masks_dilated,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

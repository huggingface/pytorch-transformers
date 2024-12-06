# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Image processor class for LightGlue."""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from ... import is_torch_available, is_vision_available
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    ImageType,
    PILImageResampling,
    get_image_type,
    infer_channel_dimension_format,
    is_pil_image,
    is_scaled_image,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, logging, requires_backends


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from .modeling_lightglue import LightGlueKeypointMatchingOutput

if is_vision_available():
    import PIL

logger = logging.get_logger(__name__)


# Copied from transformers.models.superpoint.image_processing_superpoint.is_grayscale
def is_grayscale(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    if input_data_format == ChannelDimension.FIRST:
        if image.shape[0] == 1:
            return True
        return np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...])
    elif input_data_format == ChannelDimension.LAST:
        if image.shape[-1] == 1:
            return True
        return np.all(image[..., 0] == image[..., 1]) and np.all(image[..., 1] == image[..., 2])


# Copied from transformers.models.superpoint.image_processing_superpoint.convert_to_grayscale
def convert_to_grayscale(
    image: ImageInput,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> ImageInput:
    """
    Converts an image to grayscale format using the NTSC formula. Only support numpy and PIL Image. TODO support torch
    and tensorflow grayscale conversion

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (Image):
            The image to convert.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
    """
    requires_backends(convert_to_grayscale, ["vision"])

    if isinstance(image, np.ndarray):
        if is_grayscale(image, input_data_format=input_data_format):
            return image
        if input_data_format == ChannelDimension.FIRST:
            gray_image = image[0, ...] * 0.2989 + image[1, ...] * 0.5870 + image[2, ...] * 0.1140
            gray_image = np.stack([gray_image] * 3, axis=0)
        elif input_data_format == ChannelDimension.LAST:
            gray_image = image[..., 0] * 0.2989 + image[..., 1] * 0.5870 + image[..., 2] * 0.1140
            gray_image = np.stack([gray_image] * 3, axis=-1)
        return gray_image

    if not isinstance(image, PIL.Image.Image):
        return image

    image = image.convert("L")
    return image


# Copied from transformers.models.superglue.image_processing_superglue.validate_and_format_image_pairs
def validate_and_format_image_pairs(images: ImageInput):
    error_message = (
        "Input images must be a one of the following :",
        " - A pair of PIL images.",
        " - A pair of 3D arrays.",
        " - A 4D array with shape (2, H, W, C).",
        " - A 5D array with shape (B, 2, H, W, C).",
        " - A list of pairs of PIL images.",
        " - A list of pairs of 3D arrays.",
        " - A list of 4D arrays with shape (2, H, W, C).",
    )

    def _flatten_image_list(image_list):
        """
        Flattens a list of images.
        In the case of images being an array of shape (B, H, W, C), returns a B long list of (H, W, C) images.
        """
        return list(image_list)

    def _flatten_image_list_sequence(image_list_sequence):
        """
        Flattens a list of list of images.
        In the case of images being an array of shape (B, 2, H, W, C), returns a B * 2 long list of (H, W, C) images.
        """
        return [image for image_list in image_list_sequence for image in image_list]

    def _is_pair_of_PIL(images):
        """images is a pair of PIL images."""
        return len(images) == 2 and all(is_pil_image(image) for image in images)

    def _is_3d_array(image):
        """images is a 3D array."""
        return is_valid_image(image) and get_image_type(image) != ImageType.PIL and len(image.shape) == 3

    def _is_pair_of_3d_arrays(images):
        """images is a pair of 3D arrays."""
        return all(_is_3d_array(image) for image in images) and len(images) == 2

    def _is_list_of_images_with_length_different_from_two(images):
        """images is a flat list of either PIL images or 3D arrays but not a pair of images."""
        return (
            all(is_valid_image(image) and (is_pil_image(image) or _is_3d_array(image)) for image in images)
            and len(images) != 2
        )

    def _is_list_of_4d_arrays(images):
        """images is a list of 4D arrays with shape (2, H, W, C)."""
        return all(_is_4d_array(image) for image in images)

    def _is_4d_array(image):
        """images is a 4D array with shape (2, H, W, C)."""
        return is_valid_image(image) and not is_pil_image(image) and len(image.shape) == 4 and image.shape[0] == 2

    def _is_5d_array(images):
        """images is a 5D array with shape (B, 2, H, W, C)."""
        return (
            is_valid_image(images)
            and get_image_type(images) != ImageType.PIL
            and len(images.shape) == 5
            and images.shape[1] == 2
        )

    def _format_image_list(images):
        """
        Function that takes a valid image input and turns it into a list of images.

        A valid image input is one of the following:
        - A pair of PIL images.
        - A pair of 3D arrays.
        - A list of 4D arrays with shape (2, H, W, C).
        - A 5D array with shape (B, 2, H, W, C).

        Raises a ValueError if the input is one of the following :
        - A list of images of length != 2.
        - A single PIL image.
        - A single 3D array.
        """
        if isinstance(images, list):
            if _is_pair_of_PIL(images) or _is_pair_of_3d_arrays(images):
                return images
            if _is_list_of_images_with_length_different_from_two(images):
                raise ValueError(error_message)
            if _is_list_of_4d_arrays(images):
                return _flatten_image_list_sequence(images)
        if is_valid_image(images):
            if is_pil_image(images):
                raise ValueError(error_message)
            if _is_3d_array(images):
                raise ValueError(error_message)
            if _is_4d_array(images):
                return _flatten_image_list(images)
            if _is_5d_array(images):
                return _flatten_image_list_sequence(images)
        return images

    def _is_list_sequence(images):
        """images is a list of lists of images."""
        return isinstance(images, list) and all(isinstance(image, list) for image in images)

    def _is_list_of_pairs(images):
        """images is a list of either pairs of PIL images or pairs of 3D arrays."""
        return all(_is_pair_of_PIL(image) or _is_pair_of_3d_arrays(image) for image in images)

    def _format_image_list_sequence(images):
        """Function that takes an image pair sequence that is either a list of pairs of PIL images or a list of pairs of 3D
        arrays and turns it into a flatten list of images."""
        if _is_list_sequence(images):
            if _is_list_of_pairs(images):
                return _flatten_image_list_sequence(images)
        return images

    def _is_list_of_pil(images):
        """images is a list of PIL images with even length."""
        return (
            all(is_valid_image(image) and get_image_type(image) == ImageType.PIL for image in images)
            and len(images) % 2 == 0
        )

    def _is_list_of_3d_arrays(images):
        """images is a list of 3D arrays with even length."""
        return all(_is_3d_array(image) for image in images) and len(images) % 2 == 0

    def _validate_image_list_format(images):
        """
        Function that validates the format of the output list.

        A valid output is one of the following:
        - A list of PIL images with even length.
        - A list of 3D arrays with even length.

        Raises a ValueError if the output is not one of the above.
        """
        if _is_list_of_pil(images):
            return images
        if _is_list_of_3d_arrays(images):
            return images
        raise ValueError(error_message)

    images = _format_image_list(images)
    images = _format_image_list_sequence(images)
    images = _validate_image_list_format(images)

    return images


class LightGlueImageProcessor(BaseImageProcessor):
    r"""
    Constructs a LightGlue image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be overriden
            by `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 480, "width": 640}`):
            Resolution of the output image after `resize` is applied. Only has an effect if `do_resize` is set to
            `True`. Can be overriden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overriden by `resample` in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overriden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overriden by `rescale_factor` in the `preprocess`
            method.
        do_grayscale (`bool`, *optional*, defaults to `False`):
            Whether to convert the image to grayscale. Can be overriden by `do_grayscale` in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_grayscale: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 480, "width": 640}
        size = get_size_dict(size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_grayscale = do_grayscale

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary of the form `{"height": int, "width": int}`, specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the output image. If not provided, it will be inferred from the input
                image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """

        size = get_size_dict(size, default_to_square=False)

        return resize(
            image,
            size=(size["height"], size["width"]),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # Copied from transformers.models.superglue.image_processing_superglue.SuperGlueImageProcessor.preprocess with SuperGlue->LightGlue
    def preprocess(
        self,
        images,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_grayscale: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image pairs to preprocess. Expects either a list of 2 images or a list of list of 2 images list with
                pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set
                `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the output image after `resize` has been applied. If `size["shortest_edge"]` >= 384, the image
                is resized to `(size["shortest_edge"], size["shortest_edge"])`. Otherwise, the smaller edge of the
                image will be matched to `int(size["shortest_edge"]/ crop_pct)`, after which the image is cropped to
                `(size["shortest_edge"], size["shortest_edge"])`. Only has an effect if `do_resize` is set to `True`.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of `PILImageResampling`, filters. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_grayscale (`bool`, *optional*, defaults to `self.do_grayscale`):
                Whether to convert the image to grayscale.
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
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_grayscale = do_grayscale if do_grayscale is not None else self.do_grayscale

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)

        # Validate and convert the input images into a flattened list of images for all subsequent processing steps.
        images = validate_and_format_image_pairs(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
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

        if do_resize:
            images = [
                self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_grayscale:
            images = [convert_to_grayscale(image, input_data_format=input_data_format) for image in images]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        # Convert back the flattened list of images into a list of pairs of images.
        image_pairs = [images[i : i + 2] for i in range(0, len(images), 2)]

        data = {"pixel_values": image_pairs}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_keypoint_matching(
        self,
        outputs: "LightGlueKeypointMatchingOutput",
        target_sizes: Union[TensorType, List[Tuple]],
        threshold: float = 0.0,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Converts the raw output of [`LightGlueKeypointMatchingOutput`] into lists of keypoints, matches and scores with
        coordinates absolute to the original image sizes.

        Args:
            outputs ([`LightGlueKeypointMatchingOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` or `List[Tuple[Tuple[int, int]]]`, *optional*):
                Tensor of shape `(batch_size, 2, 2)` or list of tuples of tuples (`Tuple[int, int]`) containing the
                target size `(height, width)` of each image in the batch. This must be the original image size (before
                any processing).
            threshold (`float`, *optional*, defaults to 0.0):
                Threshold to filter out the matches with low scores.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the keypoints in the first and second image
            of the pair, the matching scores and the matching indices.
        """
        if outputs.mask.shape[0] != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the mask")
        if not all(len(target_size) == 2 for target_size in target_sizes):
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        if isinstance(target_sizes, List):
            image_pair_sizes = torch.tensor(target_sizes, device=outputs.mask.device)
        else:
            if target_sizes.shape[1] != 2 or target_sizes.shape[2] != 2:
                raise ValueError(
                    "Each element of target_sizes must contain the size (h, w) of each image of the batch"
                )
            image_pair_sizes = target_sizes

        keypoints = outputs.keypoints.clone()
        keypoints = keypoints * image_pair_sizes.flip(-1).reshape(-1, 2, 1, 2)
        keypoints = keypoints.to(torch.int32)

        results = []
        for mask_pair, keypoints_pair, matches, scores in zip(
            outputs.mask, keypoints, outputs.matches[:, 0], outputs.matching_scores[:, 0]
        ):
            mask0 = mask_pair[0] > 0
            mask1 = mask_pair[1] > 0
            keypoints0 = keypoints_pair[0][mask0]
            keypoints1 = keypoints_pair[1][mask1]
            matches0 = matches[mask0]
            scores0 = scores[mask0]

            # Filter out matches with low scores
            valid_matches = torch.logical_and(scores0 > threshold, matches0 != -1)

            matched_keypoints0 = keypoints0[valid_matches]
            matched_keypoints1 = keypoints1[matches0[valid_matches]]
            matching_scores = scores0[valid_matches]

            results.append(
                {
                    "keypoints0": matched_keypoints0,
                    "keypoints1": matched_keypoints1,
                    "matching_scores": matching_scores,
                }
            )

        return results


__all__ = ["LightGlueImageProcessor"]

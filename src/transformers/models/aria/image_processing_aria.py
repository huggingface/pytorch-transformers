#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/aria/modular_aria.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_aria.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
from typing import List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, select_best_resolution
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_valid_image,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched video from {images}")


def divide_to_patches(image: np.array, patch_size: int, input_data_format) -> List[np.array]:
    """
    Divides an image into patches of a specified size.

    Args:
        image (`np.array`):
            The input image.
        patch_size (`int`):
            The size of each patch.
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        list: A list of np.array representing the patches.
    """
    patches = []
    height, width = get_image_size(image, channel_dim=input_data_format)
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            if input_data_format == ChannelDimension.LAST:
                patch = image[i : i + patch_size, j : j + patch_size]
            else:
                patch = image[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)

    return patches


class AriaImageProcessor(BaseImageProcessor):
    """
    A vision processor for the Aria model that handles image preprocessing.
    Initialize the AriaImageProcessor.

    Args:
        max_image_size (`int`, *optional*, defaults to 980):
            Maximum image size.
        min_image_size (`int`, *optional*, defaults to 336):
            Minimum image size.
        image_mean (`list`, *optional*, defaults to [0.5, 0.5, 0.5]):
            Mean values for normalization.
        image_std (`list`, *optional*, defaults to [0.5, 0.5, 0.5]):
            Standard deviation values for normalization.
        split_ratio (`list`, *optional*, defaults to a list of common split ratios as tuples):
            The ratio for splitting the image.
    """

    def __init__(
        self,
        max_image_size=980,
        min_image_size=336,
        image_mean=None,
        image_std=None,
        split_ratio: Optional[List[Tuple[int, int]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if image_mean is None:
            image_mean = [0.5, 0.5, 0.5]
        if image_std is None:
            image_std = [0.5, 0.5, 0.5]
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.image_mean = image_mean
        self.image_std = image_std
        if split_ratio is None:
            self.split_ratio = [
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (1, 6),
                (1, 7),
                (1, 8),
                (2, 4),
                (2, 3),
                (2, 2),
                (2, 1),
                (3, 1),
                (3, 2),
                (4, 1),
                (4, 2),
                (5, 1),
                (6, 1),
                (7, 1),
                (8, 1),
            ]
        else:
            self.split_ratio = split_ratio

    def preprocess(
        self,
        images: Union[ImageInput, List[ImageInput]],
        max_image_size: Optional[int] = None,
        min_image_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        split_image: Optional[bool] = False,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = True,
        do_normalize: Optional[bool] = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Process a list of images.

        Args:
            images (ImageInput or list of ImageInput):
                The input image or a list of images.
            max_image_size (`int`, *optional*, defaults to `self.max_image_size` (980)):
                Maximum image size.
            min_image_size (`int`, *optional*, defaults to `self.min_image_size` (336)):
                Minimum image size.
            return_tensors (`str` or `TensorType`, *optional*, defaults to "pt"):
                The type of tensor to return.
            split_image (`bool`, *optional*, defaults to False):
                Whether to split the image.
            image_mean (`float`, *optional*, defaults to None):
                The mean value of the image.
            image_std (`float`, *optional*, defaults to None):
                The standard deviation of the image.
            do_convert_rgb (`bool`, *optional*, defaults to True):
                Whether to convert the image to RGB.
            do_normalize (`bool`, *optional*, defaults to True):
                Whether to normalize the image.
            resample (PILImageResampling, *optional*, defaults to BICUBIC):
                The resampling filter to use if resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`:
                        image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`:
                        image in (height, width, num_channels) format.
                If unset, will use same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. Can be one of:
                    - `"channels_first"` or `ChannelDimension.FIRST`:
                        image in (num_channels, height, width) format.
                    - `"channels_last"` or `ChannelDimension.LAST`:
                        image in (height, width, num_channels) format.
                If unset, will use the inferred format of the input image.

        Returns:
            BatchFeature:
                A BatchFeature object containing:
                - 'pixel_values':
                    Tensor of processed image pixel values.
                - 'pixel_mask':
                    Boolean pixel mask. This mask is a 2D tensor of shape (max_image_size, max_image_size) where:
                    - True (1) values indicate pixels that belong to the original resized image.
                    - False (0) values indicate pixels that are part of the padding.
                  The mask helps distinguish between actual image content and padded areas in subsequent processing steps.
                - 'num_crops':
                    The maximum number of crops across all images.
        """
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        max_image_size = self.max_image_size if max_image_size is None else max_image_size
        min_image_size = self.min_image_size if min_image_size is None else min_image_size
        if max_image_size not in [490, 980]:
            raise ValueError("max_image_size must be either 490 or 980")

        images = make_batched_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        pixel_values = []
        pixel_masks = []
        num_crops = None

        for image in images:
            if split_image:
                crop_images = self.get_image_patches(
                    image,
                    self.split_ratio,
                    max_image_size,
                    data_format=input_data_format,
                    input_data_format=input_data_format,
                )
            else:
                crop_images = [image]
            if num_crops is None or len(crop_images) > num_crops:
                num_crops = len(crop_images)

            for crop_image in crop_images:
                # At this point the scale is the rescaling factor that would bring the image to max_size in its larger dimension
                h, w = get_image_size(crop_image)
                scale = max_image_size / max(h, w)
                if w >= h:
                    new_size = (max(int(h * scale), min_image_size), max_image_size)  # h, w
                else:
                    new_size = (max_image_size, max(int(w * scale), min_image_size))  # h, w

                crop_image_resized = resize(
                    crop_image,
                    new_size,
                    resample=resample,
                    data_format=input_data_format,
                    input_data_format=input_data_format,
                )

                padding_bottom, padding_right = max_image_size - new_size[0], max_image_size - new_size[1]
                crop_image_padded = pad(
                    crop_image_resized,
                    ((0, padding_bottom), (0, padding_right)),
                    data_format=input_data_format,
                    input_data_format=input_data_format,
                )

                # Create a pixel mask
                pixel_mask = np.zeros((max_image_size, max_image_size), dtype=bool)
                pixel_mask[: new_size[0], : new_size[1]] = 1
                pixel_masks.append(pixel_mask)

                if do_normalize:
                    crop_image_padded = self.normalize(
                        crop_image_padded / 255.0,
                        self.image_mean,
                        self.image_std,
                        data_format=input_data_format,
                        input_data_format=input_data_format,
                    )
                    crop_image_padded = (
                        to_channel_dimension_format(crop_image_padded, data_format, input_data_format)
                        if data_format is not None
                        else crop_image_padded
                    )

                pixel_values.append(crop_image_padded)
        return BatchFeature(
            data={
                "pixel_values": np.stack(pixel_values, axis=0),
                "pixel_mask": np.stack(pixel_masks, axis=0),
                "num_crops": num_crops,
            },
            tensor_type=return_tensors,
        )

    # Modified from models.llava_next.image_preprocessing_llava_next.LlavaNextImageProcessor.get_image_patches
    def get_image_patches(
        self,
        image: np.array,
        grid_pinpoints: List[Tuple[int, int]],
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ) -> List[np.array]:
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image (`np.array`):
                The input image to be processed.
            grid_pinpoints (List[Tuple[int, int]]):
                A list of possible resolutions as tuples.
            patch_size (`int`):
                Size of the patches to divide the image into.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            data_format (`ChannelDimension` or `str`):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            `List[np.array]`: A list of NumPy arrays containing the processed image patches.
        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=resample, input_data_format=input_data_format
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution, input_data_format=input_data_format)

        patches = divide_to_patches(padded_image, patch_size=patch_size, input_data_format=input_data_format)

        # make sure that all patches are in the input data format
        patches = [
            to_channel_dimension_format(patch, channel_dim=data_format, input_channel_dim=input_data_format)
            for patch in patches
        ]
        return patches


__all__ = ["AriaImageProcessor"]

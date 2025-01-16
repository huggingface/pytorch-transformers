# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    PaddingMode,
    get_image_size,
    pad,
    resize,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_vision_available,
    make_nested_list_of_images,
    to_numpy_array,
    validate_preprocess_arguments,
)
from ...utils import TensorType, logging


if is_vision_available():
    import PIL
    from PIL import Image


logger = logging.get_logger(__name__)


@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(max_image_tiles: int) -> List[Tuple[int, int]]:
    """
    Computes all allowed aspect ratios for a given maximum number of input tiles.

    This function calculates all possible arrangements of tiles that can be formed
    within the constraint of the maximum number of tiles. Each arrangement is
    represented by its aspect ratio (width/height) and the corresponding tile configuration.

    Args:
        max_image_tiles (`int`):
            The maximum number of tiles allowed.

    Returns:
        `List[Tuple[int, int]]`: A list of tuples, each tuple representing a valid (width, height)
        configuration in terms of number of tiles.

    Example:
        >>> get_all_supported_aspect_ratios(4)
        [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (4, 1)]

    """
    aspect_ratios = []
    for width in range(1, max_image_tiles + 1):
        for height in range(1, max_image_tiles + 1):
            if width * height <= max_image_tiles:
                aspect_ratios.append((width, height))
    return aspect_ratios


def get_image_size_fit_to_canvas(
    image_height: int,
    image_width: int,
    canvas_height: int,
    canvas_width: int,
    tile_size: int,
) -> Tuple[int, int]:
    """
    Calculates the new size of an image to fit within a canvas while maintaining aspect ratio.

    This function calculates the optimal size for an image to fit within a canvas defined by
    canvas_height and canvas_width, while ensuring that the image dimensions are not smaller than
    tile_size. If the image is larger than the canvas, the returned size will fit within the canvas.
    If the image already fits within the canvas, the size remains unchanged.
    The aspect ratio of the original image is preserved.

    Args:
        image_height (`int`):
            The height of the original image.
        image_width (`int`):
            The width of the original image.
        canvas_height (`int`):
            The height of the canvas.
        canvas_width (`int`):
            The width of the canvas.
        tile_size (`int`):
            The tile size.

    Returns:
        `Tuple[int, int]`: A tuple containing the new height and width of the image.

    """
    # Set target image size in between `tile_size` and canvas_size
    target_width = np.clip(image_width, tile_size, canvas_width)
    target_height = np.clip(image_height, tile_size, canvas_height)

    scale_h = target_height / image_height
    scale_w = target_width / image_width

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.floor(image_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.floor(image_width * scale_h), target_width)

    return new_height, new_width


@lru_cache(maxsize=100)
def get_optimal_tiled_canvas(
    image_height: int,
    image_width: int,
    max_image_tiles: int,
    tile_size: int,
) -> Tuple[int, int]:
    r"""
    Determines the best canvas based on image and tile size and maximum number of tiles.

    First, calculates possible resolutions based on the maximum number of tiles and tile size.
    For example for max_image_tiles=2, tile_size=100, possible tile arrangements are:
    [(1, 1), (1, 2), (2, 1)] and corresponding canvas sizes are:
    [(100, 100), (100, 200), (200, 100)]

    For each possible resolution, calculates the scaling factors for
    width and height, and selects the smallest one, which is the limiting side.
    E.g. to match the canvas you can upscale height by 2x, and width by 1.5x,
    therefore, the maximum upscaling you can do is min(2, 1.5) = 1.5.

    If upscaling is possible (any of the scaling factors is greater than 1),
    then picks the smallest upscaling factor > 1.

    If upscaling is not possible, then picks the largest scaling factor <= 1, i.e.
    reduce downscaling as much as possible.

    If there are multiple resolutions with the same max scale, we pick the one with the lowest area,
    to minimize padding. E.g., the same image can be upscaled to 224x224 and 224x448, but the latter
    has more padding.

    Example of canvases made from tiles:

    To visualize how the image can fit onto different tile grids, let's try fitting an ASCII cat into the tiles.

    Here's an ASCII cat image you want to fit into the tiles:

       /\_/\
      ( o.o )
       > ^ <

    If `num_tiles=6`, possible tile grids would look like this:

    **2x3 Canvas (2 tiles wide, 3 tiles tall)**: -> total of 6 tiles
    +-------+-------+
    | /\_/\ |   0   |   <- Cat image split across two tiles horizontally
    +-------+-------+
    | > ^ < |   0   |   <- Remaining part of the cat occupies the left tile
    +-------+-------+
    |( o.o )|   0   |
    +-------+-------+

    **3x2 Canvas (3 tiles wide, 2 tiles tall)**: -> total of 6 tiles
    +-------+-------+-------+
    | /\_/\ |( o.o )|   0   |   <- Cat image occupies the first two tiles, 1 tile remains empty
    +-------+-------+-------+
    | > ^ < |   0   |   0   |   <- Remaining part of the cat occupies the left tile
    +-------+-------+-------+

    **1x6 Canvas (1 tile wide, 6 tiles tall)**: -> total of 6 tiles
    +-------+
    | /\_/\ |   <- Top part of the cat
    +-------+
    |( o.o )|   <- Middle part of the cat
    +-------+
    | > ^ < |   <- Bottom part of the cat
    +-------+
    |   0   |
    +-------+
    |   0   |
    +-------+
    |   0   |
    +-------+

    Given that the tiles you get depend on the chosen aspect ratio, you have to add
    embedding in the modeling code to help it know if it got a 3x2 or a 1x6 or a 2x3
    aspect ratio.

    The function tests these arrangements to find the smallest canvas where the image fits.
    If multiple canvases fit, it selects the one where the dimensions are closest to the image size.

    In this case the first canvas is the closest to the original image.

    You then feed all of the tiles to the model:

        +-------+-------+-------+-------+-------+-------+
    -   | /\_/\ |( o.o )| > ^ < |   0   |   0   |   0   |  <- Last canvas
        +-------+-------+-------+-------+-------+-------+

        +-------+-------+-------+-------+-------+-------+
    -   | /\_/\ | 0     |( o.o )|   0   | > ^ < |   0   | <- First canvas
        +-------+-------+-------+-------+-------+-------+

        +-------+-------+-------+-------+-------+-------+
    -   | /\_/\ |( o.o )|   0   | > ^ < |   0   |   0   | <- second canvas
        +-------+-------+-------+-------+-------+-------+

    For each tile, you have num_channels (usually RGB so 3), tile_width, tile_height

    Args:
        image_height (`int`):
            The height of the image.
        image_width (`int`):
            The width of the image.
        max_image_tiles (`int`):
            The maximum number of tiles any image can be split into.
        tile_size (`int`):
            The tile size.

    Returns:
        `Tuple[int, int]`: The best canvas resolution [height, width] for the given image.
    """
    possible_tile_arrangements = get_all_supported_aspect_ratios(max_image_tiles)
    possible_canvas_sizes = np.array(possible_tile_arrangements) * tile_size

    # get all possible resolutions heights/widths
    target_heights, target_widths = np.array(possible_canvas_sizes).T

    # get scaling factors to resize the image without distortion
    scale_h = target_heights / image_height
    scale_w = target_widths / image_width

    # get the min scale between width and height (limiting side -> no distortion)
    scales = np.where(scale_w > scale_h, scale_h, scale_w)

    # filter only scales that allow upscaling
    upscaling_options = scales[scales >= 1]
    if len(upscaling_options) > 0:
        selected_scale = np.min(upscaling_options)
    else:
        # no upscaling possible,
        # get the minimum downscaling (max scale for scales<1)
        downscaling_options = scales[scales < 1]
        selected_scale = np.max(downscaling_options)

    # get all resolutions that support this scaling factor,
    # e.g. you can upscale to 224x224, 224x448, 224x672 without distortion
    chosen_canvas = possible_canvas_sizes[scales == selected_scale]

    # if there are multiple resolutions,
    # get the one with minimum area to reduce padding
    if len(chosen_canvas) > 1:
        areas = chosen_canvas[:, 0] * chosen_canvas[:, 1]
        optimal_idx = np.argmin(areas)
        optimal_canvas = chosen_canvas[optimal_idx]
    else:
        optimal_canvas = chosen_canvas[0]

    return optimal_canvas


def split_to_tiles(image: np.ndarray, num_tiles_height: int, num_tiles_width: int) -> np.ndarray:
    """
    Split an image into a specified number of tiles along its width and height dimensions.

    Args:
        image (`np.ndarray`):
            Input image with shape (num_channels, height, width).
        num_tiles_height (`int`):
            Number of tiles to split the image into along its height.
        num_tiles_width (`int`):
            Number of tiles to split the image into along its width.

    Returns:
        `np.ndarray`:
            Array of image tiles with shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width).
    """
    num_channels, height, width = image.shape
    tile_height = height // num_tiles_height
    tile_width = width // num_tiles_width

    image = image.reshape(num_channels, num_tiles_height, tile_height, num_tiles_width, tile_width)

    # Permute to (num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
    image = image.transpose(1, 3, 0, 2, 4)

    # Reshape into the desired output shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)
    image = image.reshape(num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)

    return np.ascontiguousarray(image)


def build_aspect_ratio_mask(aspect_ratios: List[List[Tuple[int, int]]], max_image_tiles: int) -> np.ndarray:
    """
    Builds a mask for the aspect ratios of the images.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of lists containing aspect ratios for each image in the batch.
            Each aspect ratio is represented as a tuple of (width, height) in terms of number of tiles.
        max_image_tiles (`int`):
            The maximum number of tiles any image can be split into.

    Returns:
        `np.ndarray`: A 3D numpy array of shape (batch_size, max_num_images, max_image_tiles).
            The mask contains 1s for valid tiles and 0s for padding.
    """
    batch_size = len(aspect_ratios)
    max_num_images = max([len(row) for row in aspect_ratios])

    aspect_ratio_mask = np.zeros((batch_size, max_num_images, max_image_tiles), dtype=np.int64)

    # Set the first tile to 1 for all aspect ratios
    # because in original implementation aspect ratios are padded with (1, 1),
    # but original code examples are not built to handle batches, so we might remove it later
    aspect_ratio_mask[:, :, 0] = 1

    # Set the aspect ratio mask for the rest of the tiles
    for i, sample_aspect_ratios in enumerate(aspect_ratios):
        for j, (num_tiles_w, num_tiles_h) in enumerate(sample_aspect_ratios):
            aspect_ratio_mask[i, j, : num_tiles_w * num_tiles_h] = 1

    return aspect_ratio_mask


def pack_images(
    batch_images: List[List[np.ndarray]],
    max_image_tiles: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Stack a list of lists of images with variable lengths into a numpy array, applying zero padding as needed.
    Each list in the input represents a batch sample, and each image within a list is expected to be
    pre-split into tiles. The resulting array will have a shape of
    (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width).

    Args:
        batch_images (`List[List[np.ndarray]]`):
            A list of lists of image tiles. Each inner list represents
            a batch sample containing multiple images, where each image is pre-split into tiles.
            The shape of each tile array is (num_tiles, channels, tile_height, tile_width).
        max_image_tiles (int):
            The maximum number of tiles any image was potantially split.

    Returns:
        `Tuple[np.ndarray, List[List[int]]]`: A tuple containing:
            - stacked_images (`np.ndarray`):
                A numpy array of stacked images with shape
                (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width).
            - all_num_tiles (`List[List[int]]`):
                A list of lists containing the number of tiles
                for each image in each batch sample.
    """

    # Determine output shape
    batch_size = len(batch_images)
    max_num_images = max([len(images) for images in batch_images])
    shapes = [image.shape for images in batch_images for image in images]
    _, channels, tile_height, tile_width = shapes[0]

    # Initialize the stacked images array with zeros
    stacked_images = np.zeros(
        (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width),
        dtype=np.float32,
    )

    # Fill the stacked images array with the tiled images from the batch
    all_num_tiles = []
    for i, images in enumerate(batch_images):
        num_sample_tiles = []
        for j, image in enumerate(images):
            num_tiles = image.shape[0]
            stacked_images[i, j, :num_tiles] = image
            num_sample_tiles.append(num_tiles)
        all_num_tiles.append(num_sample_tiles)

    return stacked_images, all_num_tiles


def pack_aspect_ratios(aspect_ratios: List[List[Tuple[int, int]]], pad_value: int = 1) -> np.ndarray:
    """
    Stack a list of aspect ratios into a numpy array.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of aspect ratios.
        pad_value (`int`, *optional*, defaults to 1):
            The value to pad the aspect ratios with.

    Returns:
        `np.ndarray`:
            The aspect ratios stacked into a numpy array with shape (batch_size, max_num_images, 2).
    """
    batch_size = len(aspect_ratios)
    max_num_images = max([len(row) for row in aspect_ratios])

    aspect_ratios_stacked = np.full((batch_size, max_num_images, 2), pad_value, dtype=np.int64)
    for i, row in enumerate(aspect_ratios):
        if len(row) > 0:
            aspect_ratios_stacked[i, : len(row)] = np.array(row)
    return aspect_ratios_stacked


def convert_aspect_ratios_to_ids(aspect_ratios: List[List[Tuple[int, int]]], max_image_tiles: int) -> np.ndarray:
    """
    Convert aspect ratio tuples to unique ids.

    For batch padding we use 0, because there might be different number of images in each batch.
    The aspect ratio ids start from 1, with 1 corresponding to the first supported aspect ratio.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of aspect ratios for each image in the batch.
        max_image_tiles (`int`):
            The maximum number of tiles any image can be split into.

    Returns:
        `np.ndarray`:
            The aspect ratios ids as a numpy array with shape (batch_size, max_num_images).
            Each id corresponds to the index of the aspect ratio in the list of supported aspect ratios,
            offset by 1 (so 0 can be used for padding).
    """

    batch_size = len(aspect_ratios)
    max_num_images = max([len(row) for row in aspect_ratios])
    supported_aspect_ratios = get_all_supported_aspect_ratios(max_image_tiles)

    aspect_ratios_ids = np.zeros((batch_size, max_num_images), dtype=np.int64)
    for i, sample_aspect_ratios in enumerate(aspect_ratios):
        for j, (num_tiles_h, num_tiles_w) in enumerate(sample_aspect_ratios):
            aspect_ratios_ids[i, j] = supported_aspect_ratios.index((num_tiles_h, num_tiles_w)) + 1
    return aspect_ratios_ids


def to_channel_dimension_format(
    image: np.ndarray,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> np.ndarray:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`.

    Args:
        image (`numpy.ndarray`):
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `np.ndarray`:
            The image with the channel dimension set to `channel_dim`.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.transpose((2, 0, 1))
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.transpose((1, 2, 0))
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return image


# Copied from transformers.models.idefics2.image_processing_idefics2.convert_to_rgb
def convert_to_rgb(image: ImageInput) -> ImageInput:
    """
    Converts an image to RGB format. Only converts if the image is of type PIL.Image.Image, otherwise returns the image
    as is.
    Args:
        image (Image):
            The image to convert.
    """
    if not isinstance(image, PIL.Image.Image):
        return image

    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


def _validate_size(size: Dict[str, int]) -> None:
    if not ("height" in size and "width" in size):
        raise ValueError(f"Argument `size` must be a dictionary with keys 'height' and 'width'. Got: {size}")
    if size["height"] != size["width"]:
        raise ValueError(f"Argument `size` must have the same height and width, got {size}")


def _validate_mllama_preprocess_arguments(do_resize, size, do_pad, max_image_tiles):
    if not do_pad:
        raise ValueError("MllamaImageProcessor doesn't support `do_pad=False` mode.")
    if not do_resize:
        raise ValueError("MllamaImageProcessor doesn't support `do_resize=False` mode.")
    if max_image_tiles is None or max_image_tiles <= 0:
        raise ValueError(f"MllamaImageProcessor `max_image_tiles` must be a positive integer, got {max_image_tiles}.")
    _validate_size(size)


class MllamaImageProcessor(BaseImageProcessor):
    """
    Constructs a Mllama image processor.

    Args:
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB. This is useful if the input image is of a different format e.g. RGBA.
            Only has an effect if the input image is in the PIL format.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Size of the image tile. Should be a dictionary containing 'height' and 'width' keys, both with integer values.
            The height and width values should be equal.
        resample (`int`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
            has an effect if `do_resize` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image.
        rescale_factor (`float`, *optional*, defaults to 0.0):
            Rescale factor to rescale the image by if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
            `True`.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether or not to pad the images to the largest height and width in the batch.
        max_image_tiles (`int`, *optional*, defaults to 4):
            The maximum number of tiles to split the image into.
    """

    model_input_names = ["pixel_values", "num_tiles", "aspect_ratio_ids", "aspect_ratio_mask"]

    def __init__(
        self,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        max_image_tiles: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_convert_rgb = do_convert_rgb
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 224, "width": 224}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_pad = do_pad
        self.max_image_tiles = max_image_tiles

        _validate_mllama_preprocess_arguments(self.do_resize, self.size, self.do_pad, self.max_image_tiles)

    def preprocess(
        self,
        images: ImageInput,
        do_convert_rgb: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        max_image_tiles: Optional[int] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        """
        Preprocess a batch of images.

        Args:
            images (`ImageInput`):
                A list of images to preprocess.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image tile. Should be a dictionary containing 'height' and 'width' keys, both with integer values.
                The height and width values should be equal.
            resample (`int`, *optional*, defaults to `self.resample`):
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
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether or not to pad the images to the largest height and width in the batch.
            max_image_tiles (`int`, *optional*, defaults to `self.max_image_tiles`):
                The maximum number of tiles to split the image into.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.

        Returns:
            `BatchFeature` of the following structure:
                - **pixel_values** (`TensorType`): The preprocessed pixel values.
                - **aspect_ratio_ids** (`TensorType`): The aspect ratio ids of the images.
                - **num_tiles** (`List[List[int]]`): The number of tiles for each image in the batch.
        """
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        max_image_tiles = max_image_tiles if max_image_tiles is not None else self.max_image_tiles

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

        # extra validation
        _validate_mllama_preprocess_arguments(do_resize, size, do_pad, max_image_tiles)

        images_list = make_nested_list_of_images(images)

        if self.do_convert_rgb:
            images_list = [[convert_to_rgb(image) for image in images] for images in images_list]

        images_list = [[to_numpy_array(image) for image in images] for images in images_list]

        batch_images = []
        batch_aspect_ratios = []

        # iterate over batch samples
        for images in images_list:
            sample_images = []
            sample_aspect_ratios = []

            # iterate over images in a batch sample
            for image in images:
                # convert images to channels first format for faster processing
                # LAST is slower for `pad` and not supported by `split_to_tiles`
                data_format = ChannelDimension.FIRST
                image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

                # do_resize=False is not supported, validated
                image, aspect_ratio = self.resize(
                    image=image,
                    size=size,
                    resample=resample,
                    max_image_tiles=max_image_tiles,
                    input_data_format=data_format,
                    data_format=data_format,
                )

                # do_pad=False is not supported, validated
                image = self.pad(
                    image=image,
                    size=size,
                    aspect_ratio=aspect_ratio,
                    input_data_format=data_format,
                    data_format=data_format,
                )

                if do_rescale:
                    image = self.rescale(
                        image=image,
                        scale=rescale_factor,
                        input_data_format=input_data_format,
                        data_format=data_format,
                    )

                if do_normalize:
                    image = self.normalize(
                        image=image,
                        mean=image_mean,
                        std=image_std,
                        input_data_format=input_data_format,
                        data_format=data_format,
                    )

                num_tiles_height, num_tiles_width = aspect_ratio
                image = split_to_tiles(image, num_tiles_height, num_tiles_width)

                sample_images.append(image)
                sample_aspect_ratios.append((num_tiles_height, num_tiles_width))

            batch_images.append(sample_images)
            batch_aspect_ratios.append(sample_aspect_ratios)

        images, num_tiles = pack_images(batch_images, max_image_tiles)

        aspect_ratio_ids = convert_aspect_ratios_to_ids(batch_aspect_ratios, max_image_tiles=max_image_tiles)
        aspect_ratio_mask = build_aspect_ratio_mask(batch_aspect_ratios, max_image_tiles=max_image_tiles)

        # images (np.ndarray) with shape (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
        # aspect_ratio_ids (np.ndarray) with shape (batch_size, max_num_images) - aspect ratio ids for each image, padded to max_num_images with 0
        # num_tiles (List[List[int]]) with (batch_size, num_images_in_batch) - real number of tiles for each image, not padded
        # aspect_ratio_mask (np.ndarray) with shape (batch_size, max_num_images, max_image_tiles) - number of tiles for each image, padded to max_num_images with 0
        encoded_inputs = BatchFeature(
            data={
                "pixel_values": images,
                "aspect_ratio_ids": aspect_ratio_ids,
                "aspect_ratio_mask": aspect_ratio_mask,
            },
            tensor_type=return_tensors,
        )
        encoded_inputs["num_tiles"] = num_tiles

        return encoded_inputs

    def pad(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        aspect_ratio: Tuple[int, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image to the `size` x `aspect_ratio`. For example, if size is {height: 224, width: 224} and aspect ratio is
        (1, 2), the image will be padded to 224x448.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            aspect_ratio (`Tuple[int, int]`):
                The aspect ratio of the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `np.ndarray`: The padded image.
        """

        _validate_size(size)

        image_height, image_width = get_image_size(image, channel_dim=input_data_format)
        num_tiles_height, num_tiles_width = aspect_ratio
        padded_height = num_tiles_height * size["height"]
        padded_width = num_tiles_width * size["width"]
        pad_size = ((0, padded_height - image_height), (0, padded_width - image_width))

        image = pad(
            image,
            pad_size,
            mode=PaddingMode.CONSTANT,
            constant_values=0,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        return image

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        max_image_tiles: int,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Union[np.ndarray, Tuple[int, int]]:
        """
        Resizes an image to fit within a tiled canvas while maintaining its aspect ratio.
        The optimal canvas size is calculated based on the maximum number of tiles and the tile size.

        The function first determines the best tile arrangement for the image, then resizes the image
        to fit within this canvas. The resized image and the number of tiles along the height and width
        dimensions are returned.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            max_image_tiles (`int`):
                The maximum number of tiles to split the image into.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `Union[np.ndarray, Tuple[int, int]]`: The resized image and a tuple containing the number of tiles
            along the height and width dimensions.
        """

        _validate_size(size)

        image_height, image_width = get_image_size(image, channel_dim=input_data_format)
        tile_size = size["height"]

        canvas_height, canvas_width = get_optimal_tiled_canvas(
            image_height=image_height,
            image_width=image_width,
            max_image_tiles=max_image_tiles,
            tile_size=tile_size,
        )
        num_tiles_height = canvas_height // tile_size
        num_tiles_width = canvas_width // tile_size

        new_height, new_width = get_image_size_fit_to_canvas(
            image_height=image_height,
            image_width=image_width,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            tile_size=tile_size,
        )

        image = resize(
            image,
            (new_height, new_width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        return image, (num_tiles_height, num_tiles_width)


__all__ = ["MllamaImageProcessor"]

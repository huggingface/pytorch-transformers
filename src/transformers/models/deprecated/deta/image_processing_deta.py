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
"""Image processor class for Deformable DETR."""

import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ....feature_extraction_utils import BatchFeature
from ....image_processing_utils import BaseImageProcessor, get_size_dict
from ....image_transforms import (
    PaddingMode,
    center_to_corners_format,
    corners_to_center_format,
    pad,
    rescale,
    resize,
    rgb_to_id,
    to_channel_dimension_format,
)
from ....image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    AnnotationFormat,
    AnnotationType,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_batched,
    is_scaled_image,
    to_numpy_array,
    valid_images,
    validate_annotations,
    validate_preprocess_arguments,
)
from ....utils import (
    is_flax_available,
    is_jax_tensor,
    is_tf_available,
    is_tf_tensor,
    is_torch_available,
    is_torch_tensor,
    is_torchvision_available,
    is_vision_available,
    logging,
)
from ....utils.generic import TensorType


if is_torch_available():
    import torch


if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms

if is_vision_available():
    import PIL


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)


def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`Tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    height, width = image_size
    raw_size = None
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            raw_size = max_size * min_original_size / max_original_size
            size = int(round(raw_size))

    if (height <= width and height == size) or (width <= height and width == size):
        oh, ow = height, width
    elif width < height:
        ow = size
        if max_size is not None and raw_size is not None:
            oh = int(raw_size * height / width)
        else:
            oh = int(size * height / width)
    else:
        oh = size
        if max_size is not None and raw_size is not None:
            ow = int(raw_size * width / height)
        else:
            ow = int(size * width / height)

    return (oh, ow)


def get_resize_output_image_size(
    input_image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int]],
    max_size: Optional[int] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size. If the desired output size
    is a tuple or list, the output image size is returned as is. If the desired output size is an integer, the output
    image size is computed by keeping the aspect ratio of the input image size.

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        size (`int` or `Tuple[int, int]` or `List[int]`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.
    """
    image_size = get_image_size(input_image, input_data_format)
    if isinstance(size, (list, tuple)):
        return size

    return get_size_with_aspect_ratio(image_size, size, max_size)


def get_image_size_for_max_height_width(
    input_image: np.ndarray,
    max_height: int,
    max_width: int,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    Computes the output image size given the input image and the maximum allowed height and width. Keep aspect ratio.
    Important, even if image_height < max_height and image_width < max_width, the image will be resized
    to at least one of the edges be equal to max_height or max_width.

    For example:
        - input_size: (100, 200), max_height: 50, max_width: 50 -> output_size: (25, 50)
        - input_size: (100, 200), max_height: 200, max_width: 500 -> output_size: (200, 400)

    Args:
        input_image (`np.ndarray`):
            The image to resize.
        max_height (`int`):
            The maximum allowed height.
        max_width (`int`):
            The maximum allowed width.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.
    """
    image_size = get_image_size(input_image, input_data_format)
    height, width = image_size
    height_scale = max_height / height
    width_scale = max_width / width
    min_scale = min(height_scale, width_scale)
    new_height = int(height * min_scale)
    new_width = int(width * min_scale)
    return new_height, new_width


def get_numpy_to_framework_fn(arr) -> Callable:
    """
    Returns a function that converts a numpy array to the framework of the input array.

    Args:
        arr (`np.ndarray`): The array to convert.
    """
    if isinstance(arr, np.ndarray):
        return np.array
    if is_tf_available() and is_tf_tensor(arr):
        import tensorflow as tf

        return tf.convert_to_tensor
    if is_torch_available() and is_torch_tensor(arr):
        import torch

        return torch.tensor
    if is_flax_available() and is_jax_tensor(arr):
        import jax.numpy as jnp

        return jnp.array
    raise ValueError(f"Cannot convert arrays of type {type(arr)}")


def safe_squeeze(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Squeezes an array, but only if the axis specified has dim 1.
    """
    if axis is None:
        return arr.squeeze()

    try:
        return arr.squeeze(axis=axis)
    except ValueError:
        return arr


def normalize_annotation(annotation: Dict, image_size: Tuple[int, int]) -> Dict:
    image_height, image_width = image_size
    norm_annotation = {}
    for key, value in annotation.items():
        if key == "boxes":
            boxes = value
            boxes = corners_to_center_format(boxes)
            boxes /= np.asarray([image_width, image_height, image_width, image_height], dtype=np.float32)
            norm_annotation[key] = boxes
        else:
            norm_annotation[key] = value
    return norm_annotation


def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)


def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


def convert_coco_poly_to_mask(segmentations, height: int, width: int) -> np.ndarray:
    """
    Convert a COCO polygon annotation to a mask.

    Args:
        segmentations (`List[List[float]]`):
            List of polygons, each polygon represented by a list of x-y coordinates.
        height (`int`):
            Height of the mask.
        width (`int`):
            Width of the mask.
    """
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError("Pycocotools is not installed in your environment.")

    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = np.asarray(mask, dtype=np.uint8)
        mask = np.any(mask, axis=2)
        masks.append(mask)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)

    return masks


def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """
    Convert the target in COCO format into the format expected by DETA.
    """
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    # Get all COCO annotations for the given image.
    annotations = target["annotations"]
    annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

    classes = [obj["category_id"] for obj in annotations]
    classes = np.asarray(classes, dtype=np.int64)

    # for conversion to coco api
    area = np.asarray([obj["area"] for obj in annotations], dtype=np.float32)
    iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations], dtype=np.int64)

    boxes = [obj["bbox"] for obj in annotations]
    # guard against no boxes via resizing
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    new_target = {}
    new_target["image_id"] = image_id
    new_target["class_labels"] = classes[keep]
    new_target["boxes"] = boxes[keep]
    new_target["area"] = area[keep]
    new_target["iscrowd"] = iscrowd[keep]
    new_target["orig_size"] = np.asarray([int(image_height), int(image_width)], dtype=np.int64)

    if annotations and "keypoints" in annotations[0]:
        keypoints = [obj["keypoints"] for obj in annotations]
        # Converting the filtered keypoints list to a numpy array
        keypoints = np.asarray(keypoints, dtype=np.float32)
        # Apply the keep mask here to filter the relevant annotations
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    if return_segmentation_masks:
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width)
        new_target["masks"] = masks[keep]

    return new_target


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    Compute the bounding boxes around the provided panoptic segmentation masks.

    Args:
        masks: masks in format `[number_masks, height, width]` where N is the number of masks

    Returns:
        boxes: bounding boxes in format `[number_masks, 4]` in xyxy format
    """
    if masks.size == 0:
        return np.zeros((0, 4))

    h, w = masks.shape[-2:]
    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    # see https://github.com/pytorch/pytorch/issues/50276
    y, x = np.meshgrid(y, x, indexing="ij")

    x_mask = masks * np.expand_dims(x, axis=0)
    x_max = x_mask.reshape(x_mask.shape[0], -1).max(-1)
    x = np.ma.array(x_mask, mask=~(np.array(masks, dtype=bool)))
    x_min = x.filled(fill_value=1e8)
    x_min = x_min.reshape(x_min.shape[0], -1).min(-1)

    y_mask = masks * np.expand_dims(y, axis=0)
    y_max = y_mask.reshape(x_mask.shape[0], -1).max(-1)
    y = np.ma.array(y_mask, mask=~(np.array(masks, dtype=bool)))
    y_min = y.filled(fill_value=1e8)
    y_min = y_min.reshape(y_min.shape[0], -1).min(-1)

    return np.stack([x_min, y_min, x_max, y_max], 1)


def prepare_coco_panoptic_annotation(
    image: np.ndarray,
    target: Dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> Dict:
    """
    Prepare a coco panoptic annotation for DETA.
    """
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    new_target = {}
    new_target["image_id"] = np.asarray([target["image_id"] if "image_id" in target else target["id"]], dtype=np.int64)
    new_target["size"] = np.asarray([image_height, image_width], dtype=np.int64)
    new_target["orig_size"] = np.asarray([image_height, image_width], dtype=np.int64)

    if "segments_info" in target:
        masks = np.asarray(PIL.Image.open(annotation_path), dtype=np.uint32)
        masks = rgb_to_id(masks)

        ids = np.array([segment_info["id"] for segment_info in target["segments_info"]])
        masks = masks == ids[:, None, None]
        masks = masks.astype(np.uint8)
        if return_masks:
            new_target["masks"] = masks
        new_target["boxes"] = masks_to_boxes(masks)
        new_target["class_labels"] = np.array(
            [segment_info["category_id"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        new_target["iscrowd"] = np.asarray(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]], dtype=np.int64
        )
        new_target["area"] = np.asarray(
            [segment_info["area"] for segment_info in target["segments_info"]], dtype=np.float32
        )

    return new_target


def resize_annotation(
    annotation: Dict[str, Any],
    orig_size: Tuple[int, int],
    target_size: Tuple[int, int],
    threshold: float = 0.5,
    resample: PILImageResampling = PILImageResampling.NEAREST,
):
    """
    Resizes an annotation to a target size.

    Args:
        annotation (`Dict[str, Any]`):
            The annotation dictionary.
        orig_size (`Tuple[int, int]`):
            The original size of the input image.
        target_size (`Tuple[int, int]`):
            The target size of the image, as returned by the preprocessing `resize` step.
        threshold (`float`, *optional*, defaults to 0.5):
            The threshold used to binarize the segmentation masks.
        resample (`PILImageResampling`, defaults to `PILImageResampling.NEAREST`):
            The resampling filter to use when resizing the masks.
    """
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size))
    ratio_height, ratio_width = ratios

    new_annotation = {}
    new_annotation["size"] = target_size

    for key, value in annotation.items():
        if key == "boxes":
            boxes = value
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            new_annotation["boxes"] = scaled_boxes
        elif key == "area":
            area = value
            scaled_area = area * (ratio_width * ratio_height)
            new_annotation["area"] = scaled_area
        elif key == "masks":
            masks = value[:, None]
            masks = np.array([resize(mask, target_size, resample=resample) for mask in masks])
            masks = masks.astype(np.float32)
            masks = masks[:, 0] > threshold
            new_annotation["masks"] = masks
        elif key == "size":
            new_annotation["size"] = target_size
        else:
            new_annotation[key] = value

    return new_annotation


class DetaImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Deformable DETR image processor.

    Args:
        format (`str`, *optional*, defaults to `"coco_detection"`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be
            overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 800, "longest_edge": 1333}`):
            Size of the image's `(height, width)` dimensions after resizing. Can be overridden by the `size` parameter
            in the `preprocess` method. Available options are:
                - `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
                    Do NOT keep the aspect ratio.
                - `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
                    the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
                    less or equal to `longest_edge`.
                - `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
                    aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
                    `max_width`.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Controls whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
            `preprocess` method.
        do_normalize:
            Controls whether to normalize the image. Can be overridden by the `do_normalize` parameter in the
            `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
            channel. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
            for each channel. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_annotations (`bool`, *optional*, defaults to `True`):
            Controls whether to convert the annotations to the format expected by the DETR model. Converts the
            bounding boxes to the format `(center_x, center_y, width, height)` and in the range `[0, 1]`.
            Can be overridden by the `do_convert_annotations` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Controls whether to pad the image. Can be overridden by the `do_pad` parameter in the `preprocess`
            method. If `True`, padding will be applied to the bottom and right of the image with zeros.
            If `pad_size` is provided, the image will be padded to the specified dimensions.
            Otherwise, the image will be padded to the maximum height and width of the batch.
        pad_size (`Dict[str, int]`, *optional*):
            The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
            provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
            height and width in the batch.
    """

    model_input_names = ["pixel_values", "pixel_mask"]

    def __init__(
        self,
        format: Union[str, AnnotationFormat] = AnnotationFormat.COCO_DETECTION,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        do_convert_annotations: bool = True,
        do_pad: bool = True,
        pad_size: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> None:
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        size = get_size_dict(size, default_to_square=False)

        if do_convert_annotations is None:
            do_convert_annotations = do_normalize

        super().__init__(**kwargs)
        self.format = format
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_convert_annotations = do_convert_annotations
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
        self.pad_size = pad_size

    def prepare_annotation(
        self,
        image: np.ndarray,
        target: Dict,
        format: Optional[AnnotationFormat] = None,
        return_segmentation_masks: bool = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Dict:
        """
        Prepare an annotation for feeding into DETA model.
        """
        format = format if format is not None else self.format

        if format == AnnotationFormat.COCO_DETECTION:
            return_segmentation_masks = False if return_segmentation_masks is None else return_segmentation_masks
            target = prepare_coco_detection_annotation(
                image, target, return_segmentation_masks, input_data_format=input_data_format
            )
        elif format == AnnotationFormat.COCO_PANOPTIC:
            return_segmentation_masks = True if return_segmentation_masks is None else return_segmentation_masks
            target = prepare_coco_panoptic_annotation(
                image,
                target,
                masks_path=masks_path,
                return_masks=return_segmentation_masks,
                input_data_format=input_data_format,
            )
        else:
            raise ValueError(f"Format {format} is not supported.")
        return target

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the image's `(height, width)` dimensions after resizing. Available options are:
                    - `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
                        Do NOT keep the aspect ratio.
                    - `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
                        the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
                        less or equal to `longest_edge`.
                    - `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
                        aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
                        `max_width`.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use if resizing the image.
            data_format (`ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
        """
        size = get_size_dict(size, default_to_square=False)
        if "shortest_edge" in size and "longest_edge" in size:
            new_size = get_resize_output_image_size(
                image, size["shortest_edge"], size["longest_edge"], input_data_format=input_data_format
            )
        elif "height" in size and "width" in size:
            new_size = (size["height"], size["width"])
        elif "max_height" in size and "max_width" in size:
            new_size = get_image_size_for_max_height_width(
                image, size["max_height"], size["max_width"], input_data_format=input_data_format
            )
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        image = resize(
            image, size=new_size, resample=resample, data_format=data_format, input_data_format=input_data_format
        )
        return image

    def resize_annotation(
        self,
        annotation,
        orig_size,
        size,
        resample: PILImageResampling = PILImageResampling.NEAREST,
    ) -> Dict:
        """
        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched
        to this number.
        """
        return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)

    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Rescale the image by the given factor. image = image * rescale_factor.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            rescale_factor (`float`):
                The value to use for rescaling.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. If unset, is inferred from the input image. Can be
                one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        """
        Normalize the boxes in the annotation from `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]` to
        `[center_x, center_y, width, height]` format and from absolute to relative pixel values.
        """
        return normalize_annotation(annotation, image_size=image_size)

    def _update_annotation_for_padded_image(
        self,
        annotation: Dict,
        input_image_size: Tuple[int, int],
        output_image_size: Tuple[int, int],
        padding,
        update_bboxes,
    ) -> Dict:
        """
        Update the annotation for a padded image.
        """
        new_annotation = {}
        new_annotation["size"] = output_image_size

        for key, value in annotation.items():
            if key == "masks":
                masks = value
                masks = pad(
                    masks,
                    padding,
                    mode=PaddingMode.CONSTANT,
                    constant_values=0,
                    input_data_format=ChannelDimension.FIRST,
                )
                masks = safe_squeeze(masks, 1)
                new_annotation["masks"] = masks
            elif key == "boxes" and update_bboxes:
                boxes = value
                boxes *= np.asarray(
                    [
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                        input_image_size[1] / output_image_size[1],
                        input_image_size[0] / output_image_size[0],
                    ]
                )
                new_annotation["boxes"] = boxes
            elif key == "size":
                new_annotation["size"] = output_image_size
            else:
                new_annotation[key] = value
        return new_annotation

    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        annotation: Optional[Dict[str, Any]] = None,
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        update_bboxes: bool = True,
    ) -> np.ndarray:
        """
        Pad an image with zeros to the given size.
        """
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))
        padded_image = pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=constant_values,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        if annotation is not None:
            annotation = self._update_annotation_for_padded_image(
                annotation, (input_height, input_width), (output_height, output_width), padding, update_bboxes
            )
        return padded_image, annotation

    def pad(
        self,
        images: List[np.ndarray],
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,
        constant_values: Union[float, Iterable[float]] = 0,
        return_pixel_mask: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        update_bboxes: bool = True,
        pad_size: Optional[Dict[str, int]] = None,
    ) -> BatchFeature:
        """
        Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width
        in the batch and optionally returns their corresponding pixel mask.

        Args:
            images (List[`np.ndarray`]):
                Images to pad.
            annotations (`AnnotationType` or `List[AnnotationType]`, *optional*):
                Annotations to transform according to the padding that is applied to the images.
            constant_values (`float` or `Iterable[float]`, *optional*):
                The value to use for the padding if `mode` is `"constant"`.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a pixel mask.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            update_bboxes (`bool`, *optional*, defaults to `True`):
                Whether to update the bounding boxes in the annotations to match the padded images. If the
                bounding boxes have not been converted to relative coordinates and `(centre_x, centre_y, width, height)`
                format, the bounding boxes will not be updated.
            pad_size (`Dict[str, int]`, *optional*):
                The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
                provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
                height and width in the batch.
        """
        pad_size = pad_size if pad_size is not None else self.pad_size
        if pad_size is not None:
            padded_size = (pad_size["height"], pad_size["width"])
        else:
            padded_size = get_max_height_width(images, input_data_format=input_data_format)

        annotation_list = annotations if annotations is not None else [None] * len(images)
        padded_images = []
        padded_annotations = []
        for image, annotation in zip(images, annotation_list):
            padded_image, padded_annotation = self._pad_image(
                image,
                padded_size,
                annotation,
                constant_values=constant_values,
                data_format=data_format,
                input_data_format=input_data_format,
                update_bboxes=update_bboxes,
            )
            padded_images.append(padded_image)
            padded_annotations.append(padded_annotation)

        data = {"pixel_values": padded_images}

        if return_pixel_mask:
            masks = [
                make_pixel_mask(image=image, output_size=padded_size, input_data_format=input_data_format)
                for image in images
            ]
            data["pixel_mask"] = masks

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        if annotations is not None:
            encoded_inputs["labels"] = [
                BatchFeature(annotation, tensor_type=return_tensors) for annotation in padded_annotations
            ]

        return encoded_inputs

    def preprocess(
        self,
        images: ImageInput,
        annotations: Optional[Union[List[Dict], List[List[Dict]]]] = None,
        return_segmentation_masks: bool = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample=None,  # PILImageResampling
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_annotations: Optional[bool] = None,
        do_pad: Optional[bool] = None,
        format: Optional[Union[str, AnnotationFormat]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        pad_size: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or a batch of images so that it can be used by the model.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging
                from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            annotations (`List[Dict]` or `List[List[Dict]]`, *optional*):
                List of annotations associated with the image or batch of images. If annotation is for object
                detection, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "annotations" (`List[Dict]`): List of annotations for an image. Each annotation should be a
                  dictionary. An image can have no annotations, in which case the list should be empty.
                If annotation is for segmentation, the annotations should be a dictionary with the following keys:
                - "image_id" (`int`): The image id.
                - "segments_info" (`List[Dict]`): List of segments for an image. Each segment should be a dictionary.
                  An image can have no segments, in which case the list should be empty.
                - "file_name" (`str`): The file name of the image.
            return_segmentation_masks (`bool`, *optional*, defaults to self.return_segmentation_masks):
                Whether to return segmentation masks.
            masks_path (`str` or `pathlib.Path`, *optional*):
                Path to the directory containing the segmentation masks.
            do_resize (`bool`, *optional*, defaults to self.do_resize):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to self.size):
                Size of the image's `(height, width)` dimensions after resizing. Available options are:
                    - `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
                        Do NOT keep the aspect ratio.
                    - `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
                        the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
                        less or equal to `longest_edge`.
                    - `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
                        aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
                        `max_width`.
            resample (`PILImageResampling`, *optional*, defaults to self.resample):
                Resampling filter to use when resizing the image.
            do_rescale (`bool`, *optional*, defaults to self.do_rescale):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to self.rescale_factor):
                Rescale factor to use when rescaling the image.
            do_normalize (`bool`, *optional*, defaults to self.do_normalize):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to self.image_mean):
                Mean to use when normalizing the image.
            image_std (`float` or `List[float]`, *optional*, defaults to self.image_std):
                Standard deviation to use when normalizing the image.
            do_convert_annotations (`bool`, *optional*, defaults to self.do_convert_annotations):
                Whether to convert the annotations to the format expected by the model. Converts the bounding
                boxes from the format `(top_left_x, top_left_y, width, height)` to `(center_x, center_y, width, height)`
                and in relative coordinates.
            do_pad (`bool`, *optional*, defaults to self.do_pad):
                Whether to pad the image. If `True`, padding will be applied to the bottom and right of
                the image with zeros. If `pad_size` is provided, the image will be padded to the specified
                dimensions. Otherwise, the image will be padded to the maximum height and width of the batch.
            format (`str` or `AnnotationFormat`, *optional*, defaults to self.format):
                Format of the annotations.
            return_tensors (`str` or `TensorType`, *optional*, defaults to self.return_tensors):
                Type of tensors to return. If `None`, will return the list of images.
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
            pad_size (`Dict[str, int]`, *optional*):
                The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
                provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
                height and width in the batch.
        """
        if "pad_and_return_pixel_mask" in kwargs:
            logger.warning_once(
                "The `pad_and_return_pixel_mask` argument is deprecated and will be removed in a future version, "
                "use `do_pad` instead.",
            )
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        size = get_size_dict(size=size, default_to_square=False)
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        do_convert_annotations = (
            self.do_convert_annotations if do_convert_annotations is None else do_convert_annotations
        )
        do_pad = self.do_pad if do_pad is None else do_pad
        pad_size = self.pad_size if pad_size is None else pad_size
        format = self.format if format is None else format

        # Here, the pad() method pads to the maximum of (width, height). It does not need to be validated.

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

        if not is_batched(images):
            images = [images]
            annotations = [annotations] if annotations is not None else None

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        if annotations is not None and len(images) != len(annotations):
            raise ValueError(
                f"The number of images ({len(images)}) and annotations ({len(annotations)}) do not match."
            )

        format = AnnotationFormat(format)
        if annotations is not None:
            validate_annotations(format, SUPPORTED_ANNOTATION_FORMATS, annotations)

        if (
            masks_path is not None
            and format == AnnotationFormat.COCO_PANOPTIC
            and not isinstance(masks_path, (pathlib.Path, str))
        ):
            raise ValueError(
                "The path to the directory containing the mask PNG files should be provided as a"
                f" `pathlib.Path` or string object, but is {type(masks_path)} instead."
            )

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        # prepare (COCO annotations as a list of Dict -> DETR target as a single Dict per image)
        if annotations is not None:
            prepared_images = []
            prepared_annotations = []
            for image, target in zip(images, annotations):
                target = self.prepare_annotation(
                    image,
                    target,
                    format,
                    return_segmentation_masks=return_segmentation_masks,
                    masks_path=masks_path,
                    input_data_format=input_data_format,
                )
                prepared_images.append(image)
                prepared_annotations.append(target)
            images = prepared_images
            annotations = prepared_annotations
            del prepared_images, prepared_annotations

        # transformations
        if do_resize:
            if annotations is not None:
                resized_images, resized_annotations = [], []
                for image, target in zip(images, annotations):
                    orig_size = get_image_size(image, input_data_format)
                    resized_image = self.resize(
                        image, size=size, resample=resample, input_data_format=input_data_format
                    )
                    resized_annotation = self.resize_annotation(
                        target, orig_size, get_image_size(resized_image, input_data_format)
                    )
                    resized_images.append(resized_image)
                    resized_annotations.append(resized_annotation)
                images = resized_images
                annotations = resized_annotations
                del resized_images, resized_annotations
            else:
                images = [
                    self.resize(image, size=size, resample=resample, input_data_format=input_data_format)
                    for image in images
                ]

        if do_rescale:
            images = [self.rescale(image, rescale_factor, input_data_format=input_data_format) for image in images]

        if do_normalize:
            images = [
                self.normalize(image, image_mean, image_std, input_data_format=input_data_format) for image in images
            ]

        if do_convert_annotations and annotations is not None:
            annotations = [
                self.normalize_annotation(annotation, get_image_size(image, input_data_format))
                for annotation, image in zip(annotations, images)
            ]

        if do_pad:
            # Pads images and returns their mask: {'pixel_values': ..., 'pixel_mask': ...}
            encoded_inputs = self.pad(
                images,
                annotations=annotations,
                return_pixel_mask=True,
                data_format=data_format,
                input_data_format=input_data_format,
                return_tensors=return_tensors,
                update_bboxes=do_convert_annotations,
                pad_size=pad_size,
            )
        else:
            images = [
                to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
                for image in images
            ]
            encoded_inputs = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)
            if annotations is not None:
                encoded_inputs["labels"] = [
                    BatchFeature(annotation, tensor_type=return_tensors) for annotation in annotations
                ]

        return encoded_inputs

    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.5,
        target_sizes: Union[TensorType, List[Tuple]] = None,
        nms_threshold: float = 0.7,
    ):
        """
        Converts the output of [`DetaForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.5):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.
            nms_threshold (`float`, *optional*, defaults to 0.7):
                NMS threshold.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes
        batch_size, num_queries, num_labels = out_logits.shape

        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        prob = out_logits.sigmoid()

        all_scores = prob.view(batch_size, num_queries * num_labels).to(out_logits.device)
        all_indexes = torch.arange(num_queries * num_labels)[None].repeat(batch_size, 1).to(out_logits.device)
        all_boxes = torch.div(all_indexes, out_logits.shape[2], rounding_mode="floor")
        all_labels = all_indexes % out_logits.shape[2]

        boxes = center_to_corners_format(out_bbox)
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for b in range(batch_size):
            box = boxes[b]
            score = all_scores[b]
            lbls = all_labels[b]

            pre_topk = score.topk(min(10000, num_queries * num_labels)).indices
            box = box[pre_topk]
            score = score[pre_topk]
            lbls = lbls[pre_topk]

            # apply NMS
            keep_inds = batched_nms(box, score, lbls, nms_threshold)[:100]
            score = score[keep_inds]
            lbls = lbls[keep_inds]
            box = box[keep_inds]

            results.append(
                {
                    "scores": score[score > threshold],
                    "labels": lbls[score > threshold],
                    "boxes": box[score > threshold],
                }
            )

        return results

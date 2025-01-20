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
"""Fast Image processor class for DETR."""

import functools
import io
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    SizeDict,
    get_image_size_for_max_height_width,
    get_max_height_width,
    safe_squeeze,
)
from ...image_transforms import (
    center_to_corners_format,
    corners_to_center_format,
    id_to_rgb,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    AnnotationFormat,
    AnnotationType,
    ChannelDimension,
    ImageInput,
    ImageType,
    PILImageResampling,
    get_image_size,
    get_image_type,
    infer_channel_dimension_format,
    make_list_of_images,
    validate_annotations,
    validate_kwargs,
)
from ...utils import (
    TensorType,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)
from .image_processing_detr import (
    compute_segments,
    convert_segmentation_to_rle,
    get_size_with_aspect_ratio,
    remove_low_and_no_objects,
)


if is_torch_available():
    import torch
    from torch import nn

if is_vision_available():
    import PIL

    from ...image_utils import pil_torch_interpolation_mapping


if is_torchvision_v2_available():
    from torchvision.io import read_image
    from torchvision.transforms.v2 import functional as F
elif is_torchvision_available():
    from torchvision.io import read_image
    from torchvision.transforms import functional as F


logger = logging.get_logger(__name__)

SUPPORTED_ANNOTATION_FORMATS = (AnnotationFormat.COCO_DETECTION, AnnotationFormat.COCO_PANOPTIC)


# inspired by https://github.com/facebookresearch/detr/blob/master/datasets/coco.py#L33
def convert_coco_poly_to_mask(segmentations, height: int, width: int, device: torch.device) -> torch.Tensor:
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
        mask = torch.as_tensor(mask, dtype=torch.uint8, device=device)
        mask = torch.any(mask, axis=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, axis=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8, device=device)

    return masks


# inspired by https://github.com/facebookresearch/detr/blob/master/datasets/coco.py#L50
def prepare_coco_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """
    Convert the target in COCO format into the format expected by DETR.
    """
    image_height, image_width = image.size()[-2:]

    image_id = target["image_id"]
    image_id = torch.as_tensor([image_id], dtype=torch.int64, device=image.device)

    # Get all COCO annotations for the given image.
    annotations = target["annotations"]
    classes = []
    area = []
    boxes = []
    keypoints = []
    for obj in annotations:
        if "iscrowd" not in obj or obj["iscrowd"] == 0:
            classes.append(obj["category_id"])
            area.append(obj["area"])
            boxes.append(obj["bbox"])
            if "keypoints" in obj:
                keypoints.append(obj["keypoints"])

    classes = torch.as_tensor(classes, dtype=torch.int64, device=image.device)
    area = torch.as_tensor(area, dtype=torch.float32, device=image.device)
    iscrowd = torch.zeros_like(classes, dtype=torch.int64, device=image.device)
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=image.device).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=image_width)
    boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=image_height)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    new_target = {
        "image_id": image_id,
        "class_labels": classes[keep],
        "boxes": boxes[keep],
        "area": area[keep],
        "iscrowd": iscrowd[keep],
        "orig_size": torch.as_tensor([int(image_height), int(image_width)], dtype=torch.int64, device=image.device),
    }

    if keypoints:
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=image.device)
        # Apply the keep mask here to filter the relevant annotations
        keypoints = keypoints[keep]
        num_keypoints = keypoints.shape[0]
        keypoints = keypoints.reshape((-1, 3)) if num_keypoints else keypoints
        new_target["keypoints"] = keypoints

    if return_segmentation_masks:
        segmentation_masks = [obj["segmentation"] for obj in annotations]
        masks = convert_coco_poly_to_mask(segmentation_masks, image_height, image_width, device=image.device)
        new_target["masks"] = masks[keep]

    return new_target


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided panoptic segmentation masks.

    Args:
        masks: masks in format `[number_masks, height, width]` where N is the number of masks

    Returns:
        boxes: bounding boxes in format `[number_masks, 4]` in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float32, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float32, device=masks.device)
    # see https://github.com/pytorch/pytorch/issues/50276
    y, x = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * torch.unsqueeze(x, 0)
    x_max = x_mask.view(x_mask.shape[0], -1).max(-1)[0]
    x_min = (
        torch.where(masks, x.unsqueeze(0), torch.tensor(1e8, device=masks.device)).view(masks.shape[0], -1).min(-1)[0]
    )

    y_mask = masks * torch.unsqueeze(y, 0)
    y_max = y_mask.view(y_mask.shape[0], -1).max(-1)[0]
    y_min = (
        torch.where(masks, y.unsqueeze(0), torch.tensor(1e8, device=masks.device)).view(masks.shape[0], -1).min(-1)[0]
    )

    return torch.stack([x_min, y_min, x_max, y_max], 1)


# 2 functions below adapted from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
# Copyright (c) 2018, Alexander Kirillov
# All rights reserved.
def rgb_to_id(color):
    """
    Converts RGB color to unique ID.
    """
    if isinstance(color, torch.Tensor) and len(color.shape) == 3:
        if color.dtype == torch.uint8:
            color = color.to(torch.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def prepare_coco_panoptic_annotation(
    image: torch.Tensor,
    target: Dict,
    masks_path: Union[str, pathlib.Path],
    return_masks: bool = True,
    input_data_format: Union[ChannelDimension, str] = None,
) -> Dict:
    """
    Prepare a coco panoptic annotation for DETR.
    """
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)
    annotation_path = pathlib.Path(masks_path) / target["file_name"]

    new_target = {}
    new_target["image_id"] = torch.as_tensor(
        [target["image_id"] if "image_id" in target else target["id"]], dtype=torch.int64, device=image.device
    )
    new_target["size"] = torch.as_tensor([image_height, image_width], dtype=torch.int64, device=image.device)
    new_target["orig_size"] = torch.as_tensor([image_height, image_width], dtype=torch.int64, device=image.device)

    if "segments_info" in target:
        masks = read_image(annotation_path).permute(1, 2, 0).to(torch.int32).to(image.device)
        masks = rgb_to_id(masks)

        ids = torch.as_tensor([segment_info["id"] for segment_info in target["segments_info"]], device=image.device)
        masks = masks == ids[:, None, None]
        masks = masks.to(torch.bool)
        if return_masks:
            new_target["masks"] = masks
        new_target["boxes"] = masks_to_boxes(masks)
        new_target["class_labels"] = torch.as_tensor(
            [segment_info["category_id"] for segment_info in target["segments_info"]],
            dtype=torch.int64,
            device=image.device,
        )
        new_target["iscrowd"] = torch.as_tensor(
            [segment_info["iscrowd"] for segment_info in target["segments_info"]],
            dtype=torch.int64,
            device=image.device,
        )
        new_target["area"] = torch.as_tensor(
            [segment_info["area"] for segment_info in target["segments_info"]],
            dtype=torch.float32,
            device=image.device,
        )

    return new_target


class DetrImageProcessorFast(BaseImageProcessorFast):
    r"""
    Constructs a fast Detr image processor.

    Args:
        format (`str`, *optional*, defaults to `AnnotationFormat.COCO_DETECTION`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's `(height, width)` dimensions to the specified `size`. Can be
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
        do_normalize (`bool`, *optional*, defaults to `True`):
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
        resample: Union[PILImageResampling, "F.InterpolationMode"] = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        do_convert_annotations: Optional[bool] = None,
        do_pad: bool = True,
        pad_size: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> None:
        if "pad_and_return_pixel_mask" in kwargs:
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        if "max_size" in kwargs:
            logger.error(
                "The `max_size` parameter is deprecated. " "Please specify in `size['longest_edge'] instead`.",
            )

        size = size if size is not None else {"shortest_edge": 800, "longest_edge": 1333}
        # Backwards compatibility
        size = {"shortest_edge": size, "longest_edge": 1333} if isinstance(size, int) else size
        size = get_size_dict(size, default_to_square=False)

        # Backwards compatibility
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
        self._valid_processor_keys = [
            "images",
            "annotations",
            "return_segmentation_masks",
            "masks_path",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "do_convert_annotations",
            "image_mean",
            "image_std",
            "do_pad",
            "pad_size",
            "format",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `DetrImageProcessorFast.from_pretrained(checkpoint, size=600,
        max_size=800)`
        """
        image_processor_dict = image_processor_dict.copy()
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
        if "pad_and_return_pixel_mask" in kwargs:
            image_processor_dict["pad_and_return_pixel_mask"] = kwargs.pop("pad_and_return_pixel_mask")
        return super().from_dict(image_processor_dict, **kwargs)

    def prepare_annotation(
        self,
        image: torch.Tensor,
        target: Dict,
        format: Optional[AnnotationFormat] = None,
        return_segmentation_masks: bool = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Dict:
        """
        Prepare an annotation for feeding into DETR model.
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
        image: torch.Tensor,
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Size of the image's `(height, width)` dimensions after resizing. Available options are:
                    - `{"height": int, "width": int}`: The image will be resized to the exact size `(height, width)`.
                        Do NOT keep the aspect ratio.
                    - `{"shortest_edge": int, "longest_edge": int}`: The image will be resized to a maximum size respecting
                        the aspect ratio and keeping the shortest edge less or equal to `shortest_edge` and the longest edge
                        less or equal to `longest_edge`.
                    - `{"max_height": int, "max_width": int}`: The image will be resized to the maximum size respecting the
                        aspect ratio and keeping the height less or equal to `max_height` and the width less or equal to
                        `max_width`.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                Resampling filter to use if resizing the image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if size.shortest_edge and size.longest_edge:
            # Resize the image so that the shortest edge or the longest edge is of the given size
            # while maintaining the aspect ratio of the original image.
            new_size = get_size_with_aspect_ratio(
                image.size()[-2:],
                size["shortest_edge"],
                size["longest_edge"],
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(image.size()[-2:], size["max_height"], size["max_width"])
        elif size.height and size.width:
            new_size = (size["height"], size["width"])
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )

        image = F.resize(
            image,
            size=new_size,
            interpolation=interpolation,
            **kwargs,
        )
        return image

    def resize_annotation(
        self,
        annotation: Dict[str, Any],
        orig_size: Tuple[int, int],
        target_size: Tuple[int, int],
        threshold: float = 0.5,
        interpolation: "F.InterpolationMode" = None,
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
            resample (`InterpolationMode`, defaults to `InterpolationMode.NEAREST`):
                The resampling filter to use when resizing the masks.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.NEAREST
        ratio_height, ratio_width = [target / orig for target, orig in zip(target_size, orig_size)]

        new_annotation = {}
        new_annotation["size"] = target_size

        for key, value in annotation.items():
            if key == "boxes":
                boxes = value
                scaled_boxes = boxes * torch.as_tensor(
                    [ratio_width, ratio_height, ratio_width, ratio_height], dtype=torch.float32, device=boxes.device
                )
                new_annotation["boxes"] = scaled_boxes
            elif key == "area":
                area = value
                scaled_area = area * (ratio_width * ratio_height)
                new_annotation["area"] = scaled_area
            elif key == "masks":
                masks = value[:, None]
                masks = [F.resize(mask, target_size, interpolation=interpolation) for mask in masks]
                masks = torch.stack(masks).to(torch.float32)
                masks = masks[:, 0] > threshold
                new_annotation["masks"] = masks
            elif key == "size":
                new_annotation["size"] = target_size
            else:
                new_annotation[key] = value

        return new_annotation

    def normalize_annotation(self, annotation: Dict, image_size: Tuple[int, int]) -> Dict:
        image_height, image_width = image_size
        norm_annotation = {}
        for key, value in annotation.items():
            if key == "boxes":
                boxes = value
                boxes = corners_to_center_format(boxes)
                boxes /= torch.as_tensor(
                    [image_width, image_height, image_width, image_height], dtype=torch.float32, device=boxes.device
                )
                norm_annotation[key] = boxes
            else:
                norm_annotation[key] = value
        return norm_annotation

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
        ratio_height, ratio_width = (input / output for output, input in zip(output_image_size, input_image_size))

        for key, value in annotation.items():
            if key == "masks":
                masks = value
                masks = F.pad(
                    masks,
                    padding,
                    fill=0,
                )
                masks = safe_squeeze(masks, 1)
                new_annotation["masks"] = masks
            elif key == "boxes" and update_bboxes:
                boxes = value
                boxes *= torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height], device=boxes.device)
                new_annotation["boxes"] = boxes
            elif key == "size":
                new_annotation["size"] = output_image_size
            else:
                new_annotation[key] = value
        return new_annotation

    def pad(
        self,
        image: torch.Tensor,
        padded_size: Tuple[int, int],
        annotation: Optional[Dict[str, Any]] = None,
        update_bboxes: bool = True,
        fill: int = 0,
    ):
        original_size = image.size()[-2:]
        padding_bottom = padded_size[0] - original_size[0]
        padding_right = padded_size[1] - original_size[1]
        if padding_bottom < 0 or padding_right < 0:
            raise ValueError(
                f"Padding dimensions are negative. Please make sure that the padded size is larger than the "
                f"original size. Got padded size: {padded_size}, original size: {original_size}."
            )
        if original_size != padded_size:
            padding = [0, 0, padding_right, padding_bottom]
            image = F.pad(image, padding, fill=fill)
            if annotation is not None:
                annotation = self._update_annotation_for_padded_image(
                    annotation, original_size, padded_size, padding, update_bboxes
                )

        # Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.
        pixel_mask = torch.zeros(padded_size, dtype=torch.int64, device=image.device)
        pixel_mask[: original_size[0], : original_size[1]] = 1

        return image, pixel_mask, annotation

    @functools.lru_cache(maxsize=1)
    def _validate_input_arguments(
        self,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
        do_resize: bool,
        size: Dict[str, int],
        resample: "PILImageResampling",
        data_format: Union[str, ChannelDimension],
        return_tensors: Union[TensorType, str],
    ):
        if return_tensors != "pt":
            raise ValueError("Only returning PyTorch tensors is currently supported.")

        if data_format != ChannelDimension.FIRST:
            raise ValueError("Only channel first data format is currently supported.")

        if do_resize and None in (size, resample):
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and None in (image_mean, image_std):
            raise ValueError("Image mean and standard deviation must be specified if do_normalize is True.")

    def preprocess(
        self,
        images: ImageInput,
        annotations: Optional[Union[AnnotationType, List[AnnotationType]]] = None,
        return_segmentation_masks: bool = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[Union[PILImageResampling, "F.InterpolationMode"]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        do_convert_annotations: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
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
            annotations (`AnnotationType` or `List[AnnotationType]`, *optional*):
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
            resample (`PILImageResampling` or `InterpolationMode`, *optional*, defaults to self.resample):
                Resampling filter to use when resizing the image.
            do_rescale (`bool`, *optional*, defaults to self.do_rescale):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to self.rescale_factor):
                Rescale factor to use when rescaling the image.
            do_normalize (`bool`, *optional*, defaults to self.do_normalize):
                Whether to normalize the image.
            do_convert_annotations (`bool`, *optional*, defaults to self.do_convert_annotations):
                Whether to convert the annotations to the format expected by the model. Converts the bounding
                boxes from the format `(top_left_x, top_left_y, width, height)` to `(center_x, center_y, width, height)`
                and in relative coordinates.
            image_mean (`float` or `List[float]`, *optional*, defaults to self.image_mean):
                Mean to use when normalizing the image.
            image_std (`float` or `List[float]`, *optional*, defaults to self.image_std):
                Standard deviation to use when normalizing the image.
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
                "use `do_pad` instead."
            )
            do_pad = kwargs.pop("pad_and_return_pixel_mask")

        if "max_size" in kwargs:
            logger.error("The `max_size` argument is deprecated, use" " `size['longest_edge']` instead.")

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
        device = kwargs.pop("device", None)

        # Make hashable for cache
        size = SizeDict(**size)
        image_mean = tuple(image_mean) if isinstance(image_mean, list) else image_mean
        image_std = tuple(image_std) if isinstance(image_std, list) else image_std

        images = make_list_of_images(images)
        image_type = get_image_type(images[0])

        if image_type not in [ImageType.PIL, ImageType.TORCH, ImageType.NUMPY]:
            raise ValueError(f"Unsupported input image type {image_type}")
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

        self._validate_input_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
            return_tensors=return_tensors,
            data_format=data_format,
        )

        if annotations is not None and isinstance(annotations, dict):
            annotations = [annotations]

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

        data = {}
        if image_type == ImageType.PIL:
            images = [F.pil_to_tensor(image) for image in images]
        elif image_type == ImageType.NUMPY:
            # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
            images = [torch.from_numpy(image).contiguous() for image in images]

        if device is not None:
            images = [image.to(device) for image in images]

        # We assume that all images have the same channel dimension format.
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])
        if input_data_format == ChannelDimension.LAST:
            images = [image.permute(2, 0, 1).contiguous() for image in images]
            input_data_format = ChannelDimension.FIRST

        if do_rescale and do_normalize:
            # fused rescale and normalize
            new_mean = torch.tensor(image_mean, device=images[0].device) * (1.0 / rescale_factor)
            new_std = torch.tensor(image_std, device=images[0].device) * (1.0 / rescale_factor)

        processed_images = []
        processed_annotations = []
        pixel_masks = []  # Initialize pixel_masks here
        for image, annotation in zip(images, annotations if annotations is not None else [None] * len(images)):
            # prepare (COCO annotations as a list of Dict -> DETR target as a single Dict per image)
            if annotations is not None:
                annotation = self.prepare_annotation(
                    image,
                    annotation,
                    format,
                    return_segmentation_masks=return_segmentation_masks,
                    masks_path=masks_path,
                    input_data_format=input_data_format,
                )

            if do_resize:
                interpolation = (
                    pil_torch_interpolation_mapping[resample]
                    if isinstance(resample, (PILImageResampling, int))
                    else resample
                )
                resized_image = self.resize(image, size=size, interpolation=interpolation)
                if annotations is not None:
                    annotation = self.resize_annotation(
                        annotation,
                        orig_size=image.size()[-2:],
                        target_size=resized_image.size()[-2:],
                    )
                image = resized_image

            if do_rescale and do_normalize:
                # fused rescale and normalize
                image = F.normalize(image.to(dtype=torch.float32), new_mean, new_std)
            elif do_rescale:
                image = image * rescale_factor
            elif do_normalize:
                image = F.normalize(image, image_mean, image_std)

            if do_convert_annotations and annotations is not None:
                annotation = self.normalize_annotation(annotation, get_image_size(image, input_data_format))

            processed_images.append(image)
            processed_annotations.append(annotation)
        images = processed_images
        annotations = processed_annotations if annotations is not None else None

        if do_pad:
            # depends on all resized image shapes so we need another loop
            if pad_size is not None:
                padded_size = (pad_size["height"], pad_size["width"])
            else:
                padded_size = get_max_height_width(images)

            padded_images = []
            padded_annotations = []
            for image, annotation in zip(images, annotations if annotations is not None else [None] * len(images)):
                # Pads images and returns their mask: {'pixel_values': ..., 'pixel_mask': ...}
                if padded_size == image.size()[-2:]:
                    padded_images.append(image)
                    pixel_masks.append(torch.ones(padded_size, dtype=torch.int64, device=image.device))
                    padded_annotations.append(annotation)
                    continue
                image, pixel_mask, annotation = self.pad(
                    image, padded_size, annotation=annotation, update_bboxes=do_convert_annotations
                )
                padded_images.append(image)
                padded_annotations.append(annotation)
                pixel_masks.append(pixel_mask)
            images = padded_images
            annotations = padded_annotations if annotations is not None else None
            data.update({"pixel_mask": torch.stack(pixel_masks, dim=0)})

        data.update({"pixel_values": torch.stack(images, dim=0)})
        encoded_inputs = BatchFeature(data, tensor_type=return_tensors)
        if annotations is not None:
            encoded_inputs["labels"] = [
                BatchFeature(annotation, tensor_type=return_tensors) for annotation in annotations
            ]
        return encoded_inputs

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process
    def post_process(self, outputs, target_sizes):
        """
        Converts the raw output of [`DetrForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (height, width) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation). For visualization, this should be the image size
                after data augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logger.warning_once(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
        )

        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        return results

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_segmentation
    def post_process_segmentation(self, outputs, target_sizes, threshold=0.9, mask_threshold=0.5):
        """
        Converts the output of [`DetrForSegmentation`] into image segmentation predictions. Only supports PyTorch.

        Args:
            outputs ([`DetrSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`):
                Torch Tensor (or list) corresponding to the requested final size (h, w) of each prediction.
            threshold (`float`, *optional*, defaults to 0.9):
                Threshold to use to filter out queries.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels, and masks for an image
            in the batch as predicted by the model.
        """
        logger.warning_once(
            "`post_process_segmentation` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_semantic_segmentation`.",
        )
        out_logits, raw_masks = outputs.logits, outputs.pred_masks
        empty_label = out_logits.shape[-1] - 1
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, size in zip(out_logits, raw_masks, target_sizes):
            # we filter empty queries and detection below threshold
            cur_scores, cur_labels = cur_logits.softmax(-1).max(-1)
            keep = cur_labels.ne(empty_label) & (cur_scores > threshold)
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]
            cur_masks = cur_masks[keep]
            cur_masks = nn.functional.interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_masks = (cur_masks.sigmoid() > mask_threshold) * 1

            predictions = {"scores": cur_scores, "labels": cur_labels, "masks": cur_masks}
            preds.append(predictions)
        return preds

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_instance
    def post_process_instance(self, results, outputs, orig_target_sizes, max_target_sizes, threshold=0.5):
        """
        Converts the output of [`DetrForSegmentation`] into actual instance segmentation predictions. Only supports
        PyTorch.

        Args:
            results (`List[Dict]`):
                Results list obtained by [`~DetrImageProcessor.post_process`], to which "masks" results will be added.
            outputs ([`DetrSegmentationOutput`]):
                Raw outputs of the model.
            orig_target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation).
            max_target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the maximum size (h, w) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation).
            threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels, boxes and masks for an
            image in the batch as predicted by the model.
        """
        logger.warning_once(
            "`post_process_instance` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_instance_segmentation`.",
        )

        if len(orig_target_sizes) != len(max_target_sizes):
            raise ValueError("Make sure to pass in as many orig_target_sizes as max_target_sizes")
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs.pred_masks.squeeze(2)
        outputs_masks = nn.functional.interpolate(
            outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False
        )
        outputs_masks = (outputs_masks.sigmoid() > threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = nn.functional.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        return results

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_panoptic
    def post_process_panoptic(self, outputs, processed_sizes, target_sizes=None, is_thing_map=None, threshold=0.85):
        """
        Converts the output of [`DetrForSegmentation`] into actual panoptic predictions. Only supports PyTorch.

        Args:
            outputs ([`DetrSegmentationOutput`]):
                Raw outputs of the model.
            processed_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`):
                Torch Tensor (or list) containing the size (h, w) of each image of the batch, i.e. the size after data
                augmentation but before batching.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)` or `List[Tuple]` of length `batch_size`, *optional*):
                Torch Tensor (or list) corresponding to the requested final size `(height, width)` of each prediction.
                If left to None, it will default to the `processed_sizes`.
            is_thing_map (`torch.Tensor` of shape `(batch_size, 2)`, *optional*):
                Dictionary mapping class indices to either True or False, depending on whether or not they are a thing.
                If not set, defaults to the `is_thing_map` of COCO panoptic.
            threshold (`float`, *optional*, defaults to 0.85):
                Threshold to use to filter out queries.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing a PNG string and segments_info values for
            an image in the batch as predicted by the model.
        """
        logger.warning_once(
            "`post_process_panoptic is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_panoptic_segmentation`.",
        )
        if target_sizes is None:
            target_sizes = processed_sizes
        if len(processed_sizes) != len(target_sizes):
            raise ValueError("Make sure to pass in as many processed_sizes as target_sizes")

        if is_thing_map is None:
            # default to is_thing_map of COCO panoptic
            is_thing_map = {i: i <= 90 for i in range(201)}

        out_logits, raw_masks, raw_boxes = outputs.logits, outputs.pred_masks, outputs.pred_boxes
        if not len(out_logits) == len(raw_masks) == len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits and masks"
            )
        empty_label = out_logits.shape[-1] - 1
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            cur_scores, cur_labels = cur_logits.softmax(-1).max(-1)
            keep = cur_labels.ne(empty_label) & (cur_scores > threshold)
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]
            cur_masks = cur_masks[keep]
            cur_masks = nn.functional.interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_boxes = center_to_corners_format(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            if len(cur_boxes) != len(cur_labels):
                raise ValueError("Not as many boxes as there are classes")

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_labels):
                if not is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = PIL.Image.fromarray(id_to_rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=PILImageResampling.NEAREST)

                np_seg_img = torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes()))
                np_seg_img = np_seg_img.view(final_h, final_w, 3)
                np_seg_img = np_seg_img.numpy()

                m_id = torch.from_numpy(rgb_to_id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_labels.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_labels)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_labels = cur_labels[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_labels = torch.ones(1, dtype=torch.long, device=cur_labels.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_labels[i].item()
                segments_info.append({"id": i, "isthing": is_thing_map[cat], "category_id": cat, "area": a})
            del cur_labels

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_object_detection
    def post_process_object_detection(
        self, outputs, threshold: float = 0.5, target_sizes: Union[TensorType, List[Tuple]] = None
    ):
        """
        Converts the raw output of [`DetrForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. Only supports PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(out_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_semantic_segmentation
    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple[int, int]] = None):
        """
        Converts the output of [`DetrForSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`DetrForSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                A list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the
                batch. If unset, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.pred_masks  # [batch_size, num_queries, height, width]

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_instance_segmentation
    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        return_coco_annotation: Optional[bool] = False,
    ) -> List[Dict]:
        """
        Converts the output of [`DetrForSegmentation`] into instance segmentation predictions. Only supports PyTorch.

        Args:
            outputs ([`DetrForSegmentation`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If unset, predictions will not be resized.
            return_coco_annotation (`bool`, *optional*):
                Defaults to `False`. If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE)
                format.
        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- A tensor of shape `(height, width)` where each pixel represents a `segment_id` or
              `List[List]` run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
              `True`. Set to `None` if no mask if found above `threshold`.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- An integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """
        class_queries_logits = outputs.logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.pred_masks  # [batch_size, num_queries, height, width]

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        mask_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Predicted label and score of each query (batch_size, num_queries)
        pred_scores, pred_labels = nn.functional.softmax(class_queries_logits, dim=-1).max(-1)

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(batch_size):
            mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(
                mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels
            )

            # No mask found
            if mask_probs_item.shape[0] <= 0:
                height, width = target_sizes[i] if target_sizes is not None else mask_probs_item.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            # Get segmentation map and segment information of batch item
            target_size = target_sizes[i] if target_sizes is not None else None
            segmentation, segments = compute_segments(
                mask_probs=mask_probs_item,
                pred_scores=pred_scores_item,
                pred_labels=pred_labels_item,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                label_ids_to_fuse=[],
                target_size=target_size,
            )

            # Return segmentation map in run-length encoding (RLE) format
            if return_coco_annotation:
                segmentation = convert_segmentation_to_rle(segmentation)

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.post_process_panoptic_segmentation
    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: Optional[Set[int]] = None,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict]:
        """
        Converts the output of [`DetrForSegmentation`] into image panoptic segmentation predictions. Only supports
        PyTorch.

        Args:
            outputs ([`DetrForSegmentation`]):
                The outputs from [`DetrForSegmentation`].
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            label_ids_to_fuse (`Set[int]`, *optional*):
                The labels in this state will have all their instances be fused together. For instance we could say
                there can only be one sky in an image, but several persons, so the label ID for sky would be in that
                set, but not the one for person.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction in batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id` or
              `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized to
              the corresponding `target_sizes` entry.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                  Multiple instances of the same class / label were fused and assigned a single `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """

        if label_ids_to_fuse is None:
            logger.warning_once("`label_ids_to_fuse` unset. No instance will be fused.")
            label_ids_to_fuse = set()

        class_queries_logits = outputs.logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.pred_masks  # [batch_size, num_queries, height, width]

        batch_size = class_queries_logits.shape[0]
        num_labels = class_queries_logits.shape[-1] - 1

        mask_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Predicted label and score of each query (batch_size, num_queries)
        pred_scores, pred_labels = nn.functional.softmax(class_queries_logits, dim=-1).max(-1)

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(batch_size):
            mask_probs_item, pred_scores_item, pred_labels_item = remove_low_and_no_objects(
                mask_probs[i], pred_scores[i], pred_labels[i], threshold, num_labels
            )

            # No mask found
            if mask_probs_item.shape[0] <= 0:
                height, width = target_sizes[i] if target_sizes is not None else mask_probs_item.shape[1:]
                segmentation = torch.zeros((height, width)) - 1
                results.append({"segmentation": segmentation, "segments_info": []})
                continue

            # Get segmentation map and segment information of batch item
            target_size = target_sizes[i] if target_sizes is not None else None
            segmentation, segments = compute_segments(
                mask_probs=mask_probs_item,
                pred_scores=pred_scores_item,
                pred_labels=pred_labels_item,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                label_ids_to_fuse=label_ids_to_fuse,
                target_size=target_size,
            )

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results


__all__ = ["DetrImageProcessorFast"]

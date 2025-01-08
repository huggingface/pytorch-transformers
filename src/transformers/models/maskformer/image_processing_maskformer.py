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
"""Image processor class for MaskFormer."""

import math
import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from ...image_processing_utils import INIT_SERVICE_KWARGS, BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    PaddingMode,
    get_resize_output_image_size,
    pad,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
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
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    TensorType,
    filter_out_non_signature_kwargs,
    is_torch_available,
    is_torch_tensor,
    logging,
)
from ...utils.deprecation import deprecate_kwarg


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from transformers import MaskFormerForInstanceSegmentationOutput


if is_torch_available():
    import torch
    from torch import nn


# Copied from transformers.models.detr.image_processing_detr.max_across_indices
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


# Copied from transformers.models.detr.image_processing_detr.get_max_height_width
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


# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask
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


# Copied from transformers.models.detr.image_processing_detr.binary_mask_to_rle
def binary_mask_to_rle(mask):
    """
    Converts given binary mask of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        mask (`torch.Tensor` or `numpy.array`):
            A binary mask tensor of shape `(height, width)` where 0 denotes background and 1 denotes the target
            segment_id or class_id.
    Returns:
        `List`: Run-length encoded list of the binary mask. Refer to COCO API for more information about the RLE
        format.
    """
    if is_torch_tensor(mask):
        mask = mask.numpy()

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)


# Copied from transformers.models.detr.image_processing_detr.convert_segmentation_to_rle
def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `List[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    segment_ids = torch.unique(segmentation)

    run_length_encodings = []
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)

    return run_length_encodings


# Copied from transformers.models.detr.image_processing_detr.remove_low_and_no_objects
def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    """
    Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and
    `labels`.

    Args:
        masks (`torch.Tensor`):
            A tensor of shape `(num_queries, height, width)`.
        scores (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        labels (`torch.Tensor`):
            A tensor of shape `(num_queries)`.
        object_mask_threshold (`float`):
            A number between 0 and 1 used to binarize the masks.
    Raises:
        `ValueError`: Raised when the first dimension doesn't match in all input tensors.
    Returns:
        `Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`]`: The `masks`, `scores` and `labels` without the region
        < `object_mask_threshold`.
    """
    if not (masks.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("mask, scores and labels must have the same shape!")

    to_keep = labels.ne(num_labels) & (scores > object_mask_threshold)

    return masks[to_keep], scores[to_keep], labels[to_keep]


# Copied from transformers.models.detr.image_processing_detr.check_segment_validity
def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    # Get the mask associated with the k class
    mask_k = mask_labels == k
    mask_k_area = mask_k.sum()

    # Compute the area of all the stuff in query k
    original_area = (mask_probs[k] >= mask_threshold).sum()
    mask_exists = mask_k_area > 0 and original_area > 0

    # Eliminate disconnected tiny segments
    if mask_exists:
        area_ratio = mask_k_area / original_area
        if not area_ratio.item() > overlap_mask_area_threshold:
            mask_exists = False

    return mask_exists, mask_k


# Copied from transformers.models.detr.image_processing_detr.compute_segments
def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: Optional[Set[int]] = None,
    target_size: Tuple[int, int] = None,
):
    height = mask_probs.shape[1] if target_size is None else target_size[0]
    width = mask_probs.shape[2] if target_size is None else target_size[1]

    segmentation = torch.zeros((height, width), dtype=torch.int32, device=mask_probs.device)
    segments: List[Dict] = []

    if target_size is not None:
        mask_probs = nn.functional.interpolate(
            mask_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

    current_segment_id = 0

    # Weigh each mask by its prediction score
    mask_probs *= pred_scores.view(-1, 1, 1)
    mask_labels = mask_probs.argmax(0)  # [height, width]

    # Keep track of instances of each class
    stuff_memory_list: Dict[str, int] = {}
    for k in range(pred_labels.shape[0]):
        pred_class = pred_labels[k].item()
        should_fuse = pred_class in label_ids_to_fuse

        # Check if mask exists and large enough to be a segment
        mask_exists, mask_k = check_segment_validity(
            mask_labels, mask_probs, k, mask_threshold, overlap_mask_area_threshold
        )

        if mask_exists:
            if pred_class in stuff_memory_list:
                current_segment_id = stuff_memory_list[pred_class]
            else:
                current_segment_id += 1

            # Add current object segment to final segmentation map
            segmentation[mask_k] = current_segment_id
            segment_score = round(pred_scores[k].item(), 6)
            segments.append(
                {
                    "id": current_segment_id,
                    "label_id": pred_class,
                    "was_fused": should_fuse,
                    "score": segment_score,
                }
            )
            if should_fuse:
                stuff_memory_list[pred_class] = current_segment_id

    return segmentation, segments


# TODO: (Amy) Move to image_transforms
def convert_segmentation_map_to_binary_masks(
    segmentation_map: "np.ndarray",
    instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
    ignore_index: Optional[int] = None,
    do_reduce_labels: bool = False,
):
    if do_reduce_labels and ignore_index is None:
        raise ValueError("If `do_reduce_labels` is True, `ignore_index` must be provided.")

    if do_reduce_labels:
        segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    # Get unique ids (class or instance ids based on input)
    all_labels = np.unique(segmentation_map)

    # Drop background label if applicable
    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    # Generate a binary mask for each object instance
    binary_masks = [(segmentation_map == i) for i in all_labels]

    # Stack the binary masks
    if binary_masks:
        binary_masks = np.stack(binary_masks, axis=0)
    else:
        binary_masks = np.zeros((0, *segmentation_map.shape))

    # Convert instance ids to class ids
    if instance_id_to_semantic_id is not None:
        labels = np.zeros(all_labels.shape[0])

        for label in all_labels:
            class_id = instance_id_to_semantic_id[label + 1 if do_reduce_labels else label]
            labels[all_labels == label] = class_id - 1 if do_reduce_labels else class_id
    else:
        labels = all_labels

    return binary_masks.astype(np.float32), labels.astype(np.int64)


def get_maskformer_resize_output_image_size(
    image: np.ndarray,
    size: Union[int, Tuple[int, int], List[int], Tuple[int]],
    max_size: Optional[int] = None,
    size_divisor: int = 0,
    default_to_square: bool = True,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> Tuple[int, int]:
    """
    Computes the output size given the desired size.

    Args:
        image (`np.ndarray`):
            The input image.
        size (`int` or `Tuple[int, int]` or `List[int]` or `Tuple[int]`):
            The size of the output image.
        max_size (`int`, *optional*):
            The maximum size of the output image.
        size_divisor (`int`, *optional*, defaults to 0):
            If `size_divisor` is given, the output image size will be divisible by the number.
        default_to_square (`bool`, *optional*, defaults to `True`):
            Whether to default to square if no size is provided.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `Tuple[int, int]`: The output size.
    """
    output_size = get_resize_output_image_size(
        input_image=image,
        size=size,
        default_to_square=default_to_square,
        max_size=max_size,
        input_data_format=input_data_format,
    )

    if size_divisor > 0:
        height, width = output_size
        height = int(math.ceil(height / size_divisor) * size_divisor)
        width = int(math.ceil(width / size_divisor) * size_divisor)
        output_size = (height, width)

    return output_size


class MaskFormerImageProcessor(BaseImageProcessor):
    r"""
    Constructs a MaskFormer image processor. The image processor can be used to prepare image(s) and optional targets
    for the model.

    This image processor inherits from [`BaseImageProcessor`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 800):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. If size is a
            sequence like `(width, height)`, output size will be matched to this. If size is an int, smaller edge of
            the image will be matched to this number. i.e, if `height > width`, then image will be rescaled to `(size *
            height / width, size)`.
        size_divisor (`int`, *optional*, defaults to 32):
            Some backbones need images divisible by a certain number. If not passed, it defaults to the value used in
            Swin Transformer.
        resample (`int`, *optional*, defaults to `Resampling.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`,
            `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`,
            `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set
            to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input to a certain `scale`.
        rescale_factor (`float`, *optional*, defaults to `1/ 255`):
            Rescale the input by the given factor. Only has an effect if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
        image_std (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
            ImageNet std.
        ignore_index (`int`, *optional*):
            Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
            denoted with 0 (background) will be replaced with `ignore_index`.
        do_reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to decrement all label values of segmentation maps by 1. Usually used for datasets where 0
            is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k).
            The background label will be replaced by `ignore_index`.
        num_labels (`int`, *optional*):
            The number of labels in the segmentation map.

    """

    model_input_names = ["pixel_values", "pixel_mask"]

    @deprecate_kwarg("reduce_labels", new_name="do_reduce_labels", version="4.44.0")
    @deprecate_kwarg("size_divisibility", new_name="size_divisor", version="4.41.0")
    @deprecate_kwarg("max_size", version="4.27.0", warn_if_greater_or_equal_version=True)
    @filter_out_non_signature_kwargs(extra=["max_size", *INIT_SERVICE_KWARGS])
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        size_divisor: int = 32,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Union[float, List[float]] = None,
        image_std: Union[float, List[float]] = None,
        ignore_index: Optional[int] = None,
        do_reduce_labels: bool = False,
        num_labels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # We make max_size a private attribute so we can pass it as a default value in the preprocess method whilst
        # `size` can still be pass in as an int
        self._max_size = kwargs.pop("max_size", 1333)

        size = size if size is not None else {"shortest_edge": 800, "longest_edge": self._max_size}
        size = get_size_dict(size, max_size=self._max_size, default_to_square=False)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.size_divisor = size_divisor
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.ignore_index = ignore_index
        self.do_reduce_labels = do_reduce_labels
        self.num_labels = num_labels

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `MaskFormerImageProcessor.from_pretrained(checkpoint, max_size=800)`
        """
        image_processor_dict = image_processor_dict.copy()
        if "max_size" in kwargs:
            image_processor_dict["max_size"] = kwargs.pop("max_size")
        if "size_divisibility" in kwargs:
            image_processor_dict["size_divisor"] = kwargs.pop("size_divisibility")
        if "reduce_labels" in image_processor_dict:
            image_processor_dict["do_reduce_labels"] = image_processor_dict.pop("reduce_labels")
        return super().from_dict(image_processor_dict, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. This method calls the superclass method and then removes the
        `_max_size` attribute from the dictionary.
        """
        image_processor_dict = super().to_dict()
        image_processor_dict.pop("_max_size", None)
        return image_processor_dict

    @deprecate_kwarg("max_size", version="4.27.0", warn_if_greater_or_equal_version=True)
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        size_divisor: int = 0,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format=None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize the image to the given size. Size can be min_size (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                The size of the output image.
            size_divisor (`int`, *optional*, defaults to 0):
                If `size_divisor` is given, the output image size will be divisible by the number.
            resample (`PILImageResampling` resampling filter, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """

        # Deprecated, backward compatibility
        max_size = kwargs.pop("max_size", None)

        size = get_size_dict(size, max_size=max_size, default_to_square=False)
        if "shortest_edge" in size and "longest_edge" in size:
            size, max_size = size["shortest_edge"], size["longest_edge"]
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
            max_size = None
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got"
                f" {size.keys()}."
            )
        size = get_maskformer_resize_output_image_size(
            image=image,
            size=size,
            max_size=max_size,
            size_divisor=size_divisor,
            default_to_square=False,
            input_data_format=input_data_format,
        )
        image = resize(
            image, size=size, resample=resample, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
        return image

    # Copied from transformers.models.detr.image_processing_detr.DetrImageProcessor.rescale
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

    def convert_segmentation_map_to_binary_masks(
        self,
        segmentation_map: "np.ndarray",
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        ignore_index: Optional[int] = None,
        do_reduce_labels: bool = False,
    ):
        do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels
        ignore_index = ignore_index if ignore_index is not None else self.ignore_index
        return convert_segmentation_map_to_binary_masks(
            segmentation_map=segmentation_map,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            ignore_index=ignore_index,
            do_reduce_labels=do_reduce_labels,
        )

    def __call__(self, images, segmentation_maps=None, **kwargs) -> BatchFeature:
        return self.preprocess(images, segmentation_maps=segmentation_maps, **kwargs)

    def _preprocess(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        size_divisor: int = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        if do_resize:
            image = self.resize(
                image, size=size, size_divisor=size_divisor, resample=resample, input_data_format=input_data_format
            )
        if do_rescale:
            image = self.rescale(image, rescale_factor=rescale_factor, input_data_format=input_data_format)
        if do_normalize:
            image = self.normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        return image

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        size_divisor: int = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
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
            input_data_format = infer_channel_dimension_format(image)
        image = self._preprocess(
            image=image,
            do_resize=do_resize,
            size=size,
            size_divisor=size_divisor,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
        )
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        size_divisor: int = 0,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single mask."""
        segmentation_map = to_numpy_array(segmentation_map)
        # Add channel dimension if missing - needed for certain transformations
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]
            input_data_format = ChannelDimension.FIRST
        else:
            added_channel_dim = False
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
        # TODO: (Amy)
        # Remork segmentation map processing to include reducing labels and resizing which doesn't
        # drop segment IDs > 255.
        segmentation_map = self._preprocess(
            image=segmentation_map,
            do_resize=do_resize,
            resample=PILImageResampling.NEAREST,
            size=size,
            size_divisor=size_divisor,
            do_rescale=False,
            do_normalize=False,
            input_data_format=input_data_format,
        )
        # Remove extra channel dimension if added for processing
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
        return segmentation_map

    @deprecate_kwarg("reduce_labels", new_name="do_reduce_labels", version="4.44.0")
    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        size_divisor: Optional[int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        ignore_index: Optional[int] = None,
        do_reduce_labels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False, max_size=self._max_size)
        size_divisor = size_divisor if size_divisor is not None else self.size_divisor
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        ignore_index = ignore_index if ignore_index is not None else self.ignore_index
        do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels

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

        if segmentation_maps is not None and not valid_images(segmentation_maps):
            raise ValueError(
                "Invalid segmentation map type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        images = make_list_of_images(images)
        if segmentation_maps is not None:
            segmentation_maps = make_list_of_images(segmentation_maps, expected_ndims=2)

        if segmentation_maps is not None and len(images) != len(segmentation_maps):
            raise ValueError("Images and segmentation maps must have the same length.")

        images = [
            self._preprocess_image(
                image,
                do_resize=do_resize,
                size=size,
                size_divisor=size_divisor,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for image in images
        ]

        if segmentation_maps is not None:
            segmentation_maps = [
                self._preprocess_mask(
                    segmentation_map, do_resize, size, size_divisor, input_data_format=input_data_format
                )
                for segmentation_map in segmentation_maps
            ]
        encoded_inputs = self.encode_inputs(
            images,
            segmentation_maps,
            instance_id_to_semantic_id,
            ignore_index,
            do_reduce_labels,
            return_tensors,
            input_data_format=data_format,
        )
        return encoded_inputs

    # Copied from transformers.models.vilt.image_processing_vilt.ViltImageProcessor._pad_image
    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
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
        return padded_image

    # Copied from transformers.models.vilt.image_processing_vilt.ViltImageProcessor.pad
    def pad(
        self,
        images: List[np.ndarray],
        constant_values: Union[float, Iterable[float]] = 0,
        return_pixel_mask: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Pads a batch of images to the bottom and right of the image with zeros to the size of largest height and width
        in the batch and optionally returns their corresponding pixel mask.

        Args:
            image (`np.ndarray`):
                Image to pad.
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
        """
        pad_size = get_max_height_width(images, input_data_format=input_data_format)

        padded_images = [
            self._pad_image(
                image,
                pad_size,
                constant_values=constant_values,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for image in images
        ]
        data = {"pixel_values": padded_images}

        if return_pixel_mask:
            masks = [
                make_pixel_mask(image=image, output_size=pad_size, input_data_format=input_data_format)
                for image in images
            ]
            data["pixel_mask"] = masks

        return BatchFeature(data=data, tensor_type=return_tensors)

    def encode_inputs(
        self,
        pixel_values_list: List[ImageInput],
        segmentation_maps: ImageInput = None,
        instance_id_to_semantic_id: Optional[Union[List[Dict[int, int]], Dict[int, int]]] = None,
        ignore_index: Optional[int] = None,
        do_reduce_labels: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

        MaskFormer addresses semantic segmentation with a mask classification paradigm, thus input segmentation maps
        will be converted to lists of binary masks and their respective labels. Let's see an example, assuming
        `segmentation_maps = [[2,6,7,9]]`, the output will contain `mask_labels =
        [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]` (four binary masks) and `class_labels = [2,6,7,9]`, the labels for
        each mask.

        Args:
            pixel_values_list (`List[ImageInput]`):
                List of images (pixel values) to be padded. Each image should be a tensor of shape `(channels, height,
                width)`.

            segmentation_maps (`ImageInput`, *optional*):
                The corresponding semantic segmentation maps with the pixel-wise annotations.

             (`bool`, *optional*, defaults to `True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            instance_id_to_semantic_id (`List[Dict[int, int]]` or `Dict[int, int]`, *optional*):
                A mapping between object instance ids and class ids. If passed, `segmentation_maps` is treated as an
                instance segmentation map where each pixel represents an instance id. Can be provided as a single
                dictionary with a global/dataset-level mapping or as a list of dictionaries (one per image), to map
                instance ids in each image separately.

            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `=True` or if `pixel_mask` is in
              `self.model_input_names`).
            - **mask_labels** -- Optional list of mask labels of shape `(labels, height, width)` to be fed to a model
              (when `annotations` are provided).
            - **class_labels** -- Optional list of class labels of shape `(labels)` to be fed to a model (when
              `annotations` are provided). They identify the labels of `mask_labels`, e.g. the label of
              `mask_labels[i][j]` if `class_labels[i][j]`.
        """
        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        do_reduce_labels = self.do_reduce_labels if do_reduce_labels is None else do_reduce_labels

        pixel_values_list = [to_numpy_array(pixel_values) for pixel_values in pixel_values_list]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(pixel_values_list[0])

        encoded_inputs = self.pad(
            pixel_values_list, return_tensors=return_tensors, input_data_format=input_data_format
        )

        if segmentation_maps is not None:
            mask_labels = []
            class_labels = []
            pad_size = get_max_height_width(pixel_values_list, input_data_format=input_data_format)
            # Convert to list of binary masks and labels
            for idx, segmentation_map in enumerate(segmentation_maps):
                segmentation_map = to_numpy_array(segmentation_map)
                if isinstance(instance_id_to_semantic_id, list):
                    instance_id = instance_id_to_semantic_id[idx]
                else:
                    instance_id = instance_id_to_semantic_id
                # Use instance2class_id mapping per image
                masks, classes = self.convert_segmentation_map_to_binary_masks(
                    segmentation_map, instance_id, ignore_index=ignore_index, do_reduce_labels=do_reduce_labels
                )
                # We add an axis to make them compatible with the transformations library
                # this will be removed in the future
                if masks.shape[0] > 0:
                    masks = [mask[None, ...] for mask in masks]
                    masks = [
                        self._pad_image(
                            image=mask,
                            output_size=pad_size,
                            constant_values=ignore_index,
                            input_data_format=ChannelDimension.FIRST,
                        )
                        for mask in masks
                    ]
                    masks = np.concatenate(masks, axis=0)
                else:
                    masks = np.zeros((0, *pad_size), dtype=np.float32)
                mask_labels.append(torch.from_numpy(masks))
                class_labels.append(torch.from_numpy(classes))

            # we cannot batch them since they don't share a common class size
            encoded_inputs["mask_labels"] = mask_labels
            encoded_inputs["class_labels"] = class_labels

        return encoded_inputs

    def post_process_segmentation(
        self, outputs: "MaskFormerForInstanceSegmentationOutput", target_size: Tuple[int, int] = None
    ) -> "torch.Tensor":
        """
        Converts the output of [`MaskFormerForInstanceSegmentationOutput`] into image segmentation predictions. Only
        supports PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentationOutput`]):
                The outputs from [`MaskFormerForInstanceSegmentation`].

            target_size (`Tuple[int, int]`, *optional*):
                If set, the `masks_queries_logits` will be resized to `target_size`.

        Returns:
            `torch.Tensor`:
                A tensor of shape (`batch_size, num_class_labels, height, width`).
        """
        warnings.warn(
            "`post_process_segmentation` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_instance_segmentation`",
            FutureWarning,
        )

        # class_queries_logits has shape [BATCH, QUERIES, CLASSES + 1]
        class_queries_logits = outputs.class_queries_logits
        # masks_queries_logits has shape [BATCH, QUERIES, HEIGHT, WIDTH]
        masks_queries_logits = outputs.masks_queries_logits
        if target_size is not None:
            masks_queries_logits = torch.nn.functional.interpolate(
                masks_queries_logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        # remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        # mask probs has shape [BATCH, QUERIES, HEIGHT, WIDTH]
        masks_probs = masks_queries_logits.sigmoid()
        # now we want to sum over the queries,
        # $ out_{c,h,w} =  \sum_q p_{q,c} * m_{q,h,w} $
        # where $ softmax(p) \in R^{q, c} $ is the mask classes
        # and $ sigmoid(m) \in R^{q, h, w}$ is the mask probabilities
        # b(atch)q(uery)c(lasses), b(atch)q(uery)h(eight)w(idth)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        return segmentation

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> "torch.Tensor":
        """
        Converts the output of [`MaskFormerForInstanceSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

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
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        return_coco_annotation: Optional[bool] = False,
        return_binary_maps: Optional[bool] = False,
    ) -> List[Dict]:
        """
        Converts the output of [`MaskFormerForInstanceSegmentationOutput`] into instance segmentation predictions. Only
        supports PyTorch. If instances could overlap, set either return_coco_annotation or return_binary_maps
        to `True` to get the correct segmentation result.

        Args:
            outputs ([`MaskFormerForInstanceSegmentation`]):
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
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
            return_coco_annotation (`bool`, *optional*, defaults to `False`):
                If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE) format.
            return_binary_maps (`bool`, *optional*, defaults to `False`):
                If set to `True`, segmentation maps are returned as a concatenated tensor of binary segmentation maps
                (one per detected instance).
        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- A tensor of shape `(height, width)` where each pixel represents a `segment_id`, or
              `List[List]` run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
              `True`, or a tensor of shape `(num_instances, height, width)` if return_binary_maps is set to `True`.
              Set to `None` if no mask if found above `threshold`.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- An integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """
        if return_coco_annotation and return_binary_maps:
            raise ValueError("return_coco_annotation and return_binary_maps can not be both set to True.")

        # [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits

        device = masks_queries_logits.device
        num_classes = class_queries_logits.shape[-1] - 1
        num_queries = class_queries_logits.shape[-2]

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(class_queries_logits.shape[0]):
            mask_pred = masks_queries_logits[i]
            mask_cls = class_queries_logits[i]

            scores = torch.nn.functional.softmax(mask_cls, dim=-1)[:, :-1]
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(num_queries, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
            mask_pred = mask_pred[topk_indices]
            pred_masks = (mask_pred > 0).float()

            # Calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores_per_image * mask_scores_per_image
            pred_classes = labels_per_image

            segmentation = torch.zeros(masks_queries_logits.shape[2:]) - 1
            if target_sizes is not None:
                segmentation = torch.zeros(target_sizes[i]) - 1
                pred_masks = torch.nn.functional.interpolate(
                    pred_masks.unsqueeze(0), size=target_sizes[i], mode="nearest"
                )[0]

            instance_maps, segments = [], []
            current_segment_id = 0
            for j in range(num_queries):
                score = pred_scores[j].item()

                if not torch.all(pred_masks[j] == 0) and score >= threshold:
                    segmentation[pred_masks[j] == 1] = current_segment_id
                    segments.append(
                        {
                            "id": current_segment_id,
                            "label_id": pred_classes[j].item(),
                            "was_fused": False,
                            "score": round(score, 6),
                        }
                    )
                    current_segment_id += 1
                    instance_maps.append(pred_masks[j])

            # Return segmentation map in run-length encoding (RLE) format
            if return_coco_annotation:
                segmentation = convert_segmentation_to_rle(segmentation)

            # Return a concatenated tensor of binary instance maps
            if return_binary_maps and len(instance_maps) != 0:
                segmentation = torch.stack(instance_maps, dim=0)

            results.append({"segmentation": segmentation, "segments_info": segments})
        return results

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
        Converts the output of [`MaskFormerForInstanceSegmentationOutput`] into image panoptic segmentation
        predictions. Only supports PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentationOutput`]):
                The outputs from [`MaskFormerForInstanceSegmentation`].
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
                final size (height, width) of each prediction in batch. If left to None, predictions will not be
                resized.

        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
              to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
              to the corresponding `target_sizes` entry.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- an integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
                  Multiple instances of the same class / label were fused and assigned a single `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """

        if label_ids_to_fuse is None:
            logger.warning("`label_ids_to_fuse` unset. No instance will be fused.")
            label_ids_to_fuse = set()

        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

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


__all__ = ["MaskFormerImageProcessor"]

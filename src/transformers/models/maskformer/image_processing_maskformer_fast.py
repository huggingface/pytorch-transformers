# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for MaskFormer."""

import math
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from ...image_processing_utils import INIT_SERVICE_KWARGS, BatchFeature, get_size_dict
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    SizeDict,
    get_image_size_for_max_height_width,
    get_max_height_width,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    ImageType,
    PILImageResampling,
    get_image_type,
    infer_channel_dimension_format,
    make_list_of_images,
    valid_images,
    validate_fast_preprocess_arguments,
    validate_kwargs,
)
from ...utils import (
    TensorType,
    filter_out_non_signature_kwargs,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)
from ...utils.deprecation import deprecate_kwarg
from .image_processing_maskformer import (
    compute_segments,
    convert_segmentation_to_rle,
    get_size_with_aspect_ratio,
    remove_low_and_no_objects,
)


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from transformers import MaskFormerForInstanceSegmentationOutput


if is_torch_available():
    import torch
    from torch import nn

if is_vision_available():
    from ...image_utils import pil_torch_interpolation_mapping


if is_torchvision_v2_available():
    from torchvision.transforms.v2 import functional as F
elif is_torchvision_available():
    from torchvision.transforms import functional as F


def convert_segmentation_map_to_binary_masks(
    segmentation_map: torch.Tensor,
    instance_id_to_semantic_id: Optional[Dict[int, int]] = None,
    ignore_index: Optional[int] = None,
    do_reduce_labels: bool = False,
):
    if do_reduce_labels and ignore_index is None:
        raise ValueError("If `do_reduce_labels` is True, `ignore_index` must be provided.")

    if do_reduce_labels:
        segmentation_map = torch.where(segmentation_map == 0, torch.tensor(ignore_index), segmentation_map - 1)

    # Get unique ids (class or instance ids based on input)
    all_labels = torch.unique(segmentation_map)

    # Drop background label if applicable
    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    # Generate a binary mask for each object instance
    binary_masks = [(segmentation_map == i).unsqueeze(0) for i in all_labels]

    # Stack the binary masks
    if binary_masks:
        binary_masks = torch.cat(binary_masks, axis=0)
    else:
        binary_masks = torch.zeros((0, *segmentation_map.shape), device=segmentation_map.device)

    # Convert instance ids to class ids
    if instance_id_to_semantic_id is not None:
        labels = torch.zeros(all_labels.shape[0], dtype=torch.long, device=segmentation_map.device)

        for label in all_labels:
            class_id = instance_id_to_semantic_id[label.item()]
            labels[all_labels == label] = class_id
    else:
        labels = all_labels

    return binary_masks, labels


class MaskFormerImageProcessorFast(BaseImageProcessorFast):
    r"""
    Constructs a fast MaskFormer image processor.

    This image processor inherits from [`BaseImageProcessorFast`] which contains most of the main methods. Users should
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
        pad_size (`Dict[str, int]`, *optional*):
            The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
            provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
            height and width in the batch.

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
        pad_size: Optional[Dict[str, int]] = None,
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
        self.pad_size = pad_size
        self._valid_processor_keys = [
            "images",
            "segmentation_maps",
            "instance_id_to_semantic_id",
            "do_resize",
            "size",
            "size_divisor",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "ignore_index",
            "do_reduce_labels",
            "pad_size",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

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

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        size_divisor: int = 0,
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
            size_divisor (`int`, *optional*, defaults to 0):
                If `size_divisor` is given, the output image size will be divisible by the number.
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

        if size_divisor > 0:
            height, width = new_size
            height = int(math.ceil(height / size_divisor) * size_divisor)
            width = int(math.ceil(width / size_divisor) * size_divisor)
            new_size = (height, width)

        image = F.resize(
            image,
            size=new_size,
            interpolation=interpolation,
            **kwargs,
        )
        return image

    def __call__(self, images, segmentation_maps=None, **kwargs) -> BatchFeature:
        return self.preprocess(images, segmentation_maps=segmentation_maps, **kwargs)

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
        pad_size: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> BatchFeature:
        do_resize = self.do_resize if do_resize is None else do_resize
        size = self.size if size is None else size
        size = get_size_dict(size, default_to_square=False, max_size=self._max_size)
        size_divisor = self.size_divisor if size_divisor is None else size_divisor
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels
        pad_size = self.pad_size if pad_size is None else pad_size
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

        validate_fast_preprocess_arguments(
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

        if segmentation_maps is not None and not valid_images(segmentation_maps):
            raise ValueError(
                "Invalid segmentation map type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if segmentation_maps is not None:
            segmentation_maps = make_list_of_images(segmentation_maps, expected_ndims=2)

        if segmentation_maps is not None and len(images) != len(segmentation_maps):
            raise ValueError("Images and segmentation maps must have the same length.")

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
        processed_segmentation_maps = []
        pixel_masks = []
        for image, segmentation_map in zip(
            images, segmentation_maps if segmentation_maps is not None else [None] * len(images)
        ):
            if do_resize:
                interpolation = (
                    pil_torch_interpolation_mapping[resample]
                    if isinstance(resample, (PILImageResampling, int))
                    else resample
                )
                image = self.resize(image, size=size, size_divisor=size_divisor, interpolation=interpolation)
                if segmentation_map is not None:
                    segmentation_map = self.resize(
                        segmentation_map, size=size, size_divisor=size_divisor, interpolation=interpolation
                    )

            if do_rescale and do_normalize:
                # fused rescale and normalize
                image = F.normalize(image.to(dtype=torch.float32), new_mean, new_std)
            elif do_rescale:
                image = image * rescale_factor
            elif do_normalize:
                image = F.normalize(image, image_mean, image_std)

            processed_images.append(image)
            processed_segmentation_maps.append(segmentation_map)

        images = processed_images
        segmentation_maps = processed_segmentation_maps if segmentation_maps is not None else None

        # depends on all resized image shapes so we need another loop
        if pad_size is not None:
            padded_size = (pad_size["height"], pad_size["width"])
        else:
            padded_size = get_max_height_width(images)

        padded_images = []
        padded_segmentation_maps = []
        for image, segmentation_map in zip(
            images, segmentation_maps if segmentation_maps is not None else [None] * len(images)
        ):
            # Pads images and returns their mask: {'pixel_values': ..., 'pixel_mask': ...}
            if padded_size == image.size()[-2:]:
                padded_images.append(image)
                pixel_masks.append(torch.ones(padded_size, dtype=torch.int64, device=image.device))
                padded_segmentation_maps.append(segmentation_maps)
                continue
            image, pixel_mask, segmentation_map = self.pad(image, padded_size, segmentation_map=segmentation_map)
            padded_images.append(image)
            padded_segmentation_maps.append(segmentation_map)
            pixel_masks.append(pixel_mask)
        images = padded_images
        segmentation_maps = padded_segmentation_maps if segmentation_maps is not None else None

        data.update({"pixel_values": torch.stack(images, dim=0)})
        data.update({"pixel_mask": torch.stack(pixel_masks, dim=0)})

        encoded_inputs = BatchFeature(data, tensor_type=return_tensors)
        if segmentation_maps is not None:
            mask_labels = []
            class_labels = []
            # Convert to list of binary masks and labels
            for idx, segmentation_map in enumerate(segmentation_maps):
                if isinstance(instance_id_to_semantic_id, list):
                    instance_id = instance_id_to_semantic_id[idx]
                else:
                    instance_id = instance_id_to_semantic_id
                # Use instance2class_id mapping per image
                masks, classes = convert_segmentation_map_to_binary_masks(
                    segmentation_map, instance_id, ignore_index=ignore_index, do_reduce_labels=do_reduce_labels
                )
                mask_labels.append(masks)
                class_labels.append(classes)

            # we cannot batch them since they don't share a common class size
            encoded_inputs["mask_labels"] = mask_labels
            encoded_inputs["class_labels"] = class_labels

        return encoded_inputs

    def pad(
        self,
        image: torch.Tensor,
        padded_size: Tuple[int, int],
        segmentation_map: Optional[torch.Tensor] = None,
        fill: int = 0,
    ) -> BatchFeature:
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
            if segmentation_map is not None:
                segmentation_map = F.pad(segmentation_map, padding, fill=fill)

        # Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.
        pixel_mask = torch.zeros(padded_size, dtype=torch.int64, device=image.device)
        pixel_mask[: original_size[0], : original_size[1]] = 1

        return image, pixel_mask, segmentation_map

    # Copied from transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.post_process_segmentation
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

    # Copied from transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.post_process_semantic_segmentation
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

    # Copied from transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.post_process_instance_segmentation
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

    # Copied from transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessor.post_process_panoptic_segmentation
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


__all__ = ["MaskFormerImageProcessorFast"]

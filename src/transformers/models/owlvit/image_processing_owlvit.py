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
"""Image processor class for OwlViT"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    center_crop,
    center_to_corners_format,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, filter_out_non_signature_kwargs, is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


def _upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.
    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


class OwlViTImageProcessor(BaseImageProcessor):
    r"""
    Constructs an OWL-ViT image processor.

    This image processor inherits from [`ImageProcessingMixin`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the shorter edge of the input to a certain `size`.
        size (`Dict[str, int]`, *optional*, defaults to {"height": 768, "width": 768}):
            The size to use for resizing the image. Only has an effect if `do_resize` is set to `True`. If `size` is a
            sequence like (h, w), output size will be matched to this. If `size` is an int, then image will be resized
            to (size, size).
        resample (`int`, *optional*, defaults to `Resampling.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`,
            `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`,
            `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set
            to `True`.
        do_center_crop (`bool`, *optional*, defaults to `False`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped.
        crop_size (`int`, *optional*, defaults to {"height": 768, "width": 768}):
            The size to use for center cropping the image. Only has an effect if `do_center_crop` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input by a certain factor.
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            The factor to use for rescaling the image. Only has an effect if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with `image_mean` and `image_std`. Desired output size when applying
            center-cropping. Only has an effect if `do_center_crop` is set to `True`.
        image_mean (`List[int]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=None,
        resample=PILImageResampling.BICUBIC,
        do_center_crop=False,
        crop_size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs,
    ):
        size = size if size is not None else {"height": 768, "width": 768}
        size = get_size_dict(size, default_to_square=True)

        crop_size = crop_size if crop_size is not None else {"height": 768, "width": 768}
        crop_size = get_size_dict(crop_size, default_to_square=True)

        # Early versions of the OWL-ViT config on the hub had "rescale" as a flag. This clashes with the
        # vision image processor method `rescale` as it would be set as an attribute during the super().__init__
        # call. This is for backwards compatibility.
        if "rescale" in kwargs:
            rescale_val = kwargs.pop("rescale")
            kwargs["do_rescale"] = rescale_val

        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to a certain size.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                The size to resize the image to. Must contain height and width keys.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                The resampling filter to use when resizing the input.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        size = get_size_dict(size, default_to_square=True)
        if "height" not in size or "width" not in size:
            raise ValueError("size dictionary must contain height and width keys")

        return resize(
            image,
            (size["height"], size["width"]),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def center_crop(
        self,
        image: np.ndarray,
        crop_size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Center crop an image to a certain size.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            crop_size (`Dict[str, int]`):
                The size to center crop the image to. Must contain height and width keys.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        crop_size = get_size_dict(crop_size, default_to_square=True)
        if "height" not in crop_size or "width" not in crop_size:
            raise ValueError("crop_size dictionary must contain height and width keys")

        return center_crop(
            image,
            (crop_size["height"], crop_size["width"]),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

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

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> BatchFeature:
        """
        Prepares an image or batch of images for the model.

        Args:
            images (`ImageInput`):
                The image or batch of images to be prepared. Expects a single or batch of images with pixel values
                ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether or not to resize the input. If `True`, will resize the input to the size specified by `size`.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                The size to resize the input to. Only has an effect if `do_resize` is set to `True`.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                The resampling filter to use when resizing the input. Only has an effect if `do_resize` is set to
                `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether or not to center crop the input. If `True`, will center crop the input to the size specified by
                `crop_size`.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                The size to center crop the input to. Only has an effect if `do_center_crop` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether or not to rescale the input. If `True`, will rescale the input by dividing it by
                `rescale_factor`.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                The factor to rescale the input by. Only has an effect if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether or not to normalize the input. If `True`, will normalize the input by subtracting `image_mean`
                and dividing by `image_std`.
            image_mean (`Union[float, List[float]]`, *optional*, defaults to `self.image_mean`):
                The mean to subtract from the input when normalizing. Only has an effect if `do_normalize` is set to
                `True`.
            image_std (`Union[float, List[float]]`, *optional*, defaults to `self.image_std`):
                The standard deviation to divide the input by when normalizing. Only has an effect if `do_normalize` is
                set to `True`.
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
                - Unset: defaults to the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

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
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
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

        if do_resize:
            images = [
                self.resize(image, size=size, resample=resample, input_data_format=input_data_format)
                for image in images
            ]

        if do_center_crop:
            images = [
                self.center_crop(image, crop_size=crop_size, input_data_format=input_data_format) for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image, rescale_factor=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]
        encoded_inputs = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)
        return encoded_inputs

    def post_process(self, outputs, target_sizes):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation). For visualization, this should be the image size after data
                augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # TODO: (amy) add support for other frameworks
        warnings.warn(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
            FutureWarning,
        )

        logits, boxes = outputs.logits, outputs.pred_boxes

        if len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results

    def post_process_object_detection(
        self, outputs, threshold: float = 0.1, target_sizes: Union[TensorType, List[Tuple]] = None
    ):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
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
        # TODO: (amy) add support for other frameworks
        logits, boxes = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(boxes)

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

    # TODO: (Amy) Make compatible with other frameworks
    def post_process_image_guided_detection(self, outputs, threshold=0.0, nms_threshold=0.3, target_sizes=None):
        """
        Converts the output of [`OwlViTForObjectDetection.image_guided_detection`] into the format expected by the COCO
        api.

        Args:
            outputs ([`OwlViTImageGuidedObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.0):
                Minimum confidence threshold to use to filter out predicted boxes.
            nms_threshold (`float`, *optional*, defaults to 0.3):
                IoU threshold for non-maximum suppression of overlapping boxes.
            target_sizes (`torch.Tensor`, *optional*):
                Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in
                the batch. If set, predicted normalized bounding boxes are rescaled to the target sizes. If left to
                None, predictions will not be unnormalized.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model. All labels are set to None as
            `OwlViTForObjectDetection.image_guided_detection` perform one-shot object detection.
        """
        logits, target_boxes = outputs.logits, outputs.target_pred_boxes

        if target_sizes is not None and len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if target_sizes is not None and target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)

        # Convert to [x0, y0, x1, y1] format
        target_boxes = center_to_corners_format(target_boxes)

        # Apply non-maximum suppression (NMS)
        if nms_threshold < 1.0:
            for idx in range(target_boxes.shape[0]):
                for i in torch.argsort(-scores[idx]):
                    if not scores[idx][i]:
                        continue

                    ious = box_iou(target_boxes[idx][i, :].unsqueeze(0), target_boxes[idx])[0][0]
                    ious[i] = -1.0  # Mask self-IoU.
                    scores[idx][ious > nms_threshold] = 0.0

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.tensor([i[0] for i in target_sizes])
                img_w = torch.tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(target_boxes.device)
            target_boxes = target_boxes * scale_fct[:, None, :]

        # Compute box display alphas based on prediction scores
        results = []
        alphas = torch.zeros_like(scores)

        for idx in range(target_boxes.shape[0]):
            # Select scores for boxes matching the current query:
            query_scores = scores[idx]
            if not query_scores.nonzero().numel():
                continue

            # Apply threshold on scores before scaling
            query_scores[query_scores < threshold] = 0.0

            # Scale box alpha such that the best box for each query has alpha 1.0 and the worst box has alpha 0.1.
            # All other boxes will either belong to a different query, or will not be shown.
            max_score = torch.max(query_scores) + 1e-6
            query_alphas = (query_scores - (max_score * 0.1)) / (max_score * 0.9)
            query_alphas = torch.clip(query_alphas, 0.0, 1.0)
            alphas[idx] = query_alphas

            mask = alphas[idx] > 0
            box_scores = alphas[idx][mask]
            boxes = target_boxes[idx][mask]
            results.append({"scores": box_scores, "labels": None, "boxes": boxes})

        return results


__all__ = ["OwlViTImageProcessor"]

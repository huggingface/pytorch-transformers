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
"""
Processor class for OmDet-Turbo.
"""

import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from ...feature_extraction_utils import BatchFeature
from ...image_transforms import center_to_corners_format
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    is_torch_available,
    is_torchvision_available,
)
from ...utils.deprecation import deprecate_kwarg


if TYPE_CHECKING:
    from .modeling_omdet_turbo import OmDetTurboObjectDetectionOutput


class OmDetTurboTextKwargs(TextKwargs, total=False):
    task: Optional[Union[str, List[str], TextInput, PreTokenizedInput]]


if is_torch_available():
    import torch


if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms


class OmDetTurboProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: OmDetTurboTextKwargs
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": 77,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
            "task": None,
        },
        "images_kwargs": {},
    }


class DictWithDeprecationWarning(dict):
    message = (
        "The `classes` key is deprecated for `OmDetTurboProcessor.post_process_grounded_object_detection` "
        "output dict and will be removed in a 4.51.0 version. Please use `text_labels` instead."
    )

    def __getitem__(self, key):
        if key == "classes":
            warnings.warn(self.message, FutureWarning)
            return super().__getitem__("text_labels")
        return super().__getitem__(key)

    def get(self, key, *args, **kwargs):
        if key == "classes":
            warnings.warn(self.message, FutureWarning)
            return super().get("text_labels", *args, **kwargs)
        return super().get(key, *args, **kwargs)


def clip_boxes(box, box_size: Tuple[int, int]):
    """
    Clip the boxes by limiting x coordinates to the range [0, width]
    and y coordinates to the range [0, height].

    Args:
        box (Tensor): The box to be clipped.
        box_size (height, width): The clipping box's size.
    """
    assert torch.isfinite(box).all(), "Box tensor contains infinite or NaN!"
    height, width = box_size
    x1 = box[:, 0].clamp(min=0, max=width)
    y1 = box[:, 1].clamp(min=0, max=height)
    x2 = box[:, 2].clamp(min=0, max=width)
    y2 = box[:, 3].clamp(min=0, max=height)
    box = torch.stack((x1, y1, x2, y2), dim=-1)

    return box


def compute_score(boxes):
    """
    Compute logit scores per class for each box (proposal) and an array of class indices
    corresponding to each proposal, flattened across the proposal_num.
    The indices in `classes` will later be used to filter and match the predicted classes
    with the input class names.
    """
    num_classes = boxes.shape[2]
    proposal_num = boxes.shape[1]
    scores = torch.sigmoid(boxes)
    classes = torch.arange(num_classes, device=boxes.device).unsqueeze(0).repeat(proposal_num, 1).flatten(0, 1)
    return scores, classes


def _post_process_boxes_for_image(
    boxes: "torch.Tensor",
    scores: "torch.Tensor",
    labels: "torch.Tensor",
    image_num_classes: int,
    image_size: Tuple[int, int],
    threshold: float,
    nms_threshold: float,
    max_num_det: Optional[int] = None,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Filter predicted results using given thresholds and NMS.

    Args:
        boxes (`torch.Tensor`):
            A Tensor of predicted class-specific or class-agnostic boxes for the image.
            Shape (num_queries, max_num_classes_in_batch * 4) if doing class-specific regression,
            or (num_queries, 4) if doing class-agnostic regression.
        scores (`torch.Tensor` of shape (num_queries, max_num_classes_in_batch + 1)):
            A Tensor of predicted class scores for the image.
        labels (`torch.Tensor` of shape (num_queries * (max_num_classes_in_batch + 1),)):
            A Tensor of predicted labels for the image.
        image_num_classes (`int`):
            The number of classes queried for detection on the image.
        image_size (`Tuple[int, int]`):
            A tuple of (height, width) for the image.
        threshold (`float`):
            Only return detections with a confidence score exceeding this threshold.
        nms_threshold (`float`):
            The threshold to use for box non-maximum suppression. Value in [0, 1].
        max_num_det (`int`, *optional*):
            The maximum number of detections to return. Default is None.

    Returns:
        Tuple: A tuple with the following:
            "boxes" (Tensor): A tensor of shape (num_filtered_objects, 4), containing the predicted boxes in (x1, y1, x2, y2) format.
            "scores" (Tensor): A tensor of shape (num_filtered_objects,), containing the predicted confidence scores for each detection.
            "labels" (Tensor): A tensor of ids, where each id is the predicted class id for the corresponding detection
    """

    # Filter by max number of detections
    proposal_num = len(boxes) if max_num_det is None else max_num_det
    scores_per_image, topk_indices = scores.flatten(0, 1).topk(proposal_num, sorted=False)
    labels_per_image = labels[topk_indices]
    boxes_per_image = boxes.view(-1, 1, 4).repeat(1, scores.shape[1], 1).view(-1, 4)
    boxes_per_image = boxes_per_image[topk_indices]

    # Convert and scale boxes to original image size
    boxes_per_image = center_to_corners_format(boxes_per_image)
    boxes_per_image = boxes_per_image * torch.tensor(image_size[::-1]).repeat(2).to(boxes_per_image.device)

    # Filtering by confidence score
    filter_mask = scores_per_image > threshold  # R x K
    score_keep = filter_mask.nonzero(as_tuple=False).view(-1)
    boxes_per_image = boxes_per_image[score_keep]
    scores_per_image = scores_per_image[score_keep]
    labels_per_image = labels_per_image[score_keep]

    # Ensure we did not overflow to non existing classes
    filter_classes_mask = labels_per_image < image_num_classes
    classes_keep = filter_classes_mask.nonzero(as_tuple=False).view(-1)
    boxes_per_image = boxes_per_image[classes_keep]
    scores_per_image = scores_per_image[classes_keep]
    labels_per_image = labels_per_image[classes_keep]

    # NMS
    keep = batched_nms(boxes_per_image, scores_per_image, labels_per_image, nms_threshold)
    boxes_per_image = boxes_per_image[keep]
    scores_per_image = scores_per_image[keep]
    labels_per_image = labels_per_image[keep]

    # Clip to image size
    boxes_per_image = clip_boxes(boxes_per_image, image_size)

    return boxes_per_image, scores_per_image, labels_per_image


class OmDetTurboProcessor(ProcessorMixin):
    r"""
    Constructs a OmDet-Turbo processor which wraps a Deformable DETR image processor and an AutoTokenizer into a
    single processor.

    [`OmDetTurboProcessor`] offers all the functionalities of [`DetrImageProcessor`] and
    [`AutoTokenizer`]. See the docstring of [`~OmDetTurboProcessor.__call__`] and [`~OmDetTurboProcessor.decode`]
    for more information.

    Args:
        image_processor (`DetrImageProcessor`):
            An instance of [`DetrImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "DetrImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[List[str], List[List[str]]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[OmDetTurboProcessorKwargs],
    ) -> BatchFeature:
        """
        This method uses [*DetrImageProcessor.__call__] method to prepare image(s) for the model, and
        [CLIPTokenizerFast.__call__] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.

        Args:
            images (`ImageInput`):
               Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255.
            text (`Union[str, List[str], List[List[str]]]`):
                The classes used to limit the scope of the open vocabulary detection. Expects a list of strings or a list
                of list of strings. Batched classes can be of different lengths.
                Examples: ["cat", "dog", "bird"], [["cat", "dog", "bird"], ["hat", "person"], ["car"]]
        Kwargs:
            task (`Union[str, List[str], TextInput, PreTokenizedInput]`):
                The grounded text used to guide open vocabulary detection. Expects a single string or a list of strings.
                Examples: "Detect a cat, a dog, and a bird.",[ "Detect everything.", "Detect trees and flowers."]
                When not provided, the default task is "Detect [class1], [class2], [class3]" etc.
            ...
        """
        if images is None or text is None:
            raise ValueError("You have to specify both `images` and `text`")

        output_kwargs = self._merge_kwargs(
            OmDetTurboProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = text.strip(" ").split(",")

        if not (len(text) and isinstance(text[0], (list, tuple))):
            text = [text]

        task = output_kwargs["text_kwargs"].pop("task", None)
        if task is None:
            task = ["Detect {}.".format(", ".join(text_single)) for text_single in text]
        elif not isinstance(task, (list, tuple)):
            task = [task]

        encoding_image_processor = self.image_processor(images, **output_kwargs["images_kwargs"])
        tasks_encoding = self.tokenizer(text=task, **output_kwargs["text_kwargs"])

        classes = text

        classes_structure = torch.tensor([len(class_single) for class_single in classes], dtype=torch.long)
        classes_flattened = [class_single for class_batch in classes for class_single in class_batch]
        classes_encoding = self.tokenizer(text=classes_flattened, **output_kwargs["text_kwargs"])

        encoding = BatchFeature()
        encoding.update({f"tasks_{key}": value for key, value in tasks_encoding.items()})
        encoding.update({f"classes_{key}": value for key, value in classes_encoding.items()})
        encoding.update({"classes_structure": classes_structure})
        encoding.update(encoding_image_processor)

        return encoding

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.batch_decode with BertTokenizerFast->PreTrainedTokenizer
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.decode with BertTokenizerFast->PreTrainedTokenizer
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def _get_default_image_size(self) -> Tuple[int, int]:
        height = (
            self.image_processor.size["height"]
            if "height" in self.image_processor.size
            else self.image_processor.size["shortest_edge"]
        )
        width = (
            self.image_processor.size["width"]
            if "width" in self.image_processor.size
            else self.image_processor.size["longest_edge"]
        )
        return height, width

    @deprecate_kwarg("score_threshold", new_name="threshold", version="4.51.0")
    @deprecate_kwarg("classes", new_name="text_labels", version="4.51.0")
    def post_process_grounded_object_detection(
        self,
        outputs: "OmDetTurboObjectDetectionOutput",
        text_labels: Optional[Union[List[str], List[List[str]]]] = None,
        threshold: float = 0.3,
        nms_threshold: float = 0.5,
        target_sizes: Optional[Union[TensorType, List[Tuple]]] = None,
        max_num_det: Optional[int] = None,
    ):
        """
        Converts the raw output of [`OmDetTurboForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format and get the associated text class.

        Args:
            outputs ([`OmDetTurboObjectDetectionOutput`]):
                Raw outputs of the model.
            text_labels (Union[List[str], List[List[str]]], *optional*):
                The input classes names. If not provided, `text_labels` will be set to `None` in `outputs`.
            threshold (float, defaults to 0.3):
                Only return detections with a confidence score exceeding this threshold.
            nms_threshold (float, defaults to 0.5):
                The threshold to use for box non-maximum suppression. Value in [0, 1].
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
            max_num_det (`int`, *optional*):
                The maximum number of detections to return.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, classes and boxes for an image
            in the batch as predicted by the model.
        """

        batch_size = len(outputs.decoder_coord_logits)

        # Inputs consistency check for target sizes
        if target_sizes is None:
            height, width = self._get_default_image_size()
            target_sizes = [(height, width)] * batch_size

        if any(len(image_size) != 2 for image_size in target_sizes):
            raise ValueError(
                "Each element of target_sizes must contain the size (height, width) of each image of the batch"
            )

        if len(target_sizes) != batch_size:
            raise ValueError("Make sure that you pass in as many target sizes as output sequences")

        # Inputs consistency check for text labels
        if text_labels is not None and isinstance(text_labels[0], str):
            text_labels = [text_labels]

        if text_labels is not None and len(text_labels) != batch_size:
            raise ValueError("Make sure that you pass in as many classes group as output sequences")

        # Convert target_sizes to list for easier handling
        if isinstance(target_sizes, torch.Tensor):
            target_sizes = target_sizes.tolist()

        batch_boxes = outputs.decoder_coord_logits
        batch_logits = outputs.decoder_class_logits
        batch_num_classes = outputs.classes_structure

        batch_scores, batch_labels = compute_score(batch_logits)

        results = []
        for boxes, scores, image_size, image_num_classes in zip(
            batch_boxes, batch_scores, target_sizes, batch_num_classes
        ):
            boxes, scores, labels = _post_process_boxes_for_image(
                boxes=boxes,
                scores=scores,
                labels=batch_labels,
                image_num_classes=image_num_classes,
                image_size=image_size,
                threshold=threshold,
                nms_threshold=nms_threshold,
                max_num_det=max_num_det,
            )
            result = DictWithDeprecationWarning(
                {"boxes": boxes, "scores": scores, "labels": labels, "text_labels": None}
            )
            results.append(result)

        # Add text labels
        if text_labels is not None:
            for result, image_text_labels in zip(results, text_labels):
                result["text_labels"] = [image_text_labels[idx] for idx in result["labels"]]

        return results


__all__ = ["OmDetTurboProcessor"]

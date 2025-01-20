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
Processor class for Grounding DINO.
"""

import pathlib
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from ...image_processing_utils import BatchFeature
from ...image_transforms import center_to_corners_format
from ...image_utils import AnnotationFormat, ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import TensorType, is_torch_available
from ...utils.deprecation import deprecate_kwarg


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from .modeling_grounding_dino import GroundingDinoObjectDetectionOutput


AnnotationType = Dict[str, Union[int, str, List[Dict]]]


def get_phrases_from_posmap(posmaps, input_ids):
    """Get token ids of phrases from posmaps and input_ids.

    Args:
        posmaps (`torch.BoolTensor` of shape `(num_boxes, hidden_size)`):
            A boolean tensor of text-thresholded logits related to the detected bounding boxes.
        input_ids (`torch.LongTensor`) of shape `(sequence_length, )`):
            A tensor of token ids.
    """
    left_idx = 0
    right_idx = posmaps.shape[-1] - 1

    # Avoiding altering the input tensor
    posmaps = posmaps.clone()

    posmaps[:, 0 : left_idx + 1] = False
    posmaps[:, right_idx:] = False

    token_ids = []
    for posmap in posmaps:
        non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
        token_ids.append([input_ids[i] for i in non_zero_idx])

    return token_ids


def _is_list_of_candidate_labels(text) -> bool:
    """Check that text is list/tuple of strings and each string is a candidate label and not merged candidate labels text.
    Merged candidate labels text is a string with candidate labels separated by a dot.
    """
    if isinstance(text, (list, tuple)):
        return all(isinstance(t, str) and "." not in t for t in text)
    return False


def _merge_candidate_labels_text(text: List[str]) -> str:
    """
    Merge candidate labels text into a single string. Ensure all labels are lowercase.
    For example, ["A cat", "a dog"] -> "a cat. a dog."
    """
    labels = [t.strip().lower() for t in text]  # ensure lowercase
    merged_labels_str = ". ".join(labels) + "."  # join with dot and add a dot at the end
    return merged_labels_str


class DictWithDeprecationWarning(dict):
    message = (
        "The key `labels` is will return integer ids in `GroundingDinoProcessor.post_process_grounded_object_detection` "
        "output since v4.51.0. Use `text_labels` instead to retrieve string object names."
    )

    def __getitem__(self, key):
        if key == "labels":
            warnings.warn(self.message, FutureWarning)
        return super().__getitem__(key)

    def get(self, key, *args, **kwargs):
        if key == "labels":
            warnings.warn(self.message, FutureWarning)
        return super().get(key, *args, **kwargs)


class GroundingDinoImagesKwargs(ImagesKwargs, total=False):
    annotations: Optional[Union[AnnotationType, List[AnnotationType]]]
    return_segmentation_masks: Optional[bool]
    masks_path: Optional[Union[str, pathlib.Path]]
    do_convert_annotations: Optional[bool]
    format: Optional[Union[str, AnnotationFormat]]


class GroundingDinoProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: GroundingDinoImagesKwargs
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": True,
            "return_length": False,
            "verbose": True,
        }
    }


class GroundingDinoProcessor(ProcessorMixin):
    r"""
    Constructs a Grounding DINO processor which wraps a Deformable DETR image processor and a BERT tokenizer into a
    single processor.

    [`GroundingDinoProcessor`] offers all the functionalities of [`GroundingDinoImageProcessor`] and
    [`AutoTokenizer`]. See the docstring of [`~GroundingDinoProcessor.__call__`] and [`~GroundingDinoProcessor.decode`]
    for more information.

    Args:
        image_processor (`GroundingDinoImageProcessor`):
            An instance of [`GroundingDinoImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "GroundingDinoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[GroundingDinoProcessorKwargs],
    ) -> BatchEncoding:
        """
        This method uses [`GroundingDinoImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Args:
            images (`ImageInput`, `List[ImageInput]`, *optional*):
                The image or batch of images to be processed. The image might be either PIL image, numpy array or a torch tensor.
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`, *optional*):
                Candidate labels to be detected on the image. The text might be one of the following:
                - A list of candidate labels (strings) to be detected on the image (e.g. ["a cat", "a dog"]).
                - A batch of candidate labels to be detected on the batch of images (e.g. [["a cat", "a dog"], ["a car", "a person"]]).
                - A merged candidate labels string to be detected on the image, separated by "." (e.g. "a cat. a dog.").
                - A batch of merged candidate labels text to be detected on the batch of images (e.g. ["a cat. a dog.", "a car. a person."]).
        """
        if images is None and text is None:
            raise ValueError("You must specify either text or images.")

        output_kwargs = self._merge_kwargs(
            GroundingDinoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Get only text
        if images is not None:
            encoding_image_processor = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            encoding_image_processor = BatchFeature()

        if text is not None:
            text = self._preprocess_input_text(text)
            text_encoding = self.tokenizer(
                text=text,
                **output_kwargs["text_kwargs"],
            )
        else:
            text_encoding = BatchEncoding()

        text_encoding.update(encoding_image_processor)

        return text_encoding

    def _preprocess_input_text(self, text):
        """
        Preprocess input text to ensure that labels are in the correct format for the model.
        If the text is a list of candidate labels, merge the candidate labels into a single string,
        for example, ["a cat", "a dog"] -> "a cat. a dog.". In case candidate labels are already in a form of
        "a cat. a dog.", the text is returned as is.
        """

        if _is_list_of_candidate_labels(text):
            text = _merge_candidate_labels_text(text)

        # for batched input
        elif isinstance(text, (list, tuple)) and all(_is_list_of_candidate_labels(t) for t in text):
            text = [_merge_candidate_labels_text(sample) for sample in text]

        return text

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

    @property
    # Copied from transformers.models.blip.processing_blip.BlipProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @deprecate_kwarg("box_threshold", new_name="threshold", version="4.51.0")
    def post_process_grounded_object_detection(
        self,
        outputs: "GroundingDinoObjectDetectionOutput",
        input_ids: Optional[TensorType] = None,
        threshold: float = 0.25,
        text_threshold: float = 0.25,
        target_sizes: Optional[Union[TensorType, List[Tuple]]] = None,
        text_labels: Optional[List[List[str]]] = None,
    ):
        """
        Converts the raw output of [`GroundingDinoForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format and get the associated text label.

        Args:
            outputs ([`GroundingDinoObjectDetectionOutput`]):
                Raw outputs of the model.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The token ids of the input text. If not provided will be taken from the model output.
            threshold (`float`, *optional*, defaults to 0.25):
                Threshold to keep object detection predictions based on confidence score.
            text_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep text detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
            text_labels (`List[List[str]]`, *optional*):
                List of candidate labels to be detected on each image. At the moment it's *NOT used*, but required
                to be in signature for the zero-shot object detection pipeline. Text labels are instead extracted
                from the `input_ids` tensor provided in `outputs`.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the
                - **scores**: tensor of confidence scores for detected objects
                - **boxes**: tensor of bounding boxes in [x0, y0, x1, y1] format
                - **labels**: list of text labels for each detected object (will be replaced with integer ids in v4.51.0)
                - **text_labels**: list of text labels for detected objects
        """
        batch_logits, batch_boxes = outputs.logits, outputs.pred_boxes
        input_ids = input_ids if input_ids is not None else outputs.input_ids

        if target_sizes is not None and len(target_sizes) != len(batch_logits):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")

        batch_probs = torch.sigmoid(batch_logits)  # (batch_size, num_queries, 256)
        batch_scores = torch.max(batch_probs, dim=-1)[0]  # (batch_size, num_queries)

        # Convert to [x0, y0, x1, y1] format
        batch_boxes = center_to_corners_format(batch_boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(batch_boxes.device)
            batch_boxes = batch_boxes * scale_fct[:, None, :]

        results = []
        for idx, (scores, boxes, probs) in enumerate(zip(batch_scores, batch_boxes, batch_probs)):
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]

            # extract text labels
            prob = probs[keep]
            label_ids = get_phrases_from_posmap(prob > text_threshold, input_ids[idx])
            objects_text_labels = self.batch_decode(label_ids)

            result = DictWithDeprecationWarning(
                {
                    "scores": scores,
                    "boxes": boxes,
                    "text_labels": objects_text_labels,
                    # TODO: @pavel, set labels to None since v4.51.0 or find a way to extract ids
                    "labels": objects_text_labels,
                }
            )
            results.append(result)

        return results


__all__ = ["GroundingDinoProcessor"]

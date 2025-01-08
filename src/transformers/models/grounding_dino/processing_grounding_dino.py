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
from typing import Dict, List, Optional, Tuple, Union

from ...image_processing_utils import BatchFeature
from ...image_transforms import center_to_corners_format
from ...image_utils import AnnotationFormat, ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import TensorType, is_torch_available


if is_torch_available():
    import torch


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

        Please refer to the docstring of the above two methods for more information.
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
            text_encoding = self.tokenizer(
                text=text,
                **output_kwargs["text_kwargs"],
            )
        else:
            text_encoding = BatchEncoding()

        text_encoding.update(encoding_image_processor)

        return text_encoding

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

    def post_process_grounded_object_detection(
        self,
        outputs,
        input_ids,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        target_sizes: Union[TensorType, List[Tuple]] = None,
    ):
        """
        Converts the raw output of [`GroundingDinoForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format and get the associated text label.

        Args:
            outputs ([`GroundingDinoObjectDetectionOutput`]):
                Raw outputs of the model.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The token ids of the input text.
            box_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep object detection predictions.
            text_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep text detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        probs = torch.sigmoid(logits)  # (batch_size, num_queries, 256)
        scores = torch.max(probs, dim=-1)[0]  # (batch_size, num_queries)

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
        for idx, (s, b, p) in enumerate(zip(scores, boxes, probs)):
            score = s[s > box_threshold]
            box = b[s > box_threshold]
            prob = p[s > box_threshold]
            label_ids = get_phrases_from_posmap(prob > text_threshold, input_ids[idx])
            label = self.batch_decode(label_ids)
            results.append({"scores": score, "labels": label, "boxes": box})

        return results


__all__ = ["GroundingDinoProcessor"]

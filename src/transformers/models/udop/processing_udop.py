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
Processor class for UDOP.
"""

from typing import List, Optional, Union

from transformers import logging

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


logger = logging.get_logger(__name__)


class UdopTextKwargs(TextKwargs, total=False):
    word_labels: Optional[Union[List[int], List[List[int]]]]
    boxes: Union[List[List[int]], List[List[List[int]]]]


class UdopProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: UdopTextKwargs
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "truncation": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_length": False,
            "verbose": True,
        },
        "images_kwargs": {},
    }


class UdopProcessor(ProcessorMixin):
    r"""
    Constructs a UDOP processor which combines a LayoutLMv3 image processor and a UDOP tokenizer into a single processor.

    [`UdopProcessor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`LayoutLMv3ImageProcessor`] to resize, rescale and normalize document images, and optionally applies OCR
    to get words and normalized bounding boxes. These are then provided to [`UdopTokenizer`] or [`UdopTokenizerFast`],
    which turns the words and bounding boxes into token-level `input_ids`, `attention_mask`, `token_type_ids`, `bbox`.
    Optionally, one can provide integer `word_labels`, which are turned into token-level `labels` for token
    classification tasks (such as FUNSD, CORD).

    Additionally, it also supports passing `text_target` and `text_pair_target` to the tokenizer, which can be used to
    prepare labels for language modeling tasks.

    Args:
        image_processor (`LayoutLMv3ImageProcessor`):
            An instance of [`LayoutLMv3ImageProcessor`]. The image processor is a required input.
        tokenizer (`UdopTokenizer` or `UdopTokenizerFast`):
            An instance of [`UdopTokenizer`] or [`UdopTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LayoutLMv3ImageProcessor"
    tokenizer_class = ("UdopTokenizer", "UdopTokenizerFast")
    # For backward compatibility. See transformers.processing_utils.ProcessorMixin.prepare_and_validate_optional_call_args for more details.
    optional_call_args = ["text_pair"]

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        # The following is to capture `text_pair` argument that may be passed as a positional argument.
        # See transformers.processing_utils.ProcessorMixin.prepare_and_validate_optional_call_args for more details,
        # or this conversation for more context: https://github.com/huggingface/transformers/pull/32544#discussion_r1720208116
        # This behavior is only needed for backward compatibility and will be removed in future versions.
        #
        *args,
        audio=None,
        videos=None,
        **kwargs: Unpack[UdopProcessorKwargs],
    ) -> BatchFeature:
        """
        This method first forwards the `images` argument to [`~UdopImageProcessor.__call__`]. In case
        [`UdopImageProcessor`] was initialized with `apply_ocr` set to `True`, it passes the obtained words and
        bounding boxes along with the additional arguments to [`~UdopTokenizer.__call__`] and returns the output,
        together with the prepared `pixel_values`. In case [`UdopImageProcessor`] was initialized with `apply_ocr` set
        to `False`, it passes the words (`text`/``text_pair`) and `boxes` specified by the user along with the
        additional arguments to [`~UdopTokenizer.__call__`] and returns the output, together with the prepared
        `pixel_values`.

        Alternatively, one can pass `text_target` and `text_pair_target` to prepare the targets of UDOP.

        Please refer to the docstring of the above two methods for more information.
        """
        # verify input
        output_kwargs = self._merge_kwargs(
            UdopProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
            **self.prepare_and_validate_optional_call_args(*args),
        )

        boxes = output_kwargs["text_kwargs"].pop("boxes", None)
        word_labels = output_kwargs["text_kwargs"].pop("word_labels", None)
        text_pair = output_kwargs["text_kwargs"].pop("text_pair", None)
        return_overflowing_tokens = output_kwargs["text_kwargs"].get("return_overflowing_tokens", False)
        return_offsets_mapping = output_kwargs["text_kwargs"].get("return_offsets_mapping", False)
        text_target = output_kwargs["text_kwargs"].get("text_target", None)

        if self.image_processor.apply_ocr and (boxes is not None):
            raise ValueError(
                "You cannot provide bounding boxes if you initialized the image processor with apply_ocr set to True."
            )

        if self.image_processor.apply_ocr and (word_labels is not None):
            raise ValueError(
                "You cannot provide word labels if you initialized the image processor with apply_ocr set to True."
            )

        if return_overflowing_tokens and not return_offsets_mapping:
            raise ValueError("You cannot return overflowing tokens without returning the offsets mapping.")

        if text_target is not None:
            # use the processor to prepare the targets of UDOP
            return self.tokenizer(
                **output_kwargs["text_kwargs"],
            )

        else:
            # use the processor to prepare the inputs of UDOP
            # first, apply the image processor
            features = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            features_words = features.pop("words", None)
            features_boxes = features.pop("boxes", None)

            output_kwargs["text_kwargs"].pop("text_target", None)
            output_kwargs["text_kwargs"].pop("text_pair_target", None)
            output_kwargs["text_kwargs"]["text_pair"] = text_pair
            output_kwargs["text_kwargs"]["boxes"] = boxes if boxes is not None else features_boxes
            output_kwargs["text_kwargs"]["word_labels"] = word_labels

            # second, apply the tokenizer
            if text is not None and self.image_processor.apply_ocr and text_pair is None:
                if isinstance(text, str):
                    text = [text]  # add batch dimension (as the image processor always adds a batch dimension)
                output_kwargs["text_kwargs"]["text_pair"] = features_words

            encoded_inputs = self.tokenizer(
                text=text if text is not None else features_words,
                **output_kwargs["text_kwargs"],
            )

            # add pixel values
            if return_overflowing_tokens is True:
                features["pixel_values"] = self.get_overflowing_images(
                    features["pixel_values"], encoded_inputs["overflow_to_sample_mapping"]
                )
            features.update(encoded_inputs)

            return features

    # Copied from transformers.models.layoutlmv3.processing_layoutlmv3.LayoutLMv3Processor.get_overflowing_images
    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        # in case there's an overflow, ensure each `input_ids` sample is mapped to its corresponding image
        images_with_overflow = []
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])

        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )

        return images_with_overflow

    # Copied from transformers.models.layoutlmv3.processing_layoutlmv3.LayoutLMv3Processor.batch_decode
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.layoutlmv3.processing_layoutlmv3.LayoutLMv3Processor.decode
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["pixel_values", "input_ids", "bbox", "attention_mask"]


__all__ = ["UdopProcessor"]

# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for Pix2Struct.
"""

from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import logging


class Pix2StructImagesKwargs(ImagesKwargs, total=False):
    max_patches: Optional[int]
    header_text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]


class Pix2StructProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Pix2StructImagesKwargs
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
        },
        "images_kwargs": {
            "max_patches": 2048,
        },
    }


logger = logging.get_logger(__name__)


class Pix2StructProcessor(ProcessorMixin):
    r"""
    Constructs a PIX2STRUCT processor which wraps a BERT tokenizer and PIX2STRUCT image processor into a single
    processor.

    [`Pix2StructProcessor`] offers all the functionalities of [`Pix2StructImageProcessor`] and [`T5TokenizerFast`]. See
    the docstring of [`~Pix2StructProcessor.__call__`] and [`~Pix2StructProcessor.decode`] for more information.

    Args:
        image_processor (`Pix2StructImageProcessor`):
            An instance of [`Pix2StructImageProcessor`]. The image processor is a required input.
        tokenizer (Union[`T5TokenizerFast`, `T5Tokenizer`]):
            An instance of ['T5TokenizerFast`] or ['T5Tokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Pix2StructImageProcessor"
    tokenizer_class = ("T5Tokenizer", "T5TokenizerFast")

    def __init__(self, image_processor, tokenizer):
        tokenizer.return_token_type_ids = False
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images=None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[Pix2StructProcessorKwargs],
    ) -> Union[BatchEncoding, BatchFeature]:
        """
        This method uses [`Pix2StructImageProcessor.preprocess`] method to prepare image(s) for the model, and
        [`T5TokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        legacy = kwargs.pop("legacy", True)
        if legacy:
            logger.warning_once(
                "Legacy behavior is being used. The current behavior will be deprecated in version 5.0.0. "
                "In the new behavior, If both images and text are provided, image_processor is not a VQA processor, and `add_special_tokens` is unset, "
                "the default value of `add_special_tokens` will be changed to `False` when calling the tokenizer. "
                "To test the new behavior, set `legacy=False`as a processor call argument."
            )

        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        output_kwargs = self._merge_kwargs(
            Pix2StructProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        add_special_tokens = output_kwargs["text_kwargs"].pop("add_special_tokens", None)
        # Get only text
        if images is None and not self.image_processor.is_vqa:
            output_kwargs["text_kwargs"]["add_special_tokens"] = (
                add_special_tokens if add_special_tokens is not None else True
            )
            self.current_processor = self.tokenizer
            text_encoding = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
            return text_encoding

        if not self.image_processor.is_vqa:
            # add pixel_values
            encoding_image_processor = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            # add pixel_values and bbox
            output_kwargs["images_kwargs"].setdefault("header_text", text)
            encoding_image_processor = self.image_processor(images, **output_kwargs["images_kwargs"])

        if text is not None and not self.image_processor.is_vqa:
            output_kwargs["text_kwargs"]["add_special_tokens"] = (
                add_special_tokens if add_special_tokens is not None else legacy
            )
            text_encoding = self.tokenizer(text=text, **output_kwargs["text_kwargs"])

            if "attention_mask" in text_encoding:
                text_encoding["decoder_attention_mask"] = text_encoding.pop("attention_mask")
            if "input_ids" in text_encoding:
                text_encoding["decoder_input_ids"] = text_encoding.pop("input_ids")
        else:
            text_encoding = None

        if text_encoding is not None:
            encoding_image_processor.update(text_encoding)

        return encoding_image_processor

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Pix2StructTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Pix2StructTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["Pix2StructProcessor"]

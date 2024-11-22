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
Processor class for InstructBLIP. Largely copy of Blip2Processor with addition of a tokenizer for the Q-Former.
"""

import os
from typing import List, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
)
from ...utils import logging
from ..auto import AutoTokenizer


logger = logging.get_logger(__name__)


class InstructBlipProcessorKwargs(ProcessingKwargs, total=False):
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
        "images_kwargs": {},
    }


class InstructBlipProcessor(ProcessorMixin):
    r"""
    Constructs an InstructBLIP processor which wraps a BLIP image processor and a LLaMa/T5 tokenizer into a single
    processor.

    [`InstructBlipProcessor`] offers all the functionalities of [`BlipImageProcessor`] and [`AutoTokenizer`]. See the
    docstring of [`~BlipProcessor.__call__`] and [`~BlipProcessor.decode`] for more information.

    Args:
        image_processor (`BlipImageProcessor`):
            An instance of [`BlipImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
        qformer_tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The Q-Former tokenizer is a required input.
        num_query_tokens (`int`, *optional*):"
            Number of tokens used by the Qformer as queries, should be same as in model's config.
    """

    attributes = ["image_processor", "tokenizer", "qformer_tokenizer"]
    valid_kwargs = ["num_query_tokens"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"
    qformer_tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, qformer_tokenizer, num_query_tokens=None, **kwargs):
        if not hasattr(tokenizer, "image_token"):
            self.image_token = AddedToken("<image>", normalized=False, special=True)
            tokenizer.add_tokens([self.image_token], special_tokens=True)
        else:
            self.image_token = tokenizer.image_token
        self.num_query_tokens = num_query_tokens
        super().__init__(image_processor, tokenizer, qformer_tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[InstructBlipProcessorKwargs],
    ) -> BatchFeature:
        """
        This method uses [`BlipImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        Args:
            images (`ImageInput`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least images or text.")

        output_kwargs = self._merge_kwargs(
            InstructBlipProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        encoding = BatchFeature()

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            # we have to concatenate lists - so we keep track of return_tensors here
            return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
            _text_encoding = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
            output_kwargs["text_kwargs"]["return_tensors"] = return_tensors
            # if we know how many query tokens, expand text inside processor. We need this hacky manipulation
            # because BLIP expects image tokens to be at the beginning even before BOS token
            if self.num_query_tokens is not None and images is not None:
                text_encoding = {}
                image_tokens = self.image_token.content * self.num_query_tokens
                image_token_encoding = self.tokenizer(
                    [image_tokens] * len(text), add_special_tokens=False, return_tensors=None
                )
                for k in _text_encoding:
                    text_encoding[k] = [
                        img_encoding + txt_encoding
                        for img_encoding, txt_encoding in zip(image_token_encoding[k], _text_encoding[k])
                    ]
            else:
                text_encoding = _text_encoding
                if images is not None:
                    logger.warning_once(
                        "Expanding inputs for image tokens in InstructBLIP should be done in processing. "
                        "Please follow instruction here (https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042) to update your InstructBLIP model. "
                        "Using processors without these attributes in the config is deprecated and will throw an error in v4.50."
                    )

            # cast to desired return tensors type after concatenating
            text_encoding = BatchEncoding(text_encoding, tensor_type=return_tensors)

            encoding.update(text_encoding)
            qformer_text_encoding = self.qformer_tokenizer(text, **output_kwargs["text_kwargs"])
            encoding["qformer_input_ids"] = qformer_text_encoding.pop("input_ids")
            encoding["qformer_attention_mask"] = qformer_text_encoding.pop("attention_mask")

        if images is not None:
            image_encoding = self.image_processor(images, **output_kwargs["images_kwargs"])
            encoding.update(image_encoding)

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

    @property
    # Copied from transformers.models.blip.processing_blip.BlipProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # overwrite to save the Q-Former tokenizer in a separate folder
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        qformer_tokenizer_path = os.path.join(save_directory, "qformer_tokenizer")
        self.qformer_tokenizer.save_pretrained(qformer_tokenizer_path)

        # We modify the attributes so that only the tokenizer and image processor are saved in the main folder
        qformer_present = "qformer_tokenizer" in self.attributes
        if qformer_present:
            self.attributes.remove("qformer_tokenizer")

        outputs = super().save_pretrained(save_directory, **kwargs)

        if qformer_present:
            self.attributes += ["qformer_tokenizer"]
        return outputs

    # overwrite to load the Q-Former tokenizer from a separate folder
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # if return_unused_kwargs a tuple is returned where the second element is 'unused_kwargs'
        if isinstance(processor, tuple):
            processor = processor[0]
        qformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="qformer_tokenizer")
        processor.qformer_tokenizer = qformer_tokenizer
        return processor

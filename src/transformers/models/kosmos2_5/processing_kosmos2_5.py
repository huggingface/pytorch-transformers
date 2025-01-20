# coding=utf-8
# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
Processor class for Kosmos2_5.
"""

from typing import List, Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import is_torch_available


if is_torch_available():
    import torch


class Kosmos2_5ImagesKwargs(ImagesKwargs, total=False):
    max_patches: Optional[int]
    num_image_tokens: Optional[int]


class Kosmos2_5ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: TextKwargs
    images_kwargs: Kosmos2_5ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "truncation": True,
            "max_length": None,
            "stride": 0,
            "pad_to_multiple_of": None,
            "return_attention_mask": None,
        },
        "images_kwargs": {
            "max_patches": 4096,
            "num_image_tokens": 2048,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class Kosmos2_5Processor(ProcessorMixin):
    r"""
    Constructs a Kosmos2_5 processor which wraps a PreTrainedTokenizerFast and Kosmos2_5 image processor into a single
    processor.

    [`Kosmos2_5Processor`] offers all the functionalities of [`Kosmos2_5ImageProcessor`] and [`PreTrainedTokenizerFast`]. See
    the docstring of [`~Kosmos2_5Processor.__call__`] and [`~Kosmos2_5Processor.decode`] for more information.

    Args:
        image_processor (`Kosmos2_5ImageProcessor`):
            An instance of [`Kosmos2_5ImageProcessor`]. The image processor is a required input.
        tokenizer (Union[`T5TokenizerFast`, `T5Tokenizer`]):
            An instance of ['T5TokenizerFast`] or ['T5Tokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Kosmos2_5ImageProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, image_processor, tokenizer):
        tokenizer.return_token_type_ids = False
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, List[TextInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[Kosmos2_5ProcessorKwargs],
    ) -> BatchFeature:
        """
        This method uses [`Kosmos2_5ImageProcessor.preprocess`] method to prepare image(s) for the model, and
        [`PreTrainedTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.

        The rest of this documentation shows the arguments specific to `Kosmos2_5Processor`.
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        if images is None:
            raise ValueError("Kosmos2_5Processor requires images to be passed.")

        output_kwargs = self._merge_kwargs(
            Kosmos2_5ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        num_image_tokens = output_kwargs["images_kwargs"].setdefault("num_image_tokens", None)

        encoding = BatchFeature()

        if images is not None:
            image_encoding = self.image_processor(images, **output_kwargs["images_kwargs"])
            image_encoding.pop("rows")
            image_encoding.pop("cols")
            encoding.update(image_encoding)

        prompt = "<s><image>" + "<s>" * num_image_tokens + "</image>"

        if text is not None:
            if isinstance(text, str):
                text = [prompt + text]
            else:
                text = [prompt + t for t in text]
            input = self.tokenizer(text, **output_kwargs["text_kwargs"])

            batch_size, seq_len = input.input_ids.shape
            image_embeds_position_mask = [0, -1] + [1] * num_image_tokens + [-1]
            image_embeds_position_mask += [0] * (seq_len - len(image_embeds_position_mask))
            image_embeds_position_mask = (
                torch.LongTensor(image_embeds_position_mask).unsqueeze(0).repeat(batch_size, 1)
            )

            encoding.update(
                {
                    "input_ids": input.input_ids,
                    "attention_mask": input.attention_mask,
                    "image_embeds_position_mask": image_embeds_position_mask,
                }
            )

        return encoding

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Kosmos2_5TokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Kosmos2_5TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["Kosmos2_5Processor"]

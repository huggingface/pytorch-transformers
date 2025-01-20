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
Processor class for BLIP-2.
"""

from typing import List, Optional, Union

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


logger = logging.get_logger(__name__)


class Blip2ProcessorKwargs(ProcessingKwargs, total=False):
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


class Blip2Processor(ProcessorMixin):
    r"""
    Constructs a BLIP-2 processor which wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.

    [`BlipProcessor`] offers all the functionalities of [`BlipImageProcessor`] and [`AutoTokenizer`]. See the docstring
    of [`~BlipProcessor.__call__`] and [`~BlipProcessor.decode`] for more information.

    Args:
        image_processor (`BlipImageProcessor`):
            An instance of [`BlipImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
        num_query_tokens (`int`, *optional*):
            Number of tokens used by the Qformer as queries, should be same as in model's config.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["num_query_tokens"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, num_query_tokens=None, **kwargs):
        tokenizer.return_token_type_ids = False
        self.current_processor = image_processor
        if not hasattr(tokenizer, "image_token"):
            self.image_token = AddedToken("<image>", normalized=False, special=True)
            tokenizer.add_tokens([self.image_token], special_tokens=True)
        else:
            self.image_token = tokenizer.image_token
        self.num_query_tokens = num_query_tokens

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput = None,
        text: Optional[Union[str, List[str], TextInput, PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[Blip2ProcessorKwargs],
    ) -> BatchEncoding:
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
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")
        output_kwargs = self._merge_kwargs(
            Blip2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # BC for explicit return_tensors
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        encoding = BatchFeature(tensor_type=return_tensors)
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            # We need this hacky manipulation because BLIP expects image tokens to be at the beginning even before BOS token
            if output_kwargs["text_kwargs"].get("max_length") is not None:
                output_kwargs["text_kwargs"]["max_length"] -= self.num_query_tokens
            text_encoding = self.tokenizer(text, **output_kwargs["text_kwargs"])

            # Image tokens should not be padded/truncated or prepended with special BOS token
            image_tokens = self.image_token.content * self.num_query_tokens
            output_kwargs["text_kwargs"]["add_special_tokens"] = False
            output_kwargs["text_kwargs"]["padding"] = False
            output_kwargs["text_kwargs"]["truncation"] = False
            image_text_encoding = self.tokenizer(image_tokens, **output_kwargs["text_kwargs"])
            for k in text_encoding:
                text_encoding[k] = [image_text_encoding[k] + sample for sample in text_encoding[k]]

            # cast to desired return tensors type
            encoding.update(BatchEncoding(text_encoding, tensor_type=return_tensors))

        # add pixel_values encoding. If we also have text_encoding, update image encoding and return it.
        # else, return the text encoding.
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


__all__ = ["Blip2Processor"]

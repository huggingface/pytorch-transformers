# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
Processor class for TrOCR.
"""

import warnings
from contextlib import contextmanager
from typing import List, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class TrOCRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


class TrOCRProcessor(ProcessorMixin):
    r"""
    Constructs a TrOCR processor which wraps a vision image processor and a TrOCR tokenizer into a single processor.

    [`TrOCRProcessor`] offers all the functionalities of [`ViTImageProcessor`/`DeiTImageProcessor`] and
    [`RobertaTokenizer`/`XLMRobertaTokenizer`]. See the [`~TrOCRProcessor.__call__`] and [`~TrOCRProcessor.decode`] for
    more information.

    Args:
        image_processor ([`ViTImageProcessor`/`DeiTImageProcessor`], *optional*):
            An instance of [`ViTImageProcessor`/`DeiTImageProcessor`]. The image processor is a required input.
        tokenizer ([`RobertaTokenizer`/`XLMRobertaTokenizer`], *optional*):
            An instance of [`RobertaTokenizer`/`XLMRobertaTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[TrOCRProcessorKwargs],
    ) -> BatchFeature:
        """
        When used in normal mode, this method forwards all its arguments to AutoImageProcessor's
        [`~AutoImageProcessor.__call__`] and returns its output. If used in the context
        [`~TrOCRProcessor.as_target_processor`] this method forwards all its arguments to TrOCRTokenizer's
        [`~TrOCRTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(images, **kwargs)

        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        output_kwargs = self._merge_kwargs(
            TrOCRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.
        """
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your images inputs, or in a separate call."
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor

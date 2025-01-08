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
Speech processor class for Speech2Text
"""

import warnings
from contextlib import contextmanager

from ...processing_utils import ProcessorMixin


class Speech2TextProcessor(ProcessorMixin):
    r"""
    Constructs a Speech2Text processor which wraps a Speech2Text feature extractor and a Speech2Text tokenizer into a
    single processor.

    [`Speech2TextProcessor`] offers all the functionalities of [`Speech2TextFeatureExtractor`] and
    [`Speech2TextTokenizer`]. See the [`~Speech2TextProcessor.__call__`] and [`~Speech2TextProcessor.decode`] for more
    information.

    Args:
        feature_extractor (`Speech2TextFeatureExtractor`):
            An instance of [`Speech2TextFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`Speech2TextTokenizer`):
            An instance of [`Speech2TextTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "Speech2TextFeatureExtractor"
    tokenizer_class = "Speech2TextTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Speech2TextFeatureExtractor's
        [`~Speech2TextFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~Speech2TextProcessor.as_target_processor`] this method forwards all its arguments to Speech2TextTokenizer's
        [`~Speech2TextTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        if "raw_speech" in kwargs:
            warnings.warn("Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.")
            audio = kwargs.pop("raw_speech")
        else:
            audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Speech2TextTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Speech2TextTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Speech2Text.
        """
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False


__all__ = ["Speech2TextProcessor"]

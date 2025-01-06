# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Speech processor class for Whisper
"""

from ...processing_utils import ProcessorMixin


class WhisperProcessor(ProcessorMixin):
    r"""
    Constructs a Whisper processor which wraps a Whisper feature extractor and a Whisper tokenizer into a single
    processor.

    [`WhisperProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`WhisperTokenizer`]. See
    the [`~WhisperProcessor.__call__`] and [`~WhisperProcessor.decode`] for more information.

    Args:
        feature_extractor (`WhisperFeatureExtractor`):
            An instance of [`WhisperFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`WhisperTokenizer`):
            An instance of [`WhisperTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "WhisperTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] and the `text`
        argument to [`~WhisperTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

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
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def get_prompt_ids(self, text: str, return_tensors="np"):
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)


__all__ = ["WhisperProcessor"]

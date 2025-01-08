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
Processor class for CLVP
"""

from ...processing_utils import ProcessorMixin


class ClvpProcessor(ProcessorMixin):
    r"""
    Constructs a CLVP processor which wraps a CLVP Feature Extractor and a CLVP Tokenizer into a single processor.

    [`ClvpProcessor`] offers all the functionalities of [`ClvpFeatureExtractor`] and [`ClvpTokenizer`]. See the
    [`~ClvpProcessor.__call__`], [`~ClvpProcessor.decode`] and [`~ClvpProcessor.batch_decode`] for more information.

    Args:
        feature_extractor (`ClvpFeatureExtractor`):
            An instance of [`ClvpFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`ClvpTokenizer`):
            An instance of [`ClvpTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "ClvpFeatureExtractor"
    tokenizer_class = "ClvpTokenizer"
    model_input_names = [
        "input_ids",
        "input_features",
        "attention_mask",
    ]

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` and `sampling_rate` arguments to [`~ClvpFeatureExtractor.__call__`] and the `text`
        argument to [`~ClvpTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """

        raw_speech = kwargs.pop("raw_speech", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)

        if raw_speech is None and text is None:
            raise ValueError("You need to specify either an `raw_speech` or `text` input to process.")

        if raw_speech is not None:
            inputs = self.feature_extractor(raw_speech, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif raw_speech is None:
            return encodings
        else:
            inputs["input_ids"] = encodings["input_ids"]
            inputs["attention_mask"] = encodings["attention_mask"]
            return inputs

    # Copied from transformers.models.whisper.processing_whisper.WhisperProcessor.batch_decode with Whisper->Clvp
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to ClvpTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.whisper.processing_whisper.WhisperProcessor.decode with Whisper->Clvp
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to ClvpTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["ClvpProcessor"]

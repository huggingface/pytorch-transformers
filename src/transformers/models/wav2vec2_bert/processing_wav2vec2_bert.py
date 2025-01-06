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
Speech processor class for Wav2Vec2-BERT
"""

import warnings
from typing import List, Optional, Union

from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from ..seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer


class Wav2Vec2BertProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


class Wav2Vec2BertProcessor(ProcessorMixin):
    r"""
    Constructs a Wav2Vec2-BERT processor which wraps a Wav2Vec2-BERT feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.

    [`Wav2Vec2Processor`] offers all the functionalities of [`SeamlessM4TFeatureExtractor`] and [`PreTrainedTokenizer`].
    See the docstring of [`~Wav2Vec2Processor.__call__`] and [`~Wav2Vec2Processor.decode`] for more information.

    Args:
        feature_extractor (`SeamlessM4TFeatureExtractor`):
            An instance of [`SeamlessM4TFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            warnings.warn(
                f"Loading a tokenizer inside {cls.__name__} from a config that does not"
                " include a `tokenizer_class` attribute is deprecated and will be "
                "removed in v5. Please add `'tokenizer_class': 'Wav2Vec2CTCTokenizer'`"
                " attribute to either your `config.json` or `tokenizer_config.json` "
                "file to suppress this warning: ",
                FutureWarning,
            )

            feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

            return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(
        self,
        audio: AudioInput = None,
        text: Optional[Union[str, List[str], TextInput, PreTokenizedInput]] = None,
        images=None,
        videos=None,
        **kwargs: Unpack[Wav2Vec2BertProcessorKwargs],
    ):
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `audio`
        and `kwargs` arguments to SeamlessM4TFeatureExtractor's [`~SeamlessM4TFeatureExtractor.__call__`] if `audio` is not
        `None` to pre-process the audio. To prepare the target sequences(s), this method forwards the `text` and `kwargs` arguments to
        PreTrainedTokenizer's [`~PreTrainedTokenizer.__call__`] if `text` is not `None`. Please refer to the doctsring of the above two methods for more information.

        Args:
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case
                of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
                and T the sample length of the audio.

            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **input_features** -- Audio input features to be fed to a model. Returned when `audio` is not `None`.
            - **attention_mask** -- List of indices specifying which timestamps should be attended to by the model when `audio` is not `None`.
            When only `text` is specified, returns the token attention mask.
            - **labels** -- List of token ids to be fed to a model. Returned when both `text` and `audio` are not `None`.
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None` and `audio` is `None`.
        """

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")
        output_kwargs = self._merge_kwargs(
            Wav2Vec2BertProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def pad(self, input_features=None, labels=None, **kwargs):
        """
        If `input_features` is not `None`, this method forwards the `input_features` and `kwargs` arguments to SeamlessM4TFeatureExtractor's [`~SeamlessM4TFeatureExtractor.pad`] to pad the input features.
        If `labels` is not `None`, this method forwards the `labels` and `kwargs` arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.pad`] to pad the label(s).
        Please refer to the doctsring of the above two methods for more information.
        """
        if input_features is None and labels is None:
            raise ValueError("You need to specify either an `input_features` or `labels` input to pad.")

        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, **kwargs)
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            input_features["labels"] = labels["input_ids"]
            return input_features

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["Wav2Vec2BertProcessor"]

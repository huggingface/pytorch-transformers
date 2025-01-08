# coding=utf-8
# Copyright 2023 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for FastSpeech2Conformer."""

import json
import os
from typing import Optional, Tuple

import regex

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging, requires_backends


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class FastSpeech2ConformerTokenizer(PreTrainedTokenizer):
    """
    Construct a FastSpeech2Conformer tokenizer.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The begin of sequence token. Note that for FastSpeech2, it is the same as the `eos_token`.
        eos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The end of sequence token. Note that for FastSpeech2, it is the same as the `bos_token`.
        pad_token (`str`, *optional*, defaults to `"<blank>"`):
            The token used for padding, for example when batching sequences of different lengths.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        should_strip_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to strip the spaces from the list of tokens.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<sos/eos>",
        eos_token="<sos/eos>",
        pad_token="<blank>",
        unk_token="<unk>",
        should_strip_spaces=False,
        **kwargs,
    ):
        requires_backends(self, "g2p_en")

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        import g2p_en

        self.g2p = g2p_en.G2p()

        self.decoder = {v: k for k, v in self.encoder.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            should_strip_spaces=should_strip_spaces,
            **kwargs,
        )

        self.should_strip_spaces = should_strip_spaces

    @property
    def vocab_size(self):
        return len(self.decoder)

    def get_vocab(self):
        "Returns vocab as a dict"
        return dict(self.encoder, **self.added_tokens_encoder)

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # expand symbols
        text = regex.sub(";", ",", text)
        text = regex.sub(":", ",", text)
        text = regex.sub("-", " ", text)
        text = regex.sub("&", "and", text)

        # strip unnecessary symbols
        text = regex.sub(r"[\(\)\[\]\<\>\"]+", "", text)

        # strip whitespaces
        text = regex.sub(r"\s+", " ", text)

        text = text.upper()

        return text, kwargs

    def _tokenize(self, text):
        """Returns a tokenized string."""
        # phonemize
        tokens = self.g2p(text)

        if self.should_strip_spaces:
            tokens = list(filter(lambda s: s != " ", tokens))

        tokens.append(self.eos_token)

        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    # Override since phonemes cannot be converted back to strings
    def decode(self, token_ids, **kwargs):
        logger.warning(
            "Phonemes cannot be reliably converted to a string due to the one-many mapping, converting to tokens instead."
        )
        return self.convert_ids_to_tokens(token_ids)

    # Override since phonemes cannot be converted back to strings
    def convert_tokens_to_string(self, tokens, **kwargs):
        logger.warning(
            "Phonemes cannot be reliably converted to a string due to the one-many mapping, returning the tokens."
        )
        return tokens

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.get_vocab(), ensure_ascii=False))

        return (vocab_file,)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["g2p"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        try:
            import g2p_en

            self.g2p = g2p_en.G2p()
        except ImportError:
            raise ImportError(
                "You need to install g2p-en to use FastSpeech2ConformerTokenizer. "
                "See https://pypi.org/project/g2p-en/ for installation."
            )


__all__ = ["FastSpeech2ConformerTokenizer"]

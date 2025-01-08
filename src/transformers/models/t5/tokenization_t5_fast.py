# coding=utf-8
# Copyright 2018 T5 Authors and HuggingFace Inc. team.
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
"""Tokenization class for model T5."""

import os
import re
import warnings
from shutil import copyfile
from typing import List, Optional, Tuple

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging


if is_sentencepiece_available():
    from .tokenization_t5 import T5Tokenizer
else:
    T5Tokenizer = None


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


# TODO(PVP) - this should be removed in Transformers v5


class T5TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" T5 tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        extra_ids (`int`, *optional*, defaults to 100):
            Add a number of extra ids added to the vocabulary for use as sentinels. These tokens are accessible as
            "<extra_id_{%d}>" where "{%d}" is a number between 0 and extra_ids-1. These tokens can be retrieved by
            calling get_sentinel_tokens method and token ids can be by calling get_sentinel_token_ids method
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
        add_prefix_space (`bool`, *optional*):
            Whether or not the tokenizer should automatically add a prefix space
        from_slow (`book`, *optional*, defaults to `False`):
            Whether or not the tokenizer should be converted from a slow one. If `add_prefix_space` is set, this will be set to `True`.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = T5Tokenizer

    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        add_prefix_space=None,
        **kwargs,
    ):
        # Add extra_ids to the special token list
        if additional_special_tokens is not None:
            extra_tokens = [x for x in additional_special_tokens if "<extra_id_" in str(x)]
            if len(extra_tokens) < 1:
                additional_special_tokens += [f"<extra_id_{i}>" for i in range(extra_ids)]
            elif extra_ids > 0 and extra_ids != len(extra_tokens):
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to T5Tokenizer. In this case the additional_special_tokens must include the extra_ids"
                    " tokens"
                )
        else:
            extra_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
            additional_special_tokens = extra_tokens

        if add_prefix_space is not None:
            logger.warning_once(
                "You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers"
            )
            kwargs["from_slow"] = True

        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    @staticmethod
    def _eventually_correct_t5_max_length(pretrained_model_name_or_path, max_model_length, init_max_model_length):
        if pretrained_model_name_or_path in T5TokenizerFast.max_model_input_sizes:
            deprecated_max_model_length = T5TokenizerFast.max_model_input_sizes[pretrained_model_name_or_path]
            if init_max_model_length is not None and init_max_model_length != max_model_length:
                return init_max_model_length
            elif init_max_model_length is None:
                warnings.warn(
                    "This tokenizer was incorrectly instantiated with a model max length of"
                    f" {deprecated_max_model_length} which will be corrected in Transformers v5.\nFor now, this"
                    " behavior is kept to avoid breaking backwards compatibility when padding/encoding with"
                    " `truncation is True`.\n- Be aware that you SHOULD NOT rely on"
                    f" {pretrained_model_name_or_path} automatically truncating your input to"
                    f" {deprecated_max_model_length} when padding/encoding.\n- If you want to encode/pad to sequences"
                    f" longer than {deprecated_max_model_length} you can either instantiate this tokenizer with"
                    " `model_max_length` or pass `max_length` when encoding/padding.\n- To avoid this warning, please"
                    " instantiate this tokenizer with `model_max_length` set to your preferred value.",
                    FutureWarning,
                )

        return max_model_length

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
            logger.info(f"Copy vocab file to {out_vocab_file}")

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        token_ids_0 = token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0
        else:
            token_ids_1 = token_ids_1 + [self.eos_token_id]
            return self.prefix_tokens + token_ids_0 + token_ids_1

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def get_sentinel_tokens(self):
        return list(
            set(filter(lambda x: bool(re.search(r"<extra_id_\d+>", x)) is not None, self.additional_special_tokens))
        )

    def get_sentinel_token_ids(self):
        return [self.convert_tokens_to_ids(token) for token in self.get_sentinel_tokens()]


__all__ = ["T5TokenizerFast"]

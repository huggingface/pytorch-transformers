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
"""Tokenization class for SigLIP model."""

import os
import re
import string
import warnings
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken


if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput
from ...utils import logging, requires_backends


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


SPIECE_UNDERLINE = "▁"


class SiglipTokenizer(PreTrainedTokenizer):
    """
    Construct a Siglip tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"</s>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
        model_max_length (`int`, *optional*, defaults to 64):
            The maximum length (in number of tokens) for model inputs.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="</s>",
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        model_max_length=64,
        do_lower_case=True,
        **kwargs,
    ) -> None:
        requires_backends(self, "protobuf")

        pad_token = (
            AddedToken(pad_token, rstrip=True, lstrip=True, normalized=False, special=True)
            if isinstance(pad_token, str)
            else pad_token
        )
        unk_token = (
            AddedToken(unk_token, rstrip=True, lstrip=True, normalized=False, special=True)
            if isinstance(unk_token, str)
            else unk_token
        )
        eos_token = (
            AddedToken(eos_token, rstrip=True, lstrip=True, normalized=False, special=True)
            if isinstance(eos_token, str)
            else eos_token
        )

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.do_lower_case = do_lower_case
        self.vocab_file = vocab_file

        self.sp_model = self.get_spm_processor()
        self.vocab_file = vocab_file

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            model_max_length=model_max_length,
            do_lower_case=do_lower_case,
            **kwargs,
        )

    def get_spm_processor(self):
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf()
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    @property
    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.vocab_size
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_vocab
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._add_eos_if_not_present
    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.create_token_type_ids_from_sequences
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

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.build_inputs_with_special_tokens
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
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.__getstate__
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.__setstate__
    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    # source: https://github.com/google-research/big_vision/blob/3b8e5ab6ad4f96e32b32826f9e1b8fd277914f9c/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94
    def canonicalize_text(self, text, *, keep_punctuation_exact_string=None):
        """Returns canonicalized `text` (puncuation removed).

        Args:
            text (`str`):
                String to be canonicalized.
            keep_punctuation_exact_string (`str`, *optional*):
                If provided, then this exact string is kept. For example providing '{}' will keep any occurrences of '{}'
                (but will still remove '{' and '}' that appear separately).
        """
        if keep_punctuation_exact_string:
            text = keep_punctuation_exact_string.join(
                self.remove_punctuation(part) for part in text.split(keep_punctuation_exact_string)
            )
        else:
            text = self.remove_punctuation(text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def tokenize(self, text: "TextInput", add_special_tokens=False, **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens.
        """
        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " "), **kwargs)

        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    @property
    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.unk_token_length
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))

    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE.

        For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give `['H', 'e', 'y']` instead of `['▁He', 'y']`.

        Thus we always encode `f"{unk_token}text"` and strip the `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        text = self.canonicalize_text(text, keep_punctuation_exact_string=None)
        tokens = self.sp_model.encode(text, out_type=str)

        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)


__all__ = ["SiglipTokenizer"]

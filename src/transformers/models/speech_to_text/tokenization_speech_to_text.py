# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Speech2Text."""

import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "spm_file": "sentencepiece.bpe.model",
}


MAX_MODEL_INPUT_SIZES = {
    "facebook/s2t-small-librispeech-asr": 1024,
}

MUSTC_LANGS = ["pt", "fr", "ru", "nl", "ro", "it", "es", "de"]

LANGUAGES = {"mustc": MUSTC_LANGS}


class Speech2TextTokenizer(PreTrainedTokenizer):
    """
    Construct an Speech2Text tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        spm_file (`str`):
            Path to the [SentencePiece](https://github.com/google/sentencepiece) model file
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        do_upper_case (`bool`, *optional*, defaults to `False`):
           Whether or not to uppercase the output when decoding.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        tgt_lang (`str`, *optional*):
            A string representing the target language.
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

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        spm_file,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        do_upper_case=False,
        do_lower_case=False,
        tgt_lang=None,
        lang_codes=None,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.do_upper_case = do_upper_case
        self.do_lower_case = do_lower_case

        self.encoder = load_json(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.spm_file = spm_file
        self.sp_model = load_spm(spm_file, self.sp_model_kwargs)

        if lang_codes is not None:
            self.lang_codes = lang_codes
            self.langs = LANGUAGES[lang_codes]
            self.lang_tokens = [f"<lang:{lang}>" for lang in self.langs]
            self.lang_code_to_id = {lang: self.sp_model.PieceToId(f"<lang:{lang}>") for lang in self.langs}
            if additional_special_tokens is not None:
                additional_special_tokens = self.lang_tokens + additional_special_tokens
            else:
                additional_special_tokens = self.lang_tokens
            self._tgt_lang = tgt_lang if tgt_lang is not None else self.langs[0]

            self.set_tgt_lang_special_tokens(self._tgt_lang)
        else:
            self.lang_code_to_id = {}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            do_upper_case=do_upper_case,
            do_lower_case=do_lower_case,
            tgt_lang=tgt_lang,
            lang_codes=lang_codes,
            sp_model_kwargs=self.sp_model_kwargs,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> Dict:
        vocab = self.encoder.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def tgt_lang(self) -> str:
        return self._tgt_lang

    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang) -> None:
        self._tgt_lang = new_tgt_lang
        self.set_tgt_lang_special_tokens(new_tgt_lang)

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[eos, tgt_lang_code] and suffix=[eos]."""
        lang_code_id = self.lang_code_to_id[tgt_lang]
        self.prefix_tokens = [lang_code_id]

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                decoded = self.sp_model.decode(current_sub_tokens)
                out_string += (decoded.upper() if self.do_upper_case else decoded) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        decoded = self.sp_model.decode(current_sub_tokens)
        out_string += decoded.upper() if self.do_upper_case else decoded
        return out_string.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

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

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1]
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = load_spm(self.spm_file, self.sp_model_kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        save_dir = Path(save_directory)
        assert save_dir.is_dir(), f"{save_directory} should be a directory"
        vocab_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        spm_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["spm_file"]
        )

        save_json(self.encoder, vocab_save_path)

        if os.path.abspath(self.spm_file) != os.path.abspath(spm_save_path) and os.path.isfile(self.spm_file):
            copyfile(self.spm_file, spm_save_path)
        elif not os.path.isfile(self.spm_file):
            with open(spm_save_path, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (str(vocab_save_path), str(spm_save_path))


def load_spm(path: str, sp_model_kwargs: Dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    spm = sentencepiece.SentencePieceProcessor(**sp_model_kwargs)
    spm.Load(str(path))
    return spm


def load_json(path: str) -> Union[Dict, List]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


__all__ = ["Speech2TextTokenizer"]

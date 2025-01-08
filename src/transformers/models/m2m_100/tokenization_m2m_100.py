# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for M2M100."""

import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece

from ...tokenization_utils import BatchEncoding, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "spm_file": "sentencepiece.bpe.model",
    "tokenizer_config_file": "tokenizer_config.json",
}


# fmt: off
FAIRSEQ_LANGUAGE_CODES = {
    "m2m100": ["af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu"],
    "wmt21": ['en', 'ha', 'is', 'ja', 'cs', 'ru', 'zh', 'de']
}
# fmt: on


class M2M100Tokenizer(PreTrainedTokenizer):
    """
    Construct an M2M100 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        spm_file (`str`):
            Path to [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary.
        src_lang (`str`, *optional*):
            A string representing the source language.
        tgt_lang (`str`, *optional*):
            A string representing the target language.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        language_codes (`str`, *optional*, defaults to `"m2m100"`):
            What language codes to use. Should be one of `"m2m100"` or `"wmt21"`.
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

    Examples:

    ```python
    >>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    >>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    >>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="en", tgt_lang="ro")
    >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
    >>> tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"
    >>> model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
    >>> outputs = model(**model_inputs)  # should work
    ```"""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        spm_file,
        src_lang=None,
        tgt_lang=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        language_codes="m2m100",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        num_madeup_words=8,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.language_codes = language_codes
        fairseq_language_code = FAIRSEQ_LANGUAGE_CODES[language_codes]
        self.lang_code_to_token = {lang_code: f"__{lang_code}__" for lang_code in fairseq_language_code}

        additional_special_tokens = kwargs.pop("additional_special_tokens", [])
        for lang_code in fairseq_language_code:
            token = self.get_lang_token(lang_code)
            if token not in additional_special_tokens and lang_code not in str(token) not in self.added_tokens_encoder:
                additional_special_tokens.append(token)

        self.vocab_file = vocab_file
        self.encoder = load_json(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.spm_file = spm_file
        self.sp_model = load_spm(spm_file, self.sp_model_kwargs)

        self.encoder_size = len(self.encoder)

        self.lang_token_to_id = {
            self.get_lang_token(lang_code): self.encoder_size + i for i, lang_code in enumerate(fairseq_language_code)
        }
        self.lang_code_to_id = {lang_code: self.encoder_size + i for i, lang_code in enumerate(fairseq_language_code)}
        self.id_to_lang_token = {v: k for k, v in self.lang_token_to_id.items()}

        self._src_lang = src_lang if src_lang is not None else "en"
        self.tgt_lang = tgt_lang
        self.cur_lang_id = self.get_lang_id(self._src_lang)

        self.num_madeup_words = num_madeup_words

        super().__init__(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            language_codes=language_codes,
            sp_model_kwargs=self.sp_model_kwargs,
            additional_special_tokens=additional_special_tokens,
            num_madeup_words=num_madeup_words,
            **kwargs,
        )
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> Dict:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        if token in self.lang_token_to_id:
            return self.lang_token_to_id[token]
        return self.encoder.get(token, self.encoder[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        if index in self.id_to_lang_token:
            return self.id_to_lang_token[index]
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

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
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An MBART sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `X [eos, src_lang_code]`
        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

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
        if not save_dir.is_dir():
            raise OSError(f"{save_directory} should be a directory")
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

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro",
        **kwargs,
    ) -> BatchEncoding:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self.src_lang)
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _build_translation_inputs(self, raw_inputs, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, **extra_kwargs)
        tgt_lang_id = self.get_lang_id(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def _switch_to_input_mode(self):
        self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        lang_token = self.get_lang_token(src_lang)
        self.cur_lang_id = self.lang_token_to_id[lang_token]
        self.prefix_tokens = [self.cur_lang_id]
        self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        lang_token = self.get_lang_token(tgt_lang)
        self.cur_lang_id = self.lang_token_to_id[lang_token]
        self.prefix_tokens = [self.cur_lang_id]
        self.suffix_tokens = [self.eos_token_id]

    def get_lang_token(self, lang: str) -> str:
        return self.lang_code_to_token[lang]

    def get_lang_id(self, lang: str) -> int:
        lang_token = self.get_lang_token(lang)
        return self.lang_token_to_id[lang_token]


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


__all__ = ["M2M100Tokenizer"]

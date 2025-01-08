# coding=utf-8
# Copyright 2020 Microsoft and the HuggingFace Inc. team.
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
"""Tokenization class for model DeBERTa."""

import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as sp

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "spm.model"}


class DebertaV2Tokenizer(PreTrainedTokenizer):
    r"""
    Constructs a DeBERTa-v2 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        bos_token (`string`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.
            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.
        eos_token (`string`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token. When building a sequence using special tokens, this is not the token that is
            used for the end of sequence. The token used is the `sep_token`.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
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
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        split_by_punct=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.do_lower_case = do_lower_case
        self.split_by_punct = split_by_punct
        self.vocab_file = vocab_file
        self._tokenizer = SPMTokenizer(
            vocab_file, None, split_by_punct=split_by_punct, sp_model_kwargs=self.sp_model_kwargs
        )
        unk_token = AddedToken(unk_token, normalized=True, special=True) if isinstance(unk_token, str) else unk_token
        super().__init__(
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            split_by_punct=split_by_punct,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )
        self._tokenizer.special_tokens = self.all_special_tokens

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab(self):
        return self._tokenizer.vocab

    def get_vocab(self):
        vocab = self.vocab.copy()
        vocab.update(self.get_added_vocab())
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        if self.do_lower_case:
            text = text.lower()
        return self._tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self._tokenizer.spm.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self._tokenizer.spm.IdToPiece(index) if index < self.vocab_size else self.unk_token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        return self._tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

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

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return self._tokenizer.save_pretrained(save_directory, filename_prefix=filename_prefix)


class SPMTokenizer:
    r"""
    Constructs a tokenizer based on [SentencePiece](https://github.com/google/sentencepiece).

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
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
    """

    def __init__(
        self, vocab_file, special_tokens, split_by_punct=False, sp_model_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.split_by_punct = split_by_punct
        self.vocab_file = vocab_file
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"{vocab_file} does not exist!")
        spm.load(vocab_file)
        bpe_vocab_size = spm.GetPieceSize()
        # Token map
        # <unk> 0+1
        # <s> 1+1
        # </s> 2+1
        self.vocab = {spm.IdToPiece(i): i for i in range(bpe_vocab_size)}
        self.ids_to_tokens = [spm.IdToPiece(i) for i in range(bpe_vocab_size)]
        # self.vocab['[PAD]'] = 0
        # self.vocab['[CLS]'] = 1
        # self.vocab['[SEP]'] = 2
        # self.vocab['[UNK]'] = 3

        self.spm = spm
        self.special_tokens = special_tokens

    def __getstate__(self):
        state = self.__dict__.copy()
        state["spm"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        self.spm.Load(self.vocab_file)

    def tokenize(self, text):
        return self._encode_as_pieces(text)

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def decode(self, tokens, start=-1, end=-1, raw_text=None):
        if raw_text is None:
            current_sub_tokens = []
            out_string = ""
            prev_is_special = False
            for token in tokens:
                # make sure that special tokens are not decoded using sentencepiece model
                if token in self.special_tokens:
                    if not prev_is_special:
                        out_string += " "
                    out_string += self.spm.decode_pieces(current_sub_tokens) + token
                    prev_is_special = True
                    current_sub_tokens = []
                else:
                    current_sub_tokens.append(token)
                    prev_is_special = False
            out_string += self.spm.decode_pieces(current_sub_tokens)
            return out_string.strip()
        else:
            words = self.split_to_words(raw_text)
            word_tokens = [self.tokenize(w) for w in words]
            token2words = [0] * len(tokens)
            tid = 0
            for i, w in enumerate(word_tokens):
                for k, t in enumerate(w):
                    token2words[tid] = i
                    tid += 1
            word_start = token2words[start]
            word_end = token2words[end] if end < len(tokens) else len(words)
            text = "".join(words[word_start:word_end])
            return text

    # TODO add a deprecation cycle as this can have different behaviour from our API
    def add_special_token(self, token):
        if token not in self.special_tokens:
            self.special_tokens.append(token)
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) - 1
                self.ids_to_tokens.append(token)
        return self.id(token)

    def part_of_whole_word(self, token, is_bos=False):
        logger.warning_once(
            "The `DebertaTokenizer.part_of_whole_word` method is deprecated and will be removed in `transformers==4.35`"
        )
        if is_bos:
            return True
        if (
            len(token) == 1
            and (_is_whitespace(list(token)[0]) or _is_control(list(token)[0]) or _is_punctuation(list(token)[0]))
        ) or token in self.special_tokens:
            return False

        word_start = b"\xe2\x96\x81".decode("utf-8")
        return not token.startswith(word_start)

    def pad(self):
        return "[PAD]"

    def bos(self):
        return "[CLS]"

    def eos(self):
        return "[SEP]"

    def unk(self):
        return "[UNK]"

    def mask(self):
        return "[MASK]"

    def sym(self, id):
        return self.ids_to_tokens[id]

    def id(self, sym):
        logger.warning_once(
            "The `DebertaTokenizer.id` method is deprecated and will be removed in `transformers==4.35`"
        )
        return self.vocab[sym] if sym in self.vocab else 1

    def _encode_as_pieces(self, text):
        text = convert_to_unicode(text)
        if self.split_by_punct:
            words = self._run_split_on_punc(text)
            pieces = [self.spm.encode(w, out_type=str) for w in words]
            return [p for w in pieces for p in w]
        else:
            return self.spm.encode(text, out_type=str)

    def split_to_words(self, text):
        pieces = self._encode_as_pieces(text)
        word_start = b"\xe2\x96\x81".decode("utf-8")
        words = []
        offset = 0
        prev_end = 0
        for i, p in enumerate(pieces):
            if p.startswith(word_start):
                if offset > prev_end:
                    words.append(text[prev_end:offset])
                prev_end = offset
                w = p.replace(word_start, "")
            else:
                w = p
            try:
                s = text.index(w, offset)
                pn = ""
                k = i + 1
                while k < len(pieces):
                    pn = pieces[k].replace(word_start, "")
                    if len(pn) > 0:
                        break
                    k += 1

                if len(pn) > 0 and pn in text[offset:s]:
                    offset = offset + 1
                else:
                    offset = s + len(w)
            except Exception:
                offset = offset + 1

        if prev_end < offset:
            words.append(text[prev_end:offset])

        return words

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def save_pretrained(self, path: str, filename_prefix: str = None):
        filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
        if filename_prefix is not None:
            filename = filename_prefix + "-" + filename
        full_path = os.path.join(path, filename)
        with open(full_path, "wb") as fs:
            fs.write(self.spm.serialized_model_proto())
        return (full_path,)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise TypeError(f"Unsupported string type: {type(text)}")


__all__ = ["DebertaV2Tokenizer"]

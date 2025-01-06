# coding=utf-8
# Copyright 2019-present CNRS, Facebook Inc. and the HuggingFace Inc. team.
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
"""Tokenization classes for Flaubert."""

import json
import os
import re
import unicodedata
from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """

    def ensure_text(s, encoding="utf-8", errors="strict"):
        if isinstance(s, bytes):
            return s.decode(encoding, errors)
        elif isinstance(s, str):
            return s
        else:
            raise TypeError(f"not expecting type '{type(s)}'")

    return ensure_text(text, encoding="utf-8", errors="ignore")


# Copied from transformers.models.xlm.tokenization_xlm.get_pairs
def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


# Copied from transformers.models.xlm.tokenization_xlm.replace_unicode_punct
def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    text = text.replace("，", ",")
    text = re.sub(r"。\s*", ". ", text)
    text = text.replace("、", ",")
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("∶", ":")
    text = text.replace("：", ":")
    text = text.replace("？", "?")
    text = text.replace("《", '"')
    text = text.replace("》", '"')
    text = text.replace("）", ")")
    text = text.replace("！", "!")
    text = text.replace("（", "(")
    text = text.replace("；", ";")
    text = text.replace("１", "1")
    text = text.replace("」", '"')
    text = text.replace("「", '"')
    text = text.replace("０", "0")
    text = text.replace("３", "3")
    text = text.replace("２", "2")
    text = text.replace("５", "5")
    text = text.replace("６", "6")
    text = text.replace("９", "9")
    text = text.replace("７", "7")
    text = text.replace("８", "8")
    text = text.replace("４", "4")
    text = re.sub(r"．\s*", ". ", text)
    text = text.replace("～", "~")
    text = text.replace("’", "'")
    text = text.replace("…", "...")
    text = text.replace("━", "-")
    text = text.replace("〈", "<")
    text = text.replace("〉", ">")
    text = text.replace("【", "[")
    text = text.replace("】", "]")
    text = text.replace("％", "%")
    return text


# Copied from transformers.models.xlm.tokenization_xlm.remove_non_printing_char
def remove_non_printing_char(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    """
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            continue
        output.append(char)
    return "".join(output)


class FlaubertTokenizer(PreTrainedTokenizer):
    """
    Construct a Flaubert tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization.
    - Normalizing all inputs text.
    - The arguments `special_tokens` and the function `set_special_tokens`, can be used to add additional symbols (like
      "__classify__") to a vocabulary.
    - The argument `do_lowercase` controls lower casing (automatically set for pretrained vocabularies).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Vocabulary file.
        merges_file (`str`):
            Merges file.
        do_lowercase (`bool`, *optional*, defaults to `False`):
            Controls lower casing.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"</s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<special1>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (`List[str]`, *optional*, defaults to `['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>']`):
            List of additional special tokens.
        lang2id (`Dict[str, int]`, *optional*):
            Dictionary mapping languages string identifiers to their IDs.
        id2lang (`Dict[int, str]`, *optional*):
            Dictionary mapping language IDs to their string identifiers.
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        merges_file,
        do_lowercase=False,
        unk_token="<unk>",
        bos_token="<s>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="</s>",
        mask_token="<special1>",
        additional_special_tokens=[
            "<special0>",
            "<special1>",
            "<special2>",
            "<special3>",
            "<special4>",
            "<special5>",
            "<special6>",
            "<special7>",
            "<special8>",
            "<special9>",
        ],
        lang2id=None,
        id2lang=None,
        **kwargs,
    ):
        do_lowercase_and_remove_accent = kwargs.pop("do_lowercase_and_remove_accent", None)
        if do_lowercase_and_remove_accent is not None:
            logger.warning(
                "`do_lowercase_and_remove_accent` is passed as a keyword argument, but this won't do anything."
                " `FlaubertTokenizer` will always set it to `False`."
            )
        # always `False`
        self.do_lowercase_and_remove_accent = False

        self.do_lowercase = do_lowercase

        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use FlaubertTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        self.sm = sacremoses

        # cache of sm.MosesPunctNormalizer instance
        self.cache_moses_punct_normalizer = {}
        # cache of sm.MosesTokenizer instance
        self.cache_moses_tokenizer = {}
        self.lang_with_custom_tokenizer = {"zh", "th", "ja"}
        self.lang2id = lang2id
        self.id2lang = id2lang
        if lang2id is not None and id2lang is not None:
            assert len(lang2id) == len(id2lang)

        self.ja_word_tokenizer = None
        self.zh_word_tokenizer = None

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

        super().__init__(
            do_lowercase=do_lowercase,
            unk_token=unk_token,
            bos_token=bos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            lang2id=lang2id,
            id2lang=id2lang,
            **kwargs,
        )

    @property
    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.do_lower_case
    def do_lower_case(self):
        return self.do_lowercase_and_remove_accent

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_punct_norm
    def moses_punct_norm(self, text, lang):
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = self.sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        return punct_normalizer.normalize(text)

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_tokenize
    def moses_tokenize(self, text, lang):
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_pipeline
    def moses_pipeline(self, text, lang):
        text = replace_unicode_punct(text)
        text = self.moses_punct_norm(text, lang)
        text = remove_non_printing_char(text)
        return text

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.ja_tokenize
    def ja_tokenize(self, text):
        if self.ja_word_tokenizer is None:
            try:
                import Mykytea

                self.ja_word_tokenizer = Mykytea.Mykytea(
                    f"-model {os.path.expanduser('~')}/local/share/kytea/model.bin"
                )
            except (AttributeError, ImportError):
                logger.error(
                    "Make sure you install KyTea (https://github.com/neubig/kytea) and it's python wrapper"
                    " (https://github.com/chezou/Mykytea-python) with the following steps"
                )
                logger.error("1. git clone git@github.com:neubig/kytea.git && cd kytea")
                logger.error("2. autoreconf -i")
                logger.error("3. ./configure --prefix=$HOME/local")
                logger.error("4. make && make install")
                logger.error("5. pip install kytea")
                raise
        return list(self.ja_word_tokenizer.getWS(text))

    @property
    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.vocab_size
    def vocab_size(self):
        return len(self.encoder)

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.get_vocab
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.bpe
    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def preprocess_text(self, text):
        text = text.replace("``", '"').replace("''", '"')
        text = convert_to_unicode(text)
        text = unicodedata.normalize("NFC", text)

        if self.do_lowercase:
            text = text.lower()

        return text

    def _tokenize(self, text, bypass_tokenizer=False):
        """
        Tokenize a string given language code using Moses.

        Details of tokenization:

            - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`

        Args:
            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)
              (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
        lang = "fr"
        if lang and self.lang2id and lang not in self.lang2id:
            logger.error(
                "Supplied language code not found in lang2id mapping. Please check that your language is supported by"
                " the loaded pretrained model."
            )

        if bypass_tokenizer:
            text = text.split()
        else:
            text = self.preprocess_text(text)
            text = self.moses_pipeline(text, lang=lang)
            text = self.moses_tokenize(text, lang=lang)

        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend(list(self.bpe(token).split(" ")))

        return split_tokens

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLM sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

        """
        bos = [self.bos_token_id]
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return bos + token_ids_0 + sep
        return bos + token_ids_0 + sep + token_ids_1 + sep

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.get_special_tokens_mask
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

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLM sequence
        pair mask has the following format:

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

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.__getstate__
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sm"] = None
        return state

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.__setstate__
    def __setstate__(self, d):
        self.__dict__ = d

        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        self.sm = sacremoses


__all__ = ["FlaubertTokenizer"]

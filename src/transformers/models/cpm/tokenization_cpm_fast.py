# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes."""

import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...tokenization_utils_fast import AddedToken, PreTrainedTokenizerFast
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


class CpmTokenizerFast(PreTrainedTokenizerFast):
    """Runs pre-tokenization with Jieba segmentation tool. It is used in CPM models."""

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        additional_special_tokens=["<eop>", "<eod>"],
        **kwargs,
    ):
        """
        Construct a CPM tokenizer. Based on [Jieba](https://pypi.org/project/jieba/) and
        [SentencePiece](https://github.com/google/sentencepiece).

        This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should
        refer to this superclass for more information regarding those methods.

        Args:
            vocab_file (`str`):
                [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
                contains the vocabulary necessary to instantiate a tokenizer.
            do_lower_case (`bool`, *optional*, defaults to `True`):
                Whether to lowercase the input when tokenizing.
            remove_space (`bool`, *optional*, defaults to `True`):
                Whether to strip the text when tokenizing (removing excess spaces before and after the string).
            keep_accents (`bool`, *optional*, defaults to `False`):
                Whether to keep accents when tokenizing.
            bos_token (`str`, *optional*, defaults to `"<s>"`):
                The beginning of sequence token that was used during pretraining. Can be used a sequence classifier
                token.

                <Tip>

                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the `cls_token`.

                </Tip>

            eos_token (`str`, *optional*, defaults to `"</s>"`):
                The end of sequence token.

                <Tip>

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the `sep_token`.

                </Tip>

            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be
                this token instead.
            sep_token (`str`, *optional*, defaults to `"<sep>"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
                for sequence classification or for a text and a question for question answering. It is also used as the
                last token of a sequence built with special tokens.
            pad_token (`str`, *optional*, defaults to `"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
            cls_token (`str`, *optional*, defaults to `"<cls>"`):
                The classifier token which is used when doing sequence classification (classification of the whole
                sequence instead of per-token classification). It is the first token of the sequence when built with
                special tokens.
            mask_token (`str`, *optional*, defaults to `"<mask>"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
            additional_special_tokens (`List[str]`, *optional*, defaults to `["<eop>", "<eod>"]`):
                Additional special tokens used by the tokenizer.

        Attributes:
            sp_model (`SentencePieceProcessor`):
                The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
        """
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self._pad_token_type_id = 3
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        try:
            import jieba
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install jieba to use CpmTokenizer or CpmTokenizerFast. "
                "See https://pypi.org/project/jieba/ for installation."
            )
        self.jieba = jieba
        self.translator = str.maketrans(" \n", "\u2582\u2583")

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # Copied from transformers.models.xlnet.tokenization_xlnet_fast.XLNetTokenizerFast.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLNet sequence has the following format:

        - single sequence: `X <sep> <cls>`
        - pair of sequences: `A <sep> B <sep> <cls>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return token_ids_0 + sep + cls
        return token_ids_0 + sep + token_ids_1 + sep + cls

    # Copied from transformers.models.xlnet.tokenization_xlnet_fast.XLNetTokenizerFast.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet
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
        cls_segment_id = [2]

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0] + cls_segment_id
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id

    # Copied from transformers.models.xlnet.tokenization_xlnet_fast.XLNetTokenizerFast.save_vocabulary
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

        return (out_vocab_file,)

    def _batch_encode_plus(self, batch_text_or_text_pairs, *args, **kwargs):
        batch_text_or_text_pairs = [
            " ".join([x.translate(self.translator) for x in self.jieba.cut(text, cut_all=False)])
            for text in batch_text_or_text_pairs
        ]
        return super()._batch_encode_plus(batch_text_or_text_pairs, *args, **kwargs)

    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = text.replace(" ", "").replace("\u2582", " ").replace("\u2583", "\n")
        return text


__all__ = ["CpmTokenizerFast"]

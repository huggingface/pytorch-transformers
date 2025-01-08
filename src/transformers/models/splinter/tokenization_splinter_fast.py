# coding=utf-8
# Copyright 2021 Tel AViv University, AllenAI and The HuggingFace Inc. team. All rights reserved.
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
"""Fast Tokenization classes for Splinter."""

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_splinter import SplinterTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


class SplinterTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" Splinter tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
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
        question_token (`str`, *optional*, defaults to `"[QUESTION]"`):
            The token used for constructing question representations.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = SplinterTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        question_token="[QUESTION]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            additional_special_tokens=(question_token,),
            **kwargs,
        )

        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
            or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
        ):
            pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))
            pre_tok_state["lowercase"] = do_lower_case
            pre_tok_state["strip_accents"] = strip_accents
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)

        self.do_lower_case = do_lower_case

    @property
    def question_token_id(self):
        """
        `Optional[int]`: Id of the question token in the vocabulary, used to condition the answer on a question
        representation.
        """
        return self.convert_tokens_to_ids(self.question_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a pair of sequence for question answering tasks by concatenating and adding special
        tokens. A Splinter sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences for question answering: `[CLS] question_tokens [QUESTION] . [SEP] context_tokens [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                The question token IDs if pad_on_right, else context tokens IDs
            token_ids_1 (`List[int]`, *optional*):
                The context token IDs if pad_on_right, else question token IDs

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        question_suffix = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        if self.padding_side == "right":
            # Input is question-then-context
            return cls + token_ids_0 + question_suffix + sep + token_ids_1 + sep
        else:
            # Input is context-then-question
            return cls + token_ids_0 + sep + token_ids_1 + question_suffix + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids)

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        question_suffix = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        if self.padding_side == "right":
            # Input is question-then-context
            return len(cls + token_ids_0 + question_suffix + sep) * [0] + len(token_ids_1 + sep) * [1]
        else:
            # Input is context-then-question
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + question_suffix + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)


__all__ = ["SplinterTokenizerFast"]

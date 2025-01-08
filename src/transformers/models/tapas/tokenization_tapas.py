# coding=utf-8
# Copyright 2020 Google Research and The HuggingFace Inc. team.
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
"""Tokenization class for TAPAS model."""

import collections
import datetime
import enum
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    VERY_LARGE_INTEGER,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
)
from ...utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available, logging


if is_pandas_available():
    import pandas as pd

logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


class TapasTruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`~TapasTokenizer.__call__`]. Useful for tab-completion in an IDE.
    """

    DROP_ROWS_TO_FIT = "drop_rows_to_fit"
    DO_NOT_TRUNCATE = "do_not_truncate"


TableValue = collections.namedtuple("TokenValue", ["token", "column_id", "row_id"])


@dataclass(frozen=True)
class TokenCoordinates:
    column_index: int
    row_index: int
    token_index: int


@dataclass
class TokenizedTable:
    rows: List[List[List[str]]]
    selected_tokens: List[TokenCoordinates]


@dataclass(frozen=True)
class SerializedExample:
    tokens: List[str]
    column_ids: List[int]
    row_ids: List[int]
    segment_ids: List[int]


def _is_inner_wordpiece(token: str):
    return token.startswith("##")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = r"""
            add_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to encode the sequences with the special tokens relative to their model.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, `str` or [`TapasTruncationStrategy`], *optional*, defaults to `False`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'drop_rows_to_fit'`: Truncate to a maximum length specified with the argument `max_length`
                  or to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate row by row, removing rows from the table.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
"""


class TapasTokenizer(PreTrainedTokenizer):
    r"""
    Construct a TAPAS tokenizer. Based on WordPiece. Flattens a table and one or more related sentences to be used by
    TAPAS models.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods. [`TapasTokenizer`] creates several token type ids to
    encode tabular structure. To be more precise, it adds 7 token type ids, in the following order: `segment_ids`,
    `column_ids`, `row_ids`, `prev_labels`, `column_ranks`, `inv_column_ranks` and `numeric_relations`:

    - segment_ids: indicate whether a token belongs to the question (0) or the table (1). 0 for special tokens and
      padding.
    - column_ids: indicate to which column of the table a token belongs (starting from 1). Is 0 for all question
      tokens, special tokens and padding.
    - row_ids: indicate to which row of the table a token belongs (starting from 1). Is 0 for all question tokens,
      special tokens and padding. Tokens of column headers are also 0.
    - prev_labels: indicate whether a token was (part of) an answer to the previous question (1) or not (0). Useful in
      a conversational setup (such as SQA).
    - column_ranks: indicate the rank of a table token relative to a column, if applicable. For example, if you have a
      column "number of movies" with values 87, 53 and 69, then the column ranks of these tokens are 3, 1 and 2
      respectively. 0 for all question tokens, special tokens and padding.
    - inv_column_ranks: indicate the inverse rank of a table token relative to a column, if applicable. For example, if
      you have a column "number of movies" with values 87, 53 and 69, then the inverse column ranks of these tokens are
      1, 3 and 2 respectively. 0 for all question tokens, special tokens and padding.
    - numeric_relations: indicate numeric relations between the question and the tokens of the table. 0 for all
      question tokens, special tokens and padding.

    [`TapasTokenizer`] runs end-to-end tokenization on a table and associated sentences: punctuation splitting and
    wordpiece.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
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
        empty_token (`str`, *optional*, defaults to `"[EMPTY]"`):
            The token used for empty cell values in a table. Empty cell values include "", "n/a", "nan" and "?".
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        cell_trim_length (`int`, *optional*, defaults to -1):
            If > 0: Trim cells so that the length is <= this value. Also disables further cell trimming, should thus be
            used with `truncation` set to `True`.
        max_column_id (`int`, *optional*):
            Max column id to extract.
        max_row_id (`int`, *optional*):
            Max row id to extract.
        strip_column_names (`bool`, *optional*, defaults to `False`):
            Whether to add empty strings instead of column names.
        update_answer_coordinates (`bool`, *optional*, defaults to `False`):
            Whether to recompute the answer coordinates from the answer text.
        min_question_length (`int`, *optional*):
            Minimum length of each question in terms of tokens (will be skipped otherwise).
        max_question_length (`int`, *optional*):
            Maximum length of each question in terms of tokens (will be skipped otherwise).
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        empty_token="[EMPTY]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        cell_trim_length: int = -1,
        max_column_id: int = None,
        max_row_id: int = None,
        strip_column_names: bool = False,
        update_answer_coordinates: bool = False,
        min_question_length=None,
        max_question_length=None,
        model_max_length: int = 512,
        additional_special_tokens: Optional[List[str]] = None,
        clean_up_tokenization_spaces=True,
        **kwargs,
    ):
        if not is_pandas_available():
            raise ImportError("Pandas is required for the TAPAS tokenizer.")

        if additional_special_tokens is not None:
            if empty_token not in additional_special_tokens:
                additional_special_tokens.append(empty_token)
        else:
            additional_special_tokens = [empty_token]

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # Additional properties
        self.cell_trim_length = cell_trim_length
        self.max_column_id = (
            max_column_id
            if max_column_id is not None
            else model_max_length
            if model_max_length is not None
            else VERY_LARGE_INTEGER
        )
        self.max_row_id = (
            max_row_id
            if max_row_id is not None
            else model_max_length
            if model_max_length is not None
            else VERY_LARGE_INTEGER
        )
        self.strip_column_names = strip_column_names
        self.update_answer_coordinates = update_answer_coordinates
        self.min_question_length = min_question_length
        self.max_question_length = max_question_length

        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            empty_token=empty_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            cell_trim_length=cell_trim_length,
            max_column_id=max_column_id,
            max_row_id=max_row_id,
            strip_column_names=strip_column_names,
            update_answer_coordinates=update_answer_coordinates,
            min_question_length=min_question_length,
            max_question_length=max_question_length,
            model_max_length=model_max_length,
            additional_special_tokens=additional_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        if format_text(text) == EMPTY_TEXT:
            return [self.additional_special_tokens[0]]
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def create_attention_mask_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
        """
        Creates the attention mask according to the query token IDs and a list of table values.

        Args:
            query_ids (`List[int]`): list of token IDs corresponding to the ID.
            table_values (`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            `List[int]`: List of ints containing the attention mask values.
        """
        return [1] * (1 + len(query_ids) + 1 + len(table_values))

    def create_segment_token_type_ids_from_sequences(
        self, query_ids: List[int], table_values: List[TableValue]
    ) -> List[int]:
        """
        Creates the segment token type IDs according to the query token IDs and a list of table values.

        Args:
            query_ids (`List[int]`): list of token IDs corresponding to the ID.
            table_values (`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            `List[int]`: List of ints containing the segment token type IDs values.
        """
        table_ids = list(zip(*table_values))[0] if table_values else []
        return [0] * (1 + len(query_ids) + 1) + [1] * len(table_ids)

    def create_column_token_type_ids_from_sequences(
        self, query_ids: List[int], table_values: List[TableValue]
    ) -> List[int]:
        """
        Creates the column token type IDs according to the query token IDs and a list of table values.

        Args:
            query_ids (`List[int]`): list of token IDs corresponding to the ID.
            table_values (`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            `List[int]`: List of ints containing the column token type IDs values.
        """
        table_column_ids = list(zip(*table_values))[1] if table_values else []
        return [0] * (1 + len(query_ids) + 1) + list(table_column_ids)

    def create_row_token_type_ids_from_sequences(
        self, query_ids: List[int], table_values: List[TableValue]
    ) -> List[int]:
        """
        Creates the row token type IDs according to the query token IDs and a list of table values.

        Args:
            query_ids (`List[int]`): list of token IDs corresponding to the ID.
            table_values (`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            `List[int]`: List of ints containing the row token type IDs values.
        """
        table_row_ids = list(zip(*table_values))[2] if table_values else []
        return [0] * (1 + len(query_ids) + 1) + list(table_row_ids)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a question and flattened table for question answering or sequence classification tasks
        by concatenating and adding special tokens.

        Args:
            token_ids_0 (`List[int]`): The ids of the question.
            token_ids_1 (`List[int]`, *optional*): The ids of the flattened table.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        if token_ids_1 is None:
            raise ValueError("With TAPAS, you must provide both question IDs and table IDs.")

        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of question IDs.
            token_ids_1 (`List[int]`, *optional*):
                List of flattened table IDs.
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
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))
        return [1] + ([0] * len(token_ids_0)) + [1]

    @add_end_docstrings(TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        table: "pd.DataFrame",
        queries: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                EncodedInput,
                List[TextInput],
                List[PreTokenizedInput],
                List[EncodedInput],
            ]
        ] = None,
        answer_coordinates: Optional[Union[List[Tuple], List[List[Tuple]]]] = None,
        answer_text: Optional[Union[List[TextInput], List[List[TextInput]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) related to a table.

        Args:
            table (`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use *.astype(str)* on a Pandas
                dataframe to convert it to string.
            queries (`str` or `List[str]`):
                Question or batch of questions related to a table to be encoded. Note that in case of a batch, all
                questions must refer to the **same** table.
            answer_coordinates (`List[Tuple]` or `List[List[Tuple]]`, *optional*):
                Answer coordinates of each table-question pair in the batch. In case only a single table-question pair
                is provided, then the answer_coordinates must be a single list of one or more tuples. Each tuple must
                be a (row_index, column_index) pair. The first data row (not the column header row) has index 0. The
                first column has index 0. In case a batch of table-question pairs is provided, then the
                answer_coordinates must be a list of lists of tuples (each list corresponding to a single
                table-question pair).
            answer_text (`List[str]` or `List[List[str]]`, *optional*):
                Answer text of each table-question pair in the batch. In case only a single table-question pair is
                provided, then the answer_text must be a single list of one or more strings. Each string must be the
                answer text of a corresponding answer coordinate. In case a batch of table-question pairs is provided,
                then the answer_coordinates must be a list of lists of strings (each list corresponding to a single
                table-question pair).
        """
        assert isinstance(table, pd.DataFrame), "Table must be of type pd.DataFrame"

        # Input type checking for clearer error
        valid_query = False

        # Check that query has a valid type
        if queries is None or isinstance(queries, str):
            valid_query = True
        elif isinstance(queries, (list, tuple)):
            if len(queries) == 0 or isinstance(queries[0], str):
                valid_query = True

        if not valid_query:
            raise ValueError(
                "queries input must of type `str` (single example), `List[str]` (batch or single pretokenized"
                " example). "
            )
        is_batched = isinstance(queries, (list, tuple))

        if is_batched:
            return self.batch_encode_plus(
                table=table,
                queries=queries,
                answer_coordinates=answer_coordinates,
                answer_text=answer_text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )
        else:
            return self.encode_plus(
                table=table,
                query=queries,
                answer_coordinates=answer_coordinates,
                answer_text=answer_text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def batch_encode_plus(
        self,
        table: "pd.DataFrame",
        queries: Optional[
            Union[
                List[TextInput],
                List[PreTokenizedInput],
                List[EncodedInput],
            ]
        ] = None,
        answer_coordinates: Optional[List[List[Tuple]]] = None,
        answer_text: Optional[List[List[TextInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepare a table and a list of strings for the model.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            table (`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use *.astype(str)* on a Pandas
                dataframe to convert it to string.
            queries (`List[str]`):
                Batch of questions related to a table to be encoded. Note that all questions must refer to the **same**
                table.
            answer_coordinates (`List[Tuple]` or `List[List[Tuple]]`, *optional*):
                Answer coordinates of each table-question pair in the batch. Each tuple must be a (row_index,
                column_index) pair. The first data row (not the column header row) has index 0. The first column has
                index 0. The answer_coordinates must be a list of lists of tuples (each list corresponding to a single
                table-question pair).
            answer_text (`List[str]` or `List[List[str]]`, *optional*):
                Answer text of each table-question pair in the batch. In case a batch of table-question pairs is
                provided, then the answer_coordinates must be a list of lists of strings (each list corresponding to a
                single table-question pair). Each string must be the answer text of a corresponding answer coordinate.
        """
        if return_token_type_ids is not None and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (answer_coordinates and not answer_text) or (not answer_coordinates and answer_text):
            raise ValueError("In case you provide answers, both answer_coordinates and answer_text should be provided")
        elif answer_coordinates is None and answer_text is None:
            answer_coordinates = answer_text = [None] * len(queries)

        if "is_split_into_words" in kwargs:
            raise NotImplementedError("Currently TapasTokenizer only supports questions as strings.")

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        return self._batch_encode_plus(
            table=table,
            queries=queries,
            answer_coordinates=answer_coordinates,
            answer_text=answer_text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _get_question_tokens(self, query):
        """Tokenizes the query, taking into account the max and min question length."""

        query_tokens = self.tokenize(query)
        if self.max_question_length is not None and len(query_tokens) > self.max_question_length:
            logger.warning("Skipping query as its tokens are longer than the max question length")
            return "", []
        if self.min_question_length is not None and len(query_tokens) < self.min_question_length:
            logger.warning("Skipping query as its tokens are shorter than the min question length")
            return "", []

        return query, query_tokens

    def _batch_encode_plus(
        self,
        table,
        queries: Union[
            List[TextInput],
            List[PreTokenizedInput],
            List[EncodedInput],
        ],
        answer_coordinates: Optional[List[List[Tuple]]] = None,
        answer_text: Optional[List[List[TextInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        table_tokens = self._tokenize_table(table)

        queries_tokens = []
        for idx, query in enumerate(queries):
            query, query_tokens = self._get_question_tokens(query)
            queries[idx] = query
            queries_tokens.append(query_tokens)

        batch_outputs = self._batch_prepare_for_model(
            table,
            queries,
            tokenized_table=table_tokens,
            queries_tokens=queries_tokens,
            answer_coordinates=answer_coordinates,
            padding=padding,
            truncation=truncation,
            answer_text=answer_text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    def _batch_prepare_for_model(
        self,
        raw_table: "pd.DataFrame",
        raw_queries: Union[
            List[TextInput],
            List[PreTokenizedInput],
            List[EncodedInput],
        ],
        tokenized_table: Optional[TokenizedTable] = None,
        queries_tokens: Optional[List[List[str]]] = None,
        answer_coordinates: Optional[List[List[Tuple]]] = None,
        answer_text: Optional[List[List[TextInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        batch_outputs = {}

        for index, example in enumerate(zip(raw_queries, queries_tokens, answer_coordinates, answer_text)):
            raw_query, query_tokens, answer_coords, answer_txt = example
            outputs = self.prepare_for_model(
                raw_table,
                raw_query,
                tokenized_table=tokenized_table,
                query_tokens=query_tokens,
                answer_coordinates=answer_coords,
                answer_text=answer_txt,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterwards
                truncation=truncation,
                max_length=max_length,
                pad_to_multiple_of=None,  # we pad in batch afterwards
                padding_side=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterwards
                return_token_type_ids=return_token_type_ids,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
                prev_answer_coordinates=answer_coordinates[index - 1] if index != 0 else None,
                prev_answer_text=answer_text[index - 1] if index != 0 else None,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING)
    def encode(
        self,
        table: "pd.DataFrame",
        query: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                EncodedInput,
            ]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[int]:
        """
        Prepare a table and a string for the model. This method does not return token type IDs, attention masks, etc.
        which are necessary for the model to work correctly. Use that method if you want to build your processing on
        your own, otherwise refer to `__call__`.

        Args:
            table (`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use *.astype(str)* on a Pandas
                dataframe to convert it to string.
            query (`str` or `List[str]`):
                Question related to a table to be encoded.
        """
        encoded_inputs = self.encode_plus(
            table,
            query=query,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        table: "pd.DataFrame",
        query: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                EncodedInput,
            ]
        ] = None,
        answer_coordinates: Optional[List[Tuple]] = None,
        answer_text: Optional[List[TextInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepare a table and a string for the model.

        Args:
            table (`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use *.astype(str)* on a Pandas
                dataframe to convert it to string.
            query (`str` or `List[str]`):
                Question related to a table to be encoded.
            answer_coordinates (`List[Tuple]` or `List[List[Tuple]]`, *optional*):
                Answer coordinates of each table-question pair in the batch. The answer_coordinates must be a single
                list of one or more tuples. Each tuple must be a (row_index, column_index) pair. The first data row
                (not the column header row) has index 0. The first column has index 0.
            answer_text (`List[str]` or `List[List[str]]`, *optional*):
                Answer text of each table-question pair in the batch. The answer_text must be a single list of one or
                more strings. Each string must be the answer text of a corresponding answer coordinate.
        """
        if return_token_type_ids is not None and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (answer_coordinates and not answer_text) or (not answer_coordinates and answer_text):
            raise ValueError("In case you provide answers, both answer_coordinates and answer_text should be provided")

        if "is_split_into_words" in kwargs:
            raise NotImplementedError("Currently TapasTokenizer only supports questions as strings.")

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        return self._encode_plus(
            table=table,
            query=query,
            answer_coordinates=answer_coordinates,
            answer_text=answer_text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _encode_plus(
        self,
        table: "pd.DataFrame",
        query: Union[
            TextInput,
            PreTokenizedInput,
            EncodedInput,
        ],
        answer_coordinates: Optional[List[Tuple]] = None,
        answer_text: Optional[List[TextInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        if query is None:
            query = ""
            logger.warning(
                "TAPAS is a question answering model but you have not passed a query. Please be aware that the "
                "model will probably not behave correctly."
            )

        table_tokens = self._tokenize_table(table)
        query, query_tokens = self._get_question_tokens(query)

        return self.prepare_for_model(
            table,
            query,
            tokenized_table=table_tokens,
            query_tokens=query_tokens,
            answer_coordinates=answer_coordinates,
            answer_text=answer_text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        raw_table: "pd.DataFrame",
        raw_query: Union[
            TextInput,
            PreTokenizedInput,
            EncodedInput,
        ],
        tokenized_table: Optional[TokenizedTable] = None,
        query_tokens: Optional[TokenizedTable] = None,
        answer_coordinates: Optional[List[Tuple]] = None,
        answer_text: Optional[List[TextInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id so that it can be used by the model. It adds special tokens, truncates
        sequences if overflowing while taking into account the special tokens.

        Args:
            raw_table (`pd.DataFrame`):
                The original table before any transformation (like tokenization) was applied to it.
            raw_query (`TextInput` or `PreTokenizedInput` or `EncodedInput`):
                The original query before any transformation (like tokenization) was applied to it.
            tokenized_table (`TokenizedTable`):
                The table after tokenization.
            query_tokens (`List[str]`):
                The query after tokenization.
            answer_coordinates (`List[Tuple]` or `List[List[Tuple]]`, *optional*):
                Answer coordinates of each table-question pair in the batch. The answer_coordinates must be a single
                list of one or more tuples. Each tuple must be a (row_index, column_index) pair. The first data row
                (not the column header row) has index 0. The first column has index 0.
            answer_text (`List[str]` or `List[List[str]]`, *optional*):
                Answer text of each table-question pair in the batch. The answer_text must be a single list of one or
                more strings. Each string must be the answer text of a corresponding answer coordinate.
        """
        if isinstance(padding, bool):
            if padding and (max_length is not None or pad_to_multiple_of is not None):
                padding = PaddingStrategy.MAX_LENGTH
            else:
                padding = PaddingStrategy.DO_NOT_PAD
        elif not isinstance(padding, PaddingStrategy):
            padding = PaddingStrategy(padding)

        if isinstance(truncation, bool):
            if truncation:
                truncation = TapasTruncationStrategy.DROP_ROWS_TO_FIT
            else:
                truncation = TapasTruncationStrategy.DO_NOT_TRUNCATE
        elif not isinstance(truncation, TapasTruncationStrategy):
            truncation = TapasTruncationStrategy(truncation)

        encoded_inputs = {}

        is_part_of_batch = False
        prev_answer_coordinates, prev_answer_text = None, None
        if "prev_answer_coordinates" in kwargs and "prev_answer_text" in kwargs:
            is_part_of_batch = True
            prev_answer_coordinates = kwargs["prev_answer_coordinates"]
            prev_answer_text = kwargs["prev_answer_text"]

        num_rows = self._get_num_rows(raw_table, truncation != TapasTruncationStrategy.DO_NOT_TRUNCATE)
        num_columns = self._get_num_columns(raw_table)
        _, _, num_tokens = self._get_table_boundaries(tokenized_table)

        if truncation != TapasTruncationStrategy.DO_NOT_TRUNCATE:
            num_rows, num_tokens = self._get_truncated_table_rows(
                query_tokens, tokenized_table, num_rows, num_columns, max_length, truncation_strategy=truncation
            )
        table_data = list(self._get_table_values(tokenized_table, num_columns, num_rows, num_tokens))

        query_ids = self.convert_tokens_to_ids(query_tokens)
        table_ids = list(zip(*table_data))[0] if len(table_data) > 0 else list(zip(*table_data))
        table_ids = self.convert_tokens_to_ids(list(table_ids))

        if "return_overflowing_tokens" in kwargs and kwargs["return_overflowing_tokens"]:
            raise ValueError("TAPAS does not return overflowing tokens as it works on tables.")

        if add_special_tokens:
            input_ids = self.build_inputs_with_special_tokens(query_ids, table_ids)
        else:
            input_ids = query_ids + table_ids

        if max_length is not None and len(input_ids) > max_length:
            raise ValueError(
                "Could not encode the query and table header given the maximum length. Encoding the query and table "
                f"header results in a length of {len(input_ids)} which is higher than the max_length of {max_length}"
            )

        encoded_inputs["input_ids"] = input_ids

        segment_ids = self.create_segment_token_type_ids_from_sequences(query_ids, table_data)
        column_ids = self.create_column_token_type_ids_from_sequences(query_ids, table_data)
        row_ids = self.create_row_token_type_ids_from_sequences(query_ids, table_data)
        if not is_part_of_batch or (prev_answer_coordinates is None and prev_answer_text is None):
            # simply set the prev_labels to zeros
            prev_labels = [0] * len(row_ids)
        else:
            prev_labels = self.get_answer_ids(
                column_ids, row_ids, table_data, prev_answer_text, prev_answer_coordinates
            )

        # FIRST: parse both the table and question in terms of numeric values

        raw_table = add_numeric_table_values(raw_table)
        raw_query = add_numeric_values_to_question(raw_query)

        # SECOND: add numeric-related features (and not parse them in these functions):

        column_ranks, inv_column_ranks = self._get_numeric_column_ranks(column_ids, row_ids, raw_table)
        numeric_relations = self._get_numeric_relations(raw_query, column_ids, row_ids, raw_table)

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if return_attention_mask:
            attention_mask = self.create_attention_mask_from_sequences(query_ids, table_data)
            encoded_inputs["attention_mask"] = attention_mask

        if answer_coordinates is not None and answer_text is not None:
            labels = self.get_answer_ids(column_ids, row_ids, table_data, answer_text, answer_coordinates)
            numeric_values = self._get_numeric_values(raw_table, column_ids, row_ids)
            numeric_values_scale = self._get_numeric_values_scale(raw_table, column_ids, row_ids)

            encoded_inputs["labels"] = labels
            encoded_inputs["numeric_values"] = numeric_values
            encoded_inputs["numeric_values_scale"] = numeric_values_scale

        if return_token_type_ids:
            token_type_ids = [
                segment_ids,
                column_ids,
                row_ids,
                prev_labels,
                column_ranks,
                inv_column_ranks,
                numeric_relations,
            ]

            token_type_ids = [list(ids) for ids in list(zip(*token_type_ids))]
            encoded_inputs["token_type_ids"] = token_type_ids

        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(query_ids, table_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(input_ids)

        # Check lengths
        if max_length is None and len(encoded_inputs["input_ids"]) > self.model_max_length and verbose:
            if not self.deprecation_warnings.get("sequence-length-is-longer-than-the-specified-maximum", False):
                logger.warning(
                    "Token indices sequence length is longer than the specified maximum sequence length "
                    f"for this model ({len(encoded_inputs['input_ids'])} > {self.model_max_length}). Running this "
                    "sequence through the model will result in indexing errors."
                )
            self.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

        # Padding
        if padding != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding.value,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def _get_truncated_table_rows(
        self,
        query_tokens: List[str],
        tokenized_table: TokenizedTable,
        num_rows: int,
        num_columns: int,
        max_length: int,
        truncation_strategy: Union[str, TapasTruncationStrategy],
    ) -> Tuple[int, int]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            query_tokens (`List[str]`):
                List of strings corresponding to the tokenized query.
            tokenized_table (`TokenizedTable`):
                Tokenized table
            num_rows (`int`):
                Total number of table rows
            num_columns (`int`):
                Total number of table columns
            max_length (`int`):
                Total maximum length.
            truncation_strategy (`str` or [`TapasTruncationStrategy]`):
                Truncation strategy to use. Seeing as this method should only be called when truncating, the only
                available strategy is the `"drop_rows_to_fit"` strategy.

        Returns:
            `Tuple(int, int)`: tuple containing the number of rows after truncation, and the number of tokens available
            for each table element.
        """
        if not isinstance(truncation_strategy, TapasTruncationStrategy):
            truncation_strategy = TapasTruncationStrategy(truncation_strategy)

        if max_length is None:
            max_length = self.model_max_length

        if truncation_strategy == TapasTruncationStrategy.DROP_ROWS_TO_FIT:
            while True:
                num_tokens = self._get_max_num_tokens(
                    query_tokens, tokenized_table, num_rows=num_rows, num_columns=num_columns, max_length=max_length
                )

                if num_tokens is not None:
                    # We could fit the table.
                    break

                # Try to drop a row to fit the table.
                num_rows -= 1

                if num_rows < 1:
                    break
        elif truncation_strategy != TapasTruncationStrategy.DO_NOT_TRUNCATE:
            raise ValueError(f"Unknown truncation strategy {truncation_strategy}.")

        return num_rows, num_tokens or 1

    def _tokenize_table(
        self,
        table=None,
    ):
        """
        Tokenizes column headers and cell texts of a table.

        Args:
            table (`pd.Dataframe`):
                Table. Returns: `TokenizedTable`: TokenizedTable object.
        """
        tokenized_rows = []
        tokenized_row = []
        # tokenize column headers
        for column in table:
            if self.strip_column_names:
                tokenized_row.append(self.tokenize(""))
            else:
                tokenized_row.append(self.tokenize(column))
        tokenized_rows.append(tokenized_row)

        # tokenize cell values
        for idx, row in table.iterrows():
            tokenized_row = []
            for cell in row:
                tokenized_row.append(self.tokenize(cell))
            tokenized_rows.append(tokenized_row)

        token_coordinates = []
        for row_index, row in enumerate(tokenized_rows):
            for column_index, cell in enumerate(row):
                for token_index, _ in enumerate(cell):
                    token_coordinates.append(
                        TokenCoordinates(
                            row_index=row_index,
                            column_index=column_index,
                            token_index=token_index,
                        )
                    )

        return TokenizedTable(
            rows=tokenized_rows,
            selected_tokens=token_coordinates,
        )

    def _question_encoding_cost(self, question_tokens):
        # Two extra spots of SEP and CLS.
        return len(question_tokens) + 2

    def _get_token_budget(self, question_tokens, max_length=None):
        """
        Computes the number of tokens left for the table after tokenizing a question, taking into account the max
        sequence length of the model.

        Args:
            question_tokens (`List[String]`):
                List of question tokens. Returns: `int`: the number of tokens left for the table, given the model max
                length.
        """
        return (max_length if max_length is not None else self.model_max_length) - self._question_encoding_cost(
            question_tokens
        )

    def _get_table_values(self, table, num_columns, num_rows, num_tokens) -> Generator[TableValue, None, None]:
        """Iterates over partial table and returns token, column and row indexes."""
        for tc in table.selected_tokens:
            # First row is header row.
            if tc.row_index >= num_rows + 1:
                continue
            if tc.column_index >= num_columns:
                continue
            cell = table.rows[tc.row_index][tc.column_index]
            token = cell[tc.token_index]
            word_begin_index = tc.token_index
            # Don't add partial words. Find the starting word piece and check if it
            # fits in the token budget.
            while word_begin_index >= 0 and _is_inner_wordpiece(cell[word_begin_index]):
                word_begin_index -= 1
            if word_begin_index >= num_tokens:
                continue
            yield TableValue(token, tc.column_index + 1, tc.row_index)

    def _get_table_boundaries(self, table):
        """Return maximal number of rows, columns and tokens."""
        max_num_tokens = 0
        max_num_columns = 0
        max_num_rows = 0
        for tc in table.selected_tokens:
            max_num_columns = max(max_num_columns, tc.column_index + 1)
            max_num_rows = max(max_num_rows, tc.row_index + 1)
            max_num_tokens = max(max_num_tokens, tc.token_index + 1)
            max_num_columns = min(self.max_column_id, max_num_columns)
            max_num_rows = min(self.max_row_id, max_num_rows)
        return max_num_rows, max_num_columns, max_num_tokens

    def _get_table_cost(self, table, num_columns, num_rows, num_tokens):
        return sum(1 for _ in self._get_table_values(table, num_columns, num_rows, num_tokens))

    def _get_max_num_tokens(self, question_tokens, tokenized_table, num_columns, num_rows, max_length):
        """Computes max number of tokens that can be squeezed into the budget."""
        token_budget = self._get_token_budget(question_tokens, max_length)
        _, _, max_num_tokens = self._get_table_boundaries(tokenized_table)
        if self.cell_trim_length >= 0 and max_num_tokens > self.cell_trim_length:
            max_num_tokens = self.cell_trim_length
        num_tokens = 0
        for num_tokens in range(max_num_tokens + 1):
            cost = self._get_table_cost(tokenized_table, num_columns, num_rows, num_tokens + 1)
            if cost > token_budget:
                break
        if num_tokens < max_num_tokens:
            if self.cell_trim_length >= 0:
                # We don't allow dynamic trimming if a cell_trim_length is set.
                return None
            if num_tokens == 0:
                return None
        return num_tokens

    def _get_num_columns(self, table):
        num_columns = table.shape[1]
        if num_columns >= self.max_column_id:
            raise ValueError("Too many columns")
        return num_columns

    def _get_num_rows(self, table, drop_rows_to_fit):
        num_rows = table.shape[0]
        if num_rows >= self.max_row_id:
            if drop_rows_to_fit:
                num_rows = self.max_row_id - 1
            else:
                raise ValueError("Too many rows")
        return num_rows

    def _serialize_text(self, question_tokens):
        """Serializes texts in index arrays."""
        tokens = []
        segment_ids = []
        column_ids = []
        row_ids = []

        # add [CLS] token at the beginning
        tokens.append(self.cls_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        for token in question_tokens:
            tokens.append(token)
            segment_ids.append(0)
            column_ids.append(0)
            row_ids.append(0)

        return tokens, segment_ids, column_ids, row_ids

    def _serialize(
        self,
        question_tokens,
        table,
        num_columns,
        num_rows,
        num_tokens,
    ):
        """Serializes table and text."""
        tokens, segment_ids, column_ids, row_ids = self._serialize_text(question_tokens)

        # add [SEP] token between question and table tokens
        tokens.append(self.sep_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        for token, column_id, row_id in self._get_table_values(table, num_columns, num_rows, num_tokens):
            tokens.append(token)
            segment_ids.append(1)
            column_ids.append(column_id)
            row_ids.append(row_id)

        return SerializedExample(
            tokens=tokens,
            segment_ids=segment_ids,
            column_ids=column_ids,
            row_ids=row_ids,
        )

    def _get_column_values(self, table, col_index):
        table_numeric_values = {}
        for row_index, row in table.iterrows():
            cell = row[col_index]
            if cell.numeric_value is not None:
                table_numeric_values[row_index] = cell.numeric_value
        return table_numeric_values

    def _get_cell_token_indexes(self, column_ids, row_ids, column_id, row_id):
        for index in range(len(column_ids)):
            if column_ids[index] - 1 == column_id and row_ids[index] - 1 == row_id:
                yield index

    def _get_numeric_column_ranks(self, column_ids, row_ids, table):
        """Returns column ranks for all numeric columns."""

        ranks = [0] * len(column_ids)
        inv_ranks = [0] * len(column_ids)

        # original code from tf_example_utils.py of the original implementation
        if table is not None:
            for col_index in range(len(table.columns)):
                table_numeric_values = self._get_column_values(table, col_index)

                if not table_numeric_values:
                    continue

                try:
                    key_fn = get_numeric_sort_key_fn(table_numeric_values.values())
                except ValueError:
                    continue

                table_numeric_values = {row_index: key_fn(value) for row_index, value in table_numeric_values.items()}

                table_numeric_values_inv = collections.defaultdict(list)
                for row_index, value in table_numeric_values.items():
                    table_numeric_values_inv[value].append(row_index)

                unique_values = sorted(table_numeric_values_inv.keys())

                for rank, value in enumerate(unique_values):
                    for row_index in table_numeric_values_inv[value]:
                        for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                            ranks[index] = rank + 1
                            inv_ranks[index] = len(unique_values) - rank

        return ranks, inv_ranks

    def _get_numeric_sort_key_fn(self, table_numeric_values, value):
        """
        Returns the sort key function for comparing value to table values. The function returned will be a suitable
        input for the key param of the sort(). See number_annotation_utils._get_numeric_sort_key_fn for details

        Args:
            table_numeric_values: Numeric values of a column
            value: Numeric value in the question

        Returns:
            A function key function to compare column and question values.
        """
        if not table_numeric_values:
            return None
        all_values = list(table_numeric_values.values())
        all_values.append(value)
        try:
            return get_numeric_sort_key_fn(all_values)
        except ValueError:
            return None

    def _get_numeric_relations(self, question, column_ids, row_ids, table):
        """
        Returns numeric relations embeddings

        Args:
            question: Question object.
            column_ids: Maps word piece position to column id.
            row_ids: Maps word piece position to row id.
            table: The table containing the numeric cell values.
        """

        numeric_relations = [0] * len(column_ids)

        # first, we add any numeric value spans to the question:
        # Create a dictionary that maps a table cell to the set of all relations
        # this cell has with any value in the question.
        cell_indices_to_relations = collections.defaultdict(set)
        if question is not None and table is not None:
            for numeric_value_span in question.numeric_spans:
                for value in numeric_value_span.values:
                    for column_index in range(len(table.columns)):
                        table_numeric_values = self._get_column_values(table, column_index)
                        sort_key_fn = self._get_numeric_sort_key_fn(table_numeric_values, value)
                        if sort_key_fn is None:
                            continue
                        for row_index, cell_value in table_numeric_values.items():
                            relation = get_numeric_relation(value, cell_value, sort_key_fn)
                            if relation is not None:
                                cell_indices_to_relations[column_index, row_index].add(relation)

        # For each cell add a special feature for all its word pieces.
        for (column_index, row_index), relations in cell_indices_to_relations.items():
            relation_set_index = 0
            for relation in relations:
                assert relation.value >= Relation.EQ.value
                relation_set_index += 2 ** (relation.value - Relation.EQ.value)
            for cell_token_index in self._get_cell_token_indexes(column_ids, row_ids, column_index, row_index):
                numeric_relations[cell_token_index] = relation_set_index

        return numeric_relations

    def _get_numeric_values(self, table, column_ids, row_ids):
        """Returns numeric values for computation of answer loss."""

        numeric_values = [float("nan")] * len(column_ids)

        if table is not None:
            num_rows = table.shape[0]
            num_columns = table.shape[1]

            for col_index in range(num_columns):
                for row_index in range(num_rows):
                    numeric_value = table.iloc[row_index, col_index].numeric_value
                    if numeric_value is not None:
                        if numeric_value.float_value is None:
                            continue
                        float_value = numeric_value.float_value
                        if float_value == float("inf"):
                            continue
                        for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                            numeric_values[index] = float_value

        return numeric_values

    def _get_numeric_values_scale(self, table, column_ids, row_ids):
        """Returns a scale to each token to down weigh the value of long words."""

        numeric_values_scale = [1.0] * len(column_ids)

        if table is None:
            return numeric_values_scale

        num_rows = table.shape[0]
        num_columns = table.shape[1]

        for col_index in range(num_columns):
            for row_index in range(num_rows):
                indices = list(self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index))
                num_indices = len(indices)
                if num_indices > 1:
                    for index in indices:
                        numeric_values_scale[index] = float(num_indices)

        return numeric_values_scale

    def _pad_to_seq_length(self, inputs):
        while len(inputs) > self.model_max_length:
            inputs.pop()
        while len(inputs) < self.model_max_length:
            inputs.append(0)

    def _get_all_answer_ids_from_coordinates(
        self,
        column_ids,
        row_ids,
        answers_list,
    ):
        """Maps lists of answer coordinates to token indexes."""
        answer_ids = [0] * len(column_ids)
        found_answers = set()
        all_answers = set()
        for answers in answers_list:
            column_index, row_index = answers
            all_answers.add((column_index, row_index))
            for index in self._get_cell_token_indexes(column_ids, row_ids, column_index, row_index):
                found_answers.add((column_index, row_index))
                answer_ids[index] = 1

        missing_count = len(all_answers) - len(found_answers)
        return answer_ids, missing_count

    def _get_all_answer_ids(self, column_ids, row_ids, answer_coordinates):
        """
        Maps answer coordinates of a question to token indexes.

        In the SQA format (TSV), the coordinates are given as (row, column) tuples. Here, we first swap them to
        (column, row) format before calling _get_all_answer_ids_from_coordinates.
        """

        def _to_coordinates(answer_coordinates_question):
            return [(coords[1], coords[0]) for coords in answer_coordinates_question]

        return self._get_all_answer_ids_from_coordinates(
            column_ids, row_ids, answers_list=(_to_coordinates(answer_coordinates))
        )

    def _find_tokens(self, text, segment):
        """Return start index of segment in text or None."""
        logging.info(f"text: {text} {segment}")
        for index in range(1 + len(text) - len(segment)):
            for seg_index, seg_token in enumerate(segment):
                if text[index + seg_index].piece != seg_token.piece:
                    break
            else:
                return index
        return None

    def _find_answer_coordinates_from_answer_text(
        self,
        tokenized_table,
        answer_text,
    ):
        """Returns all occurrences of answer_text in the table."""
        logging.info(f"answer text: {answer_text}")
        for row_index, row in enumerate(tokenized_table.rows):
            if row_index == 0:
                # We don't search for answers in the header.
                continue
            for col_index, cell in enumerate(row):
                token_index = self._find_tokens(cell, answer_text)
                if token_index is not None:
                    yield TokenCoordinates(
                        row_index=row_index,
                        column_index=col_index,
                        token_index=token_index,
                    )

    def _find_answer_ids_from_answer_texts(
        self,
        column_ids,
        row_ids,
        tokenized_table,
        answer_texts,
    ):
        """Maps question with answer texts to the first matching token indexes."""
        answer_ids = [0] * len(column_ids)
        for answer_text in answer_texts:
            for coordinates in self._find_answer_coordinates_from_answer_text(
                tokenized_table,
                answer_text,
            ):
                # Maps answer coordinates to indexes this can fail if tokens / rows have
                # been pruned.
                indexes = list(
                    self._get_cell_token_indexes(
                        column_ids,
                        row_ids,
                        column_id=coordinates.column_index,
                        row_id=coordinates.row_index - 1,
                    )
                )
                indexes.sort()
                coordinate_answer_ids = []
                if indexes:
                    begin_index = coordinates.token_index + indexes[0]
                    end_index = begin_index + len(answer_text)
                    for index in indexes:
                        if index >= begin_index and index < end_index:
                            coordinate_answer_ids.append(index)
                if len(coordinate_answer_ids) == len(answer_text):
                    for index in coordinate_answer_ids:
                        answer_ids[index] = 1
                    break
        return answer_ids

    def _get_answer_ids(self, column_ids, row_ids, answer_coordinates):
        """Maps answer coordinates of a question to token indexes."""
        answer_ids, missing_count = self._get_all_answer_ids(column_ids, row_ids, answer_coordinates)

        if missing_count:
            raise ValueError("Couldn't find all answers")
        return answer_ids

    def get_answer_ids(self, column_ids, row_ids, tokenized_table, answer_texts_question, answer_coordinates_question):
        if self.update_answer_coordinates:
            return self._find_answer_ids_from_answer_texts(
                column_ids,
                row_ids,
                tokenized_table,
                answer_texts=[self.tokenize(at) for at in answer_texts_question],
            )
        return self._get_answer_ids(column_ids, row_ids, answer_coordinates_question)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            padding_side:
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(encoded_inputs["input_ids"])

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD and len(encoded_inputs["input_ids"]) != max_length
        )

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        if needs_to_be_padded:
            difference = max_length - len(encoded_inputs["input_ids"])
            padding_side = padding_side if padding_side is not None else self.padding_side
            if padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [[self.pad_token_type_id] * 7] * difference
                    )
                if "labels" in encoded_inputs:
                    encoded_inputs["labels"] = encoded_inputs["labels"] + [0] * difference
                if "numeric_values" in encoded_inputs:
                    encoded_inputs["numeric_values"] = encoded_inputs["numeric_values"] + [float("nan")] * difference
                if "numeric_values_scale" in encoded_inputs:
                    encoded_inputs["numeric_values_scale"] = (
                        encoded_inputs["numeric_values_scale"] + [1.0] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
            elif padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [[self.pad_token_type_id] * 7] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "labels" in encoded_inputs:
                    encoded_inputs["labels"] = [0] * difference + encoded_inputs["labels"]
                if "numeric_values" in encoded_inputs:
                    encoded_inputs["numeric_values"] = [float("nan")] * difference + encoded_inputs["numeric_values"]
                if "numeric_values_scale" in encoded_inputs:
                    encoded_inputs["numeric_values_scale"] = [1.0] * difference + encoded_inputs[
                        "numeric_values_scale"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + encoded_inputs["input_ids"]
            else:
                raise ValueError("Invalid padding strategy:" + str(padding_side))

        return encoded_inputs

    # Everything related to converting logits to predictions

    def _get_cell_token_probs(self, probabilities, segment_ids, row_ids, column_ids):
        for i, p in enumerate(probabilities):
            segment_id = segment_ids[i]
            col = column_ids[i] - 1
            row = row_ids[i] - 1
            if col >= 0 and row >= 0 and segment_id == 1:
                yield i, p

    def _get_mean_cell_probs(self, probabilities, segment_ids, row_ids, column_ids):
        """Computes average probability per cell, aggregating over tokens."""
        coords_to_probs = collections.defaultdict(list)
        for i, prob in self._get_cell_token_probs(probabilities, segment_ids, row_ids, column_ids):
            col = column_ids[i] - 1
            row = row_ids[i] - 1
            coords_to_probs[(col, row)].append(prob)
        return {coords: np.array(cell_probs).mean() for coords, cell_probs in coords_to_probs.items()}

    def convert_logits_to_predictions(self, data, logits, logits_agg=None, cell_classification_threshold=0.5):
        """
        Converts logits of [`TapasForQuestionAnswering`] to actual predicted answer coordinates and optional
        aggregation indices.

        The original implementation, on which this function is based, can be found
        [here](https://github.com/google-research/tapas/blob/4908213eb4df7aa988573350278b44c4dbe3f71b/tapas/experiments/prediction_utils.py#L288).

        Args:
            data (`dict`):
                Dictionary mapping features to actual values. Should be created using [`TapasTokenizer`].
            logits (`torch.Tensor` or `tf.Tensor` of shape `(batch_size, sequence_length)`):
                Tensor containing the logits at the token level.
            logits_agg (`torch.Tensor` or `tf.Tensor` of shape `(batch_size, num_aggregation_labels)`, *optional*):
                Tensor containing the aggregation logits.
            cell_classification_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to be used for cell selection. All table cells for which their probability is larger than
                this threshold will be selected.

        Returns:
            `tuple` comprising various elements depending on the inputs:

            - predicted_answer_coordinates (`List[List[[tuple]]` of length `batch_size`): Predicted answer coordinates
              as a list of lists of tuples. Each element in the list contains the predicted answer coordinates of a
              single example in the batch, as a list of tuples. Each tuple is a cell, i.e. (row index, column index).
            - predicted_aggregation_indices (`List[int]`of length `batch_size`, *optional*, returned when
              `logits_aggregation` is provided): Predicted aggregation operator indices of the aggregation head.
        """
        # converting to numpy arrays to work with PT/TF
        logits = logits.numpy()
        if logits_agg is not None:
            logits_agg = logits_agg.numpy()
        data = {key: value.numpy() for key, value in data.items() if key != "training"}
        # input data is of type float32
        # np.log(np.finfo(np.float32).max) = 88.72284
        # Any value over 88.72284 will overflow when passed through the exponential, sending a warning
        # We disable this warning by truncating the logits.
        logits[logits < -88.7] = -88.7

        # Compute probabilities from token logits
        probabilities = 1 / (1 + np.exp(-logits)) * data["attention_mask"]
        token_types = [
            "segment_ids",
            "column_ids",
            "row_ids",
            "prev_labels",
            "column_ranks",
            "inv_column_ranks",
            "numeric_relations",
        ]

        # collect input_ids, segment ids, row ids and column ids of batch. Shape (batch_size, seq_len)
        input_ids = data["input_ids"]
        segment_ids = data["token_type_ids"][:, :, token_types.index("segment_ids")]
        row_ids = data["token_type_ids"][:, :, token_types.index("row_ids")]
        column_ids = data["token_type_ids"][:, :, token_types.index("column_ids")]

        # next, get answer coordinates for every example in the batch
        num_batch = input_ids.shape[0]
        predicted_answer_coordinates = []
        for i in range(num_batch):
            probabilities_example = probabilities[i].tolist()
            segment_ids_example = segment_ids[i]
            row_ids_example = row_ids[i]
            column_ids_example = column_ids[i]

            max_width = column_ids_example.max()
            max_height = row_ids_example.max()

            if max_width == 0 and max_height == 0:
                continue

            cell_coords_to_prob = self._get_mean_cell_probs(
                probabilities_example,
                segment_ids_example.tolist(),
                row_ids_example.tolist(),
                column_ids_example.tolist(),
            )

            # Select the answers above the classification threshold.
            answer_coordinates = []
            for col in range(max_width):
                for row in range(max_height):
                    cell_prob = cell_coords_to_prob.get((col, row), None)
                    if cell_prob is not None:
                        if cell_prob > cell_classification_threshold:
                            answer_coordinates.append((row, col))
            answer_coordinates = sorted(answer_coordinates)
            predicted_answer_coordinates.append(answer_coordinates)

        output = (predicted_answer_coordinates,)

        if logits_agg is not None:
            predicted_aggregation_indices = logits_agg.argmax(axis=-1)
            output = (predicted_answer_coordinates, predicted_aggregation_indices.tolist())

        return output

    # End of everything related to converting logits to predictions


# Copied from transformers.models.bert.tokenization_bert.BasicTokenizer
class BasicTokenizer:
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # prevents treating the same character with different unicode codepoints as different characters
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
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

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer:
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


# Below: utilities for TAPAS tokenizer (independent from PyTorch/Tensorflow).
# This includes functions to parse numeric values (dates and numbers) from both the table and questions in order
# to create the column_ranks, inv_column_ranks, numeric_values, numeric values_scale and numeric_relations in
# prepare_for_model of TapasTokenizer.
# These are meant to be used in an academic setup, for production use cases Gold mine or Aqua should be used.


# taken from constants.py of the original implementation
# URL: https://github.com/google-research/tapas/blob/master/tapas/utils/constants.py
class Relation(enum.Enum):
    HEADER_TO_CELL = 1  # Connects header to cell.
    CELL_TO_HEADER = 2  # Connects cell to header.
    QUERY_TO_HEADER = 3  # Connects query to headers.
    QUERY_TO_CELL = 4  # Connects query to cells.
    ROW_TO_CELL = 5  # Connects row to cells.
    CELL_TO_ROW = 6  # Connects cells to row.
    EQ = 7  # Annotation value is same as cell value
    LT = 8  # Annotation value is less than cell value
    GT = 9  # Annotation value is greater than cell value


@dataclass
class Date:
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None


@dataclass
class NumericValue:
    float_value: Optional[float] = None
    date: Optional[Date] = None


@dataclass
class NumericValueSpan:
    begin_index: int = None
    end_index: int = None
    values: List[NumericValue] = None


@dataclass
class Cell:
    text: str
    numeric_value: Optional[NumericValue] = None


@dataclass
class Question:
    original_text: str  # The original raw question string.
    text: str  # The question string after normalization.
    numeric_spans: Optional[List[NumericValueSpan]] = None


# Below: all functions from number_utils.py as well as 2 functions (namely get_all_spans and normalize_for_match)
# from text_utils.py of the original implementation. URL's:
# - https://github.com/google-research/tapas/blob/master/tapas/utils/number_utils.py
# - https://github.com/google-research/tapas/blob/master/tapas/utils/text_utils.py


# Constants for parsing date expressions.
# Masks that specify (by a bool) which of (year, month, day) will be populated.
_DateMask = collections.namedtuple("_DateMask", ["year", "month", "day"])

_YEAR = _DateMask(True, False, False)
_YEAR_MONTH = _DateMask(True, True, False)
_YEAR_MONTH_DAY = _DateMask(True, True, True)
_MONTH = _DateMask(False, True, False)
_MONTH_DAY = _DateMask(False, True, True)

# Pairs of patterns to pass to 'datetime.strptime' and masks specifying which
# fields will be set by the corresponding pattern.
_DATE_PATTERNS = (
    ("%B", _MONTH),
    ("%Y", _YEAR),
    ("%Ys", _YEAR),
    ("%b %Y", _YEAR_MONTH),
    ("%B %Y", _YEAR_MONTH),
    ("%B %d", _MONTH_DAY),
    ("%b %d", _MONTH_DAY),
    ("%d %b", _MONTH_DAY),
    ("%d %B", _MONTH_DAY),
    ("%B %d, %Y", _YEAR_MONTH_DAY),
    ("%d %B %Y", _YEAR_MONTH_DAY),
    ("%m-%d-%Y", _YEAR_MONTH_DAY),
    ("%Y-%m-%d", _YEAR_MONTH_DAY),
    ("%Y-%m", _YEAR_MONTH),
    ("%B %Y", _YEAR_MONTH),
    ("%d %b %Y", _YEAR_MONTH_DAY),
    ("%Y-%m-%d", _YEAR_MONTH_DAY),
    ("%b %d, %Y", _YEAR_MONTH_DAY),
    ("%d.%m.%Y", _YEAR_MONTH_DAY),
    ("%A, %b %d", _MONTH_DAY),
    ("%A, %B %d", _MONTH_DAY),
)

# This mapping is used to convert date patterns to regex patterns.
_FIELD_TO_REGEX = (
    ("%A", r"\w+"),  # Weekday as locale’s full name.
    ("%B", r"\w+"),  # Month as locale’s full name.
    ("%Y", r"\d{4}"),  # Year with century as a decimal number.
    ("%b", r"\w{3}"),  # Month as locale’s abbreviated name.
    ("%d", r"\d{1,2}"),  # Day of the month as a zero-padded decimal number.
    ("%m", r"\d{1,2}"),  # Month as a zero-padded decimal number.
)


def _process_date_pattern(dp):
    """Compute a regex for each date pattern to use as a prefilter."""
    pattern, mask = dp
    regex = pattern
    regex = regex.replace(".", re.escape("."))
    regex = regex.replace("-", re.escape("-"))
    regex = regex.replace(" ", r"\s+")
    for field, field_regex in _FIELD_TO_REGEX:
        regex = regex.replace(field, field_regex)
    # Make sure we didn't miss any of the fields.
    assert "%" not in regex, regex
    return pattern, mask, re.compile("^" + regex + "$")


def _process_date_patterns():
    return tuple(_process_date_pattern(dp) for dp in _DATE_PATTERNS)


_PROCESSED_DATE_PATTERNS = _process_date_patterns()

_MAX_DATE_NGRAM_SIZE = 5

# Following DynSp:
# https://github.com/Microsoft/DynSP/blob/master/util.py#L414.
_NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
]

_ORDINAL_WORDS = [
    "zeroth",
    "first",
    "second",
    "third",
    "fourth",
    "fith",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
]

_ORDINAL_SUFFIXES = ["st", "nd", "rd", "th"]

_NUMBER_PATTERN = re.compile(r"((^|\s)[+-])?((\.\d+)|(\d+(,\d\d\d)*(\.\d*)?))")

# Following DynSp:
# https://github.com/Microsoft/DynSP/blob/master/util.py#L293.
_MIN_YEAR = 1700
_MAX_YEAR = 2016

_INF = float("INF")


def _get_numeric_value_from_date(date, mask):
    """Converts date (datetime Python object) to a NumericValue object with a Date object value."""
    if date.year < _MIN_YEAR or date.year > _MAX_YEAR:
        raise ValueError(f"Invalid year: {date.year}")

    new_date = Date()
    if mask.year:
        new_date.year = date.year
    if mask.month:
        new_date.month = date.month
    if mask.day:
        new_date.day = date.day
    return NumericValue(date=new_date)


def _get_span_length_key(span):
    """Sorts span by decreasing length first and increasing first index second."""
    return span[1] - span[0], -span[0]


def _get_numeric_value_from_float(value):
    """Converts float (Python) to a NumericValue object with a float value."""
    return NumericValue(float_value=value)


# Doesn't parse ordinal expressions such as '18th of february 1655'.
def _parse_date(text):
    """Attempts to format a text as a standard date string (yyyy-mm-dd)."""
    text = re.sub(r"Sept\b", "Sep", text)
    for in_pattern, mask, regex in _PROCESSED_DATE_PATTERNS:
        if not regex.match(text):
            continue
        try:
            date = datetime.datetime.strptime(text, in_pattern).date()
        except ValueError:
            continue
        try:
            return _get_numeric_value_from_date(date, mask)
        except ValueError:
            continue
    return None


def _parse_number(text):
    """Parses simple cardinal and ordinals numbers."""
    for suffix in _ORDINAL_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = text.replace(",", "")
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    if value == _INF:
        return None
    return value


def get_all_spans(text, max_ngram_length):
    """
    Split a text into all possible ngrams up to 'max_ngram_length'. Split points are white space and punctuation.

    Args:
      text: Text to split.
      max_ngram_length: maximal ngram length.
    Yields:
      Spans, tuples of begin-end index.
    """
    start_indexes = []
    for index, char in enumerate(text):
        if not char.isalnum():
            continue
        if index == 0 or not text[index - 1].isalnum():
            start_indexes.append(index)
        if index + 1 == len(text) or not text[index + 1].isalnum():
            for start_index in start_indexes[-max_ngram_length:]:
                yield start_index, index + 1


def normalize_for_match(text):
    return " ".join(text.lower().split())


def format_text(text):
    """Lowercases and strips punctuation."""
    text = text.lower().strip()
    if text == "n/a" or text == "?" or text == "nan":
        text = EMPTY_TEXT

    text = re.sub(r"[^\w\d]+", " ", text).replace("_", " ")
    text = " ".join(text.split())
    text = text.strip()
    if text:
        return text
    return EMPTY_TEXT


def parse_text(text):
    """
    Extracts longest number and date spans.

    Args:
      text: text to annotate

    Returns:
      List of longest numeric value spans.
    """
    span_dict = collections.defaultdict(list)
    for match in _NUMBER_PATTERN.finditer(text):
        span_text = text[match.start() : match.end()]
        number = _parse_number(span_text)
        if number is not None:
            span_dict[match.span()].append(_get_numeric_value_from_float(number))

    for begin_index, end_index in get_all_spans(text, max_ngram_length=1):
        if (begin_index, end_index) in span_dict:
            continue
        span_text = text[begin_index:end_index]

        number = _parse_number(span_text)
        if number is not None:
            span_dict[begin_index, end_index].append(_get_numeric_value_from_float(number))
        for number, word in enumerate(_NUMBER_WORDS):
            if span_text == word:
                span_dict[begin_index, end_index].append(_get_numeric_value_from_float(float(number)))
                break
        for number, word in enumerate(_ORDINAL_WORDS):
            if span_text == word:
                span_dict[begin_index, end_index].append(_get_numeric_value_from_float(float(number)))
                break

    for begin_index, end_index in get_all_spans(text, max_ngram_length=_MAX_DATE_NGRAM_SIZE):
        span_text = text[begin_index:end_index]
        date = _parse_date(span_text)
        if date is not None:
            span_dict[begin_index, end_index].append(date)

    spans = sorted(span_dict.items(), key=lambda span_value: _get_span_length_key(span_value[0]), reverse=True)
    selected_spans = []
    for span, value in spans:
        for selected_span, _ in selected_spans:
            if selected_span[0] <= span[0] and span[1] <= selected_span[1]:
                break
        else:
            selected_spans.append((span, value))

    selected_spans.sort(key=lambda span_value: span_value[0][0])

    numeric_value_spans = []
    for span, values in selected_spans:
        numeric_value_spans.append(NumericValueSpan(begin_index=span[0], end_index=span[1], values=values))
    return numeric_value_spans


# Below: all functions from number_annotation_utils.py and 2 functions (namely filter_invalid_unicode
# and filter_invalid_unicode_from_table) from text_utils.py of the original implementation. URL's:
# - https://github.com/google-research/tapas/blob/master/tapas/utils/number_annotation_utils.py
# - https://github.com/google-research/tapas/blob/master/tapas/utils/text_utils.py


_PrimitiveNumericValue = Union[float, Tuple[Optional[float], Optional[float], Optional[float]]]
_SortKeyFn = Callable[[NumericValue], Tuple[float, Ellipsis]]

_DATE_TUPLE_SIZE = 3

EMPTY_TEXT = "EMPTY"

NUMBER_TYPE = "number"
DATE_TYPE = "date"


def _get_value_type(numeric_value):
    if numeric_value.float_value is not None:
        return NUMBER_TYPE
    elif numeric_value.date is not None:
        return DATE_TYPE
    raise ValueError(f"Unknown type: {numeric_value}")


def _get_value_as_primitive_value(numeric_value):
    """Maps a NumericValue proto to a float or tuple of float."""
    if numeric_value.float_value is not None:
        return numeric_value.float_value
    if numeric_value.date is not None:
        date = numeric_value.date
        value_tuple = [None, None, None]
        # All dates fields are cased to float to produce a simple primitive value.
        if date.year is not None:
            value_tuple[0] = float(date.year)
        if date.month is not None:
            value_tuple[1] = float(date.month)
        if date.day is not None:
            value_tuple[2] = float(date.day)
        return tuple(value_tuple)
    raise ValueError(f"Unknown type: {numeric_value}")


def _get_all_types(numeric_values):
    return {_get_value_type(value) for value in numeric_values}


def get_numeric_sort_key_fn(numeric_values):
    """
    Creates a function that can be used as a sort key or to compare the values. Maps to primitive types and finds the
    biggest common subset. Consider the values "05/05/2010" and "August 2007". With the corresponding primitive values
    (2010.,5.,5.) and (2007.,8., None). These values can be compared by year and date so we map to the sequence (2010.,
    5.), (2007., 8.). If we added a third value "2006" with primitive value (2006., None, None), we could only compare
    by the year so we would map to (2010.,), (2007.,) and (2006.,).

    Args:
     numeric_values: Values to compare

    Returns:
     A function that can be used as a sort key function (mapping numeric values to a comparable tuple)

    Raises:
      ValueError if values don't have a common type or are not comparable.
    """
    value_types = _get_all_types(numeric_values)
    if len(value_types) != 1:
        raise ValueError(f"No common value type in {numeric_values}")

    value_type = next(iter(value_types))
    if value_type == NUMBER_TYPE:
        # Primitive values are simple floats, nothing to do here.
        return _get_value_as_primitive_value

    # The type can only be Date at this point which means the primitive type
    # is a float triple.
    valid_indexes = set(range(_DATE_TUPLE_SIZE))

    for numeric_value in numeric_values:
        value = _get_value_as_primitive_value(numeric_value)
        assert isinstance(value, tuple)
        for tuple_index, inner_value in enumerate(value):
            if inner_value is None:
                valid_indexes.discard(tuple_index)

    if not valid_indexes:
        raise ValueError(f"No common value in {numeric_values}")

    def _sort_key_fn(numeric_value):
        value = _get_value_as_primitive_value(numeric_value)
        return tuple(value[index] for index in valid_indexes)

    return _sort_key_fn


def _consolidate_numeric_values(row_index_to_values, min_consolidation_fraction, debug_info):
    """
    Finds the most common numeric values in a column and returns them

    Args:
        row_index_to_values:
            For each row index all the values in that cell.
        min_consolidation_fraction:
            Fraction of cells that need to have consolidated value.
        debug_info:
            Additional information only used for logging

    Returns:
        For each row index the first value that matches the most common value. Rows that don't have a matching value
        are dropped. Empty list if values can't be consolidated.
    """
    type_counts = collections.Counter()
    for numeric_values in row_index_to_values.values():
        type_counts.update(_get_all_types(numeric_values))
    if not type_counts:
        return {}
    max_count = max(type_counts.values())
    if max_count < len(row_index_to_values) * min_consolidation_fraction:
        # logging.log_every_n(logging.INFO, f'Can\'t consolidate types: {debug_info} {row_index_to_values} {max_count}', 100)
        return {}

    valid_types = set()
    for value_type, count in type_counts.items():
        if count == max_count:
            valid_types.add(value_type)
    if len(valid_types) > 1:
        assert DATE_TYPE in valid_types
        max_type = DATE_TYPE
    else:
        max_type = next(iter(valid_types))

    new_row_index_to_value = {}
    for index, values in row_index_to_values.items():
        # Extract the first matching value.
        for value in values:
            if _get_value_type(value) == max_type:
                new_row_index_to_value[index] = value
                break

    return new_row_index_to_value


def _get_numeric_values(text):
    """Parses text and returns numeric values."""
    numeric_spans = parse_text(text)
    return itertools.chain(*(span.values for span in numeric_spans))


def _get_column_values(table, col_index):
    """
    Parses text in column and returns a dict mapping row_index to values. This is the _get_column_values function from
    number_annotation_utils.py of the original implementation

    Args:
      table: Pandas dataframe
      col_index: integer, indicating the index of the column to get the numeric values of
    """
    index_to_values = {}
    for row_index, row in table.iterrows():
        text = normalize_for_match(row[col_index].text)
        index_to_values[row_index] = list(_get_numeric_values(text))
    return index_to_values


def get_numeric_relation(value, other_value, sort_key_fn):
    """Compares two values and returns their relation or None."""
    value = sort_key_fn(value)
    other_value = sort_key_fn(other_value)
    if value == other_value:
        return Relation.EQ
    if value < other_value:
        return Relation.LT
    if value > other_value:
        return Relation.GT
    return None


def add_numeric_values_to_question(question):
    """Adds numeric value spans to a question."""
    original_text = question
    question = normalize_for_match(question)
    numeric_spans = parse_text(question)
    return Question(original_text=original_text, text=question, numeric_spans=numeric_spans)


def filter_invalid_unicode(text):
    """Return an empty string and True if 'text' is in invalid unicode."""
    return ("", True) if isinstance(text, bytes) else (text, False)


def filter_invalid_unicode_from_table(table):
    """
    Removes invalid unicode from table. Checks whether a table cell text contains an invalid unicode encoding. If yes,
    reset the table cell text to an empty str and log a warning for each invalid cell

    Args:
        table: table to clean.
    """
    # to do: add table id support
    if not hasattr(table, "table_id"):
        table.table_id = 0

    for row_index, row in table.iterrows():
        for col_index, cell in enumerate(row):
            cell, is_invalid = filter_invalid_unicode(cell)
            if is_invalid:
                logging.warning(
                    f"Scrub an invalid table body @ table_id: {table.table_id}, row_index: {row_index}, "
                    f"col_index: {col_index}",
                )
    for col_index, column in enumerate(table.columns):
        column, is_invalid = filter_invalid_unicode(column)
        if is_invalid:
            logging.warning(f"Scrub an invalid table header @ table_id: {table.table_id}, col_index: {col_index}")


def add_numeric_table_values(table, min_consolidation_fraction=0.7, debug_info=None):
    """
    Parses text in table column-wise and adds the consolidated values. Consolidation refers to finding values with a
    common types (date or number)

    Args:
        table:
            Table to annotate.
        min_consolidation_fraction:
            Fraction of cells in a column that need to have consolidated value.
        debug_info:
            Additional information used for logging.
    """
    table = table.copy()
    # First, filter table on invalid unicode
    filter_invalid_unicode_from_table(table)

    # Second, replace cell values by Cell objects
    for row_index, row in table.iterrows():
        for col_index, cell in enumerate(row):
            table.iloc[row_index, col_index] = Cell(text=cell)

    # Third, add numeric_value attributes to these Cell objects
    for col_index, column in enumerate(table.columns):
        column_values = _consolidate_numeric_values(
            _get_column_values(table, col_index),
            min_consolidation_fraction=min_consolidation_fraction,
            debug_info=(debug_info, column),
        )

        for row_index, numeric_value in column_values.items():
            table.iloc[row_index, col_index].numeric_value = numeric_value

    return table


__all__ = ["TapasTokenizer"]

# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team, The Hugging Face Team.
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
"""Tokenization classes for DPR."""

import collections
from typing import List, Optional, Union

from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, add_end_docstrings, add_start_docstrings, logging
from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_dpr import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRReaderTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


class DPRContextEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" DPRContextEncoder tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DPRContextEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = DPRContextEncoderTokenizer


class DPRQuestionEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a "fast" DPRQuestionEncoder tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DPRQuestionEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = DPRQuestionEncoderTokenizer


DPRSpanPrediction = collections.namedtuple(
    "DPRSpanPrediction", ["span_score", "relevance_score", "doc_id", "start_index", "end_index", "text"]
)

DPRReaderOutput = collections.namedtuple("DPRReaderOutput", ["start_logits", "end_logits", "relevance_logits"])


CUSTOM_DPR_READER_DOCSTRING = r"""
    Return a dictionary with the token ids of the input strings and other information to give to `.decode_best_spans`.
    It converts the strings of a question and different passages (title and text) in a sequence of IDs (integers),
    using the tokenizer and vocabulary. The resulting `input_ids` is a matrix of size `(n_passages, sequence_length)`
    with the format:

    [CLS] <question token ids> [SEP] <titles ids> [SEP] <texts ids>

    Args:
        questions (`str` or `List[str]`):
            The questions to be encoded. You can specify one question for many passages. In this case, the question
            will be duplicated like `[questions] * n_passages`. Otherwise you have to specify as many questions as in
            `titles` or `texts`.
        titles (`str` or `List[str]`):
            The passages titles to be encoded. This can be a string or a list of strings if there are several passages.
        texts (`str` or `List[str]`):
            The passages texts to be encoded. This can be a string or a list of strings if there are several passages.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
            Activates and controls padding. Accepts the following values:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
            Activates and controls truncation. Accepts the following values:

            - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to
              the maximum acceptable input length for the model if that argument is not provided. This will truncate
              token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch
              of pairs) is provided.
            - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided. This will only truncate the first
              sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided. This will only truncate the
              second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
            - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
              greater than the model maximum admissible input size).
        max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
        return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        return_attention_mask (`bool`, *optional*):
            Whether or not to return the attention mask. If not set, will return the attention mask according to the
            specific tokenizer's default, defined by the `return_outputs` attribute.

            [What are attention masks?](../glossary#attention-mask)

    Return:
        `Dict[str, List[List[int]]]`: A dictionary with the following keys:

        - `input_ids`: List of token ids to be fed to a model.
        - `attention_mask`: List of indices specifying which tokens should be attended to by the model.
    """


@add_start_docstrings(CUSTOM_DPR_READER_DOCSTRING)
class CustomDPRReaderTokenizerMixin:
    def __call__(
        self,
        questions,
        titles: Optional[str] = None,
        texts: Optional[str] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs,
    ) -> BatchEncoding:
        if titles is None and texts is None:
            return super().__call__(
                questions,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )
        elif titles is None or texts is None:
            text_pair = titles if texts is None else texts
            return super().__call__(
                questions,
                text_pair,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )
        titles = titles if not isinstance(titles, str) else [titles]
        texts = texts if not isinstance(texts, str) else [texts]
        n_passages = len(titles)
        questions = questions if not isinstance(questions, str) else [questions] * n_passages
        assert len(titles) == len(
            texts
        ), f"There should be as many titles than texts but got {len(titles)} titles and {len(texts)} texts."
        encoded_question_and_titles = super().__call__(questions, titles, padding=False, truncation=False)["input_ids"]
        encoded_texts = super().__call__(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        encoded_inputs = {
            "input_ids": [
                (encoded_question_and_title + encoded_text)[:max_length]
                if max_length is not None and truncation
                else encoded_question_and_title + encoded_text
                for encoded_question_and_title, encoded_text in zip(encoded_question_and_titles, encoded_texts)
            ]
        }
        if return_attention_mask is not False:
            attention_mask = []
            for input_ids in encoded_inputs["input_ids"]:
                attention_mask.append([int(input_id != self.pad_token_id) for input_id in input_ids])
            encoded_inputs["attention_mask"] = attention_mask
        return self.pad(encoded_inputs, padding=padding, max_length=max_length, return_tensors=return_tensors)

    def decode_best_spans(
        self,
        reader_input: BatchEncoding,
        reader_output: DPRReaderOutput,
        num_spans: int = 16,
        max_answer_length: int = 64,
        num_spans_per_passage: int = 4,
    ) -> List[DPRSpanPrediction]:
        """
        Get the span predictions for the extractive Q&A model.

        Returns: *List* of *DPRReaderOutput* sorted by descending *(relevance_score, span_score)*. Each
        *DPRReaderOutput* is a *Tuple* with:

            - **span_score**: `float` that corresponds to the score given by the reader for this span compared to other
              spans in the same passage. It corresponds to the sum of the start and end logits of the span.
            - **relevance_score**: `float` that corresponds to the score of the each passage to answer the question,
              compared to all the other passages. It corresponds to the output of the QA classifier of the DPRReader.
            - **doc_id**: `int` the id of the passage. - ***start_index**: `int` the start index of the span
              (inclusive). - **end_index**: `int` the end index of the span (inclusive).

        Examples:

        ```python
        >>> from transformers import DPRReader, DPRReaderTokenizer

        >>> tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
        >>> model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
        >>> encoded_inputs = tokenizer(
        ...     questions=["What is love ?"],
        ...     titles=["Haddaway"],
        ...     texts=["'What Is Love' is a song recorded by the artist Haddaway"],
        ...     return_tensors="pt",
        ... )
        >>> outputs = model(**encoded_inputs)
        >>> predicted_spans = tokenizer.decode_best_spans(encoded_inputs, outputs)
        >>> print(predicted_spans[0].text)  # best span
        a song
        ```"""
        input_ids = reader_input["input_ids"]
        start_logits, end_logits, relevance_logits = reader_output[:3]
        n_passages = len(relevance_logits)
        sorted_docs = sorted(range(n_passages), reverse=True, key=relevance_logits.__getitem__)
        nbest_spans_predictions: List[DPRReaderOutput] = []
        for doc_id in sorted_docs:
            sequence_ids = list(input_ids[doc_id])
            # assuming question & title information is at the beginning of the sequence
            passage_offset = sequence_ids.index(self.sep_token_id, 2) + 1  # second sep id
            if sequence_ids[-1] == self.pad_token_id:
                sequence_len = sequence_ids.index(self.pad_token_id)
            else:
                sequence_len = len(sequence_ids)

            best_spans = self._get_best_spans(
                start_logits=start_logits[doc_id][passage_offset:sequence_len],
                end_logits=end_logits[doc_id][passage_offset:sequence_len],
                max_answer_length=max_answer_length,
                top_spans=num_spans_per_passage,
            )
            for start_index, end_index in best_spans:
                start_index += passage_offset
                end_index += passage_offset
                nbest_spans_predictions.append(
                    DPRSpanPrediction(
                        span_score=start_logits[doc_id][start_index] + end_logits[doc_id][end_index],
                        relevance_score=relevance_logits[doc_id],
                        doc_id=doc_id,
                        start_index=start_index,
                        end_index=end_index,
                        text=self.decode(sequence_ids[start_index : end_index + 1]),
                    )
                )
            if len(nbest_spans_predictions) >= num_spans:
                break
        return nbest_spans_predictions[:num_spans]

    def _get_best_spans(
        self,
        start_logits: List[int],
        end_logits: List[int],
        max_answer_length: int,
        top_spans: int,
    ) -> List[DPRSpanPrediction]:
        """
        Finds the best answer span for the extractive Q&A model for one passage. It returns the best span by descending
        `span_score` order and keeping max `top_spans` spans. Spans longer that `max_answer_length` are ignored.
        """
        scores = []
        for start_index, start_score in enumerate(start_logits):
            for answer_length, end_score in enumerate(end_logits[start_index : start_index + max_answer_length]):
                scores.append(((start_index, start_index + answer_length), start_score + end_score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        chosen_span_intervals = []
        for (start_index, end_index), score in scores:
            assert start_index <= end_index, f"Wrong span indices: [{start_index}:{end_index}]"
            length = end_index - start_index + 1
            assert length <= max_answer_length, f"Span is too long: {length} > {max_answer_length}"
            if any(
                start_index <= prev_start_index <= prev_end_index <= end_index
                or prev_start_index <= start_index <= end_index <= prev_end_index
                for (prev_start_index, prev_end_index) in chosen_span_intervals
            ):
                continue
            chosen_span_intervals.append((start_index, end_index))

            if len(chosen_span_intervals) == top_spans:
                break
        return chosen_span_intervals


@add_end_docstrings(CUSTOM_DPR_READER_DOCSTRING)
class DPRReaderTokenizerFast(CustomDPRReaderTokenizerMixin, BertTokenizerFast):
    r"""
    Constructs a "fast" DPRReader tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DPRReaderTokenizerFast`] is almost identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece. The difference is that is has three inputs strings: question, titles and texts
    that are combined to be fed to the [`DPRReader`] model.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.

    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = DPRReaderTokenizer


__all__ = ["DPRContextEncoderTokenizerFast", "DPRQuestionEncoderTokenizerFast", "DPRReaderTokenizerFast"]

# coding=utf-8
# Copyright 2021 Studio Ousia and the HuggingFace Inc. team.
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
# limitations under the License
"""Tokenization classes for mLUKE."""

import itertools
import json
import os
from collections.abc import Mapping
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sentencepiece as spm

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
    to_py_obj,
)
from ...utils import add_end_docstrings, is_tf_tensor, is_torch_tensor, logging


logger = logging.get_logger(__name__)

EntitySpan = Tuple[int, int]
EntitySpanInput = List[EntitySpan]
Entity = str
EntityInput = List[Entity]

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "entity_vocab_file": "entity_vocab.json"}


ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = r"""
            return_token_type_ids (`bool`, *optional*):
                Whether to return token type IDs. If left to the default, will return the token type IDs according to
                the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are token type IDs?](../glossary#token-type-ids)
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_overflowing_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
                of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
                of returning overflowing tokens.
            return_special_tokens_mask (`bool`, *optional*, defaults to `False`):
                Whether or not to return special tokens mask information.
            return_offsets_mapping (`bool`, *optional*, defaults to `False`):
                Whether or not to return `(char_start, char_end)` for each token.

                This is only available on fast tokenizers inheriting from [`PreTrainedTokenizerFast`], if using
                Python's tokenizer, this method will raise `NotImplementedError`.
            return_length  (`bool`, *optional*, defaults to `False`):
                Whether or not to return the lengths of the encoded inputs.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
            **kwargs: passed to the `self.tokenize()` method

        Return:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.

              [What are input IDs?](../glossary#input-ids)

            - **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
              if *"token_type_ids"* is in `self.model_input_names`).

              [What are token type IDs?](../glossary#token-type-ids)

            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

              [What are attention masks?](../glossary#attention-mask)

            - **entity_ids** -- List of entity ids to be fed to a model.

              [What are input IDs?](../glossary#input-ids)

            - **entity_position_ids** -- List of entity positions in the input sequence to be fed to a model.

            - **entity_token_type_ids** -- List of entity token type ids to be fed to a model (when
              `return_token_type_ids=True` or if *"entity_token_type_ids"* is in `self.model_input_names`).

              [What are token type IDs?](../glossary#token-type-ids)

            - **entity_attention_mask** -- List of indices specifying which entities should be attended to by the model
              (when `return_attention_mask=True` or if *"entity_attention_mask"* is in `self.model_input_names`).

              [What are attention masks?](../glossary#attention-mask)

            - **entity_start_positions** -- List of the start positions of entities in the word token sequence (when
              `task="entity_span_classification"`).
            - **entity_end_positions** -- List of the end positions of entities in the word token sequence (when
              `task="entity_span_classification"`).
            - **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
              `return_overflowing_tokens=True`).
            - **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
              `return_overflowing_tokens=True`).
            - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
              regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
            - **length** -- The length of the inputs (when `return_length=True`)

"""


class MLukeTokenizer(PreTrainedTokenizer):
    """
    Adapted from [`XLMRobertaTokenizer`] and [`LukeTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        entity_vocab_file (`str`):
            Path to the entity vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        task (`str`, *optional*):
            Task for which you want to prepare sequences. One of `"entity_classification"`,
            `"entity_pair_classification"`, or `"entity_span_classification"`. If you specify this argument, the entity
            sequence is automatically created based on the given entity span(s).
        max_entity_length (`int`, *optional*, defaults to 32):
            The maximum length of `entity_ids`.
        max_mention_length (`int`, *optional*, defaults to 30):
            The maximum number of tokens inside an entity span.
        entity_token_1 (`str`, *optional*, defaults to `<ent>`):
            The special token used to represent an entity span in a word token sequence. This token is only used when
            `task` is set to `"entity_classification"` or `"entity_pair_classification"`.
        entity_token_2 (`str`, *optional*, defaults to `<ent2>`):
            The special token used to represent an entity span in a word token sequence. This token is only used when
            `task` is set to `"entity_pair_classification"`.
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`):
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

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        entity_vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        task=None,
        max_entity_length=32,
        max_mention_length=30,
        entity_token_1="<ent>",
        entity_token_2="<ent2>",
        entity_unk_token="[UNK]",
        entity_pad_token="[PAD]",
        entity_mask_token="[MASK]",
        entity_mask2_token="[MASK2]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # we add 2 special tokens for downstream tasks
        # for more information about lstrip and rstrip, see https://github.com/huggingface/transformers/pull/2778
        entity_token_1 = (
            AddedToken(entity_token_1, lstrip=False, rstrip=False)
            if isinstance(entity_token_1, str)
            else entity_token_1
        )
        entity_token_2 = (
            AddedToken(entity_token_2, lstrip=False, rstrip=False)
            if isinstance(entity_token_2, str)
            else entity_token_2
        )
        additional_special_tokens = kwargs.pop("additional_special_tokens", [])
        additional_special_tokens += [entity_token_1, entity_token_2]

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # Original fairseq vocab and spm vocab must be "aligned":
        # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
        # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
        # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
        # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

        # Mimic fairseq token-to-id alignment for the first 4 token
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.fairseq_offset = 1

        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        with open(entity_vocab_file, encoding="utf-8") as entity_vocab_handle:
            self.entity_vocab = json.load(entity_vocab_handle)
        for entity_special_token in [entity_unk_token, entity_pad_token, entity_mask_token, entity_mask2_token]:
            if entity_special_token not in self.entity_vocab:
                raise ValueError(
                    f"Specified entity special token ``{entity_special_token}`` is not found in entity_vocab. "
                    f"Probably an incorrect entity vocab file is loaded: {entity_vocab_file}."
                )
        self.entity_unk_token_id = self.entity_vocab[entity_unk_token]
        self.entity_pad_token_id = self.entity_vocab[entity_pad_token]
        self.entity_mask_token_id = self.entity_vocab[entity_mask_token]
        self.entity_mask2_token_id = self.entity_vocab[entity_mask2_token]

        self.task = task
        if task is None or task == "entity_span_classification":
            self.max_entity_length = max_entity_length
        elif task == "entity_classification":
            self.max_entity_length = 1
        elif task == "entity_pair_classification":
            self.max_entity_length = 2
        else:
            raise ValueError(
                f"Task {task} not supported. Select task from ['entity_classification', 'entity_pair_classification',"
                " 'entity_span_classification'] only."
            )

        self.max_mention_length = max_mention_length

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            task=task,
            max_entity_length=max_entity_length,
            max_mention_length=max_mention_length,
            entity_token_1=entity_token_1,
            entity_token_2=entity_token_2,
            entity_unk_token=entity_unk_token,
            entity_pad_token=entity_pad_token,
            entity_mask_token=entity_mask_token,
            entity_mask2_token=entity_mask2_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    # Copied from transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer.vocab_size
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token

    # Copied from transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer.get_vocab
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer._tokenize
    def _tokenize(self, text: str) -> List[str]:
        # TODO check if the t5/llama PR also applies here
        return self.sp_model.encode(text, out_type=str)

    # Copied from transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer.__call__
    def __call__(
        self,
        text: Union[TextInput, List[TextInput]],
        text_pair: Optional[Union[TextInput, List[TextInput]]] = None,
        entity_spans: Optional[Union[EntitySpanInput, List[EntitySpanInput]]] = None,
        entity_spans_pair: Optional[Union[EntitySpanInput, List[EntitySpanInput]]] = None,
        entities: Optional[Union[EntityInput, List[EntityInput]]] = None,
        entities_pair: Optional[Union[EntityInput, List[EntityInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: Optional[bool] = False,
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
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences, depending on the task you want to prepare them for.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
                tokenizer does not support tokenization based on pretokenized strings.
            text_pair (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
                tokenizer does not support tokenization based on pretokenized strings.
            entity_spans (`List[Tuple[int, int]]`, `List[List[Tuple[int, int]]]`, *optional*):
                The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
                with two integers denoting character-based start and end positions of entities. If you specify
                `"entity_classification"` or `"entity_pair_classification"` as the `task` argument in the constructor,
                the length of each sequence must be 1 or 2, respectively. If you specify `entities`, the length of each
                sequence must be equal to the length of each sequence of `entities`.
            entity_spans_pair (`List[Tuple[int, int]]`, `List[List[Tuple[int, int]]]`, *optional*):
                The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
                with two integers denoting character-based start and end positions of entities. If you specify the
                `task` argument in the constructor, this argument is ignored. If you specify `entities_pair`, the
                length of each sequence must be equal to the length of each sequence of `entities_pair`.
            entities (`List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
                representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
                Angeles). This argument is ignored if you specify the `task` argument in the constructor. The length of
                each sequence must be equal to the length of each sequence of `entity_spans`. If you specify
                `entity_spans` without specifying this argument, the entity sequence or the batch of entity sequences
                is automatically constructed by filling it with the [MASK] entity.
            entities_pair (`List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
                representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
                Angeles). This argument is ignored if you specify the `task` argument in the constructor. The length of
                each sequence must be equal to the length of each sequence of `entity_spans_pair`. If you specify
                `entity_spans_pair` without specifying this argument, the entity sequence or the batch of entity
                sequences is automatically constructed by filling it with the [MASK] entity.
            max_entity_length (`int`, *optional*):
                The maximum length of `entity_ids`.
        """
        # Input type checking for clearer error
        is_valid_single_text = isinstance(text, str)
        is_valid_batch_text = isinstance(text, (list, tuple)) and (len(text) == 0 or (isinstance(text[0], str)))
        if not (is_valid_single_text or is_valid_batch_text):
            raise ValueError("text input must be of type `str` (single example) or `List[str]` (batch).")

        is_valid_single_text_pair = isinstance(text_pair, str)
        is_valid_batch_text_pair = isinstance(text_pair, (list, tuple)) and (
            len(text_pair) == 0 or isinstance(text_pair[0], str)
        )
        if not (text_pair is None or is_valid_single_text_pair or is_valid_batch_text_pair):
            raise ValueError("text_pair input must be of type `str` (single example) or `List[str]` (batch).")

        is_batched = bool(isinstance(text, (list, tuple)))

        if is_batched:
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            if entities is None:
                batch_entities_or_entities_pairs = None
            else:
                batch_entities_or_entities_pairs = (
                    list(zip(entities, entities_pair)) if entities_pair is not None else entities
                )

            if entity_spans is None:
                batch_entity_spans_or_entity_spans_pairs = None
            else:
                batch_entity_spans_or_entity_spans_pairs = (
                    list(zip(entity_spans, entity_spans_pair)) if entity_spans_pair is not None else entity_spans
                )

            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                batch_entity_spans_or_entity_spans_pairs=batch_entity_spans_or_entity_spans_pairs,
                batch_entities_or_entities_pairs=batch_entities_or_entities_pairs,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                max_entity_length=max_entity_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
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
                text=text,
                text_pair=text_pair,
                entity_spans=entity_spans,
                entity_spans_pair=entity_spans_pair,
                entities=entities,
                entities_pair=entities_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                max_entity_length=max_entity_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
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

    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer._encode_plus
    def _encode_plus(
        self,
        text: Union[TextInput],
        text_pair: Optional[Union[TextInput]] = None,
        entity_spans: Optional[EntitySpanInput] = None,
        entity_spans_pair: Optional[EntitySpanInput] = None,
        entities: Optional[EntityInput] = None,
        entities_pair: Optional[EntityInput] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: Optional[bool] = False,
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
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        if is_split_into_words:
            raise NotImplementedError("is_split_into_words is not supported in this tokenizer.")

        (
            first_ids,
            second_ids,
            first_entity_ids,
            second_entity_ids,
            first_entity_token_spans,
            second_entity_token_spans,
        ) = self._create_input_sequence(
            text=text,
            text_pair=text_pair,
            entities=entities,
            entities_pair=entities_pair,
            entity_spans=entity_spans,
            entity_spans_pair=entity_spans_pair,
            **kwargs,
        )

        # prepare_for_model will create the attention_mask and token_type_ids
        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            entity_ids=first_entity_ids,
            pair_entity_ids=second_entity_ids,
            entity_token_spans=first_entity_token_spans,
            pair_entity_token_spans=second_entity_token_spans,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            max_entity_length=max_entity_length,
            stride=stride,
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

    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer._batch_encode_plus
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair]],
        batch_entity_spans_or_entity_spans_pairs: Optional[
            Union[List[EntitySpanInput], List[Tuple[EntitySpanInput, EntitySpanInput]]]
        ] = None,
        batch_entities_or_entities_pairs: Optional[
            Union[List[EntityInput], List[Tuple[EntityInput, EntityInput]]]
        ] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: Optional[bool] = False,
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
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        if is_split_into_words:
            raise NotImplementedError("is_split_into_words is not supported in this tokenizer.")

        # input_ids is a list of tuples (one for each example in the batch)
        input_ids = []
        entity_ids = []
        entity_token_spans = []
        for index, text_or_text_pair in enumerate(batch_text_or_text_pairs):
            if not isinstance(text_or_text_pair, (list, tuple)):
                text, text_pair = text_or_text_pair, None
            else:
                text, text_pair = text_or_text_pair

            entities, entities_pair = None, None
            if batch_entities_or_entities_pairs is not None:
                entities_or_entities_pairs = batch_entities_or_entities_pairs[index]
                if entities_or_entities_pairs:
                    if isinstance(entities_or_entities_pairs[0], str):
                        entities, entities_pair = entities_or_entities_pairs, None
                    else:
                        entities, entities_pair = entities_or_entities_pairs

            entity_spans, entity_spans_pair = None, None
            if batch_entity_spans_or_entity_spans_pairs is not None:
                entity_spans_or_entity_spans_pairs = batch_entity_spans_or_entity_spans_pairs[index]
                if len(entity_spans_or_entity_spans_pairs) > 0 and isinstance(
                    entity_spans_or_entity_spans_pairs[0], list
                ):
                    entity_spans, entity_spans_pair = entity_spans_or_entity_spans_pairs
                else:
                    entity_spans, entity_spans_pair = entity_spans_or_entity_spans_pairs, None

            (
                first_ids,
                second_ids,
                first_entity_ids,
                second_entity_ids,
                first_entity_token_spans,
                second_entity_token_spans,
            ) = self._create_input_sequence(
                text=text,
                text_pair=text_pair,
                entities=entities,
                entities_pair=entities_pair,
                entity_spans=entity_spans,
                entity_spans_pair=entity_spans_pair,
                **kwargs,
            )
            input_ids.append((first_ids, second_ids))
            entity_ids.append((first_entity_ids, second_entity_ids))
            entity_token_spans.append((first_entity_token_spans, second_entity_token_spans))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            batch_entity_ids_pairs=entity_ids,
            batch_entity_token_spans_pairs=entity_token_spans,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            max_entity_length=max_entity_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)

    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer._check_entity_input_format
    def _check_entity_input_format(self, entities: Optional[EntityInput], entity_spans: Optional[EntitySpanInput]):
        if not isinstance(entity_spans, list):
            raise TypeError("entity_spans should be given as a list")
        elif len(entity_spans) > 0 and not isinstance(entity_spans[0], tuple):
            raise ValueError(
                "entity_spans should be given as a list of tuples containing the start and end character indices"
            )

        if entities is not None:
            if not isinstance(entities, list):
                raise ValueError("If you specify entities, they should be given as a list")

            if len(entities) > 0 and not isinstance(entities[0], str):
                raise ValueError("If you specify entities, they should be given as a list of entity names")

            if len(entities) != len(entity_spans):
                raise ValueError("If you specify entities, entities and entity_spans must be the same length")

    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer._create_input_sequence
    def _create_input_sequence(
        self,
        text: Union[TextInput],
        text_pair: Optional[Union[TextInput]] = None,
        entities: Optional[EntityInput] = None,
        entities_pair: Optional[EntityInput] = None,
        entity_spans: Optional[EntitySpanInput] = None,
        entity_spans_pair: Optional[EntitySpanInput] = None,
        **kwargs,
    ) -> Tuple[list, list, list, list, list, list]:
        def get_input_ids(text):
            tokens = self.tokenize(text, **kwargs)
            return self.convert_tokens_to_ids(tokens)

        def get_input_ids_and_entity_token_spans(text, entity_spans):
            if entity_spans is None:
                return get_input_ids(text), None

            cur = 0
            input_ids = []
            entity_token_spans = [None] * len(entity_spans)

            split_char_positions = sorted(frozenset(itertools.chain(*entity_spans)))
            char_pos2token_pos = {}

            for split_char_position in split_char_positions:
                orig_split_char_position = split_char_position
                if (
                    split_char_position > 0 and text[split_char_position - 1] == " "
                ):  # whitespace should be prepended to the following token
                    split_char_position -= 1
                if cur != split_char_position:
                    input_ids += get_input_ids(text[cur:split_char_position])
                    cur = split_char_position
                char_pos2token_pos[orig_split_char_position] = len(input_ids)

            input_ids += get_input_ids(text[cur:])

            entity_token_spans = [
                (char_pos2token_pos[char_start], char_pos2token_pos[char_end]) for char_start, char_end in entity_spans
            ]

            return input_ids, entity_token_spans

        first_ids, second_ids = None, None
        first_entity_ids, second_entity_ids = None, None
        first_entity_token_spans, second_entity_token_spans = None, None

        if self.task is None:
            if entity_spans is None:
                first_ids = get_input_ids(text)
            else:
                self._check_entity_input_format(entities, entity_spans)

                first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(text, entity_spans)
                if entities is None:
                    first_entity_ids = [self.entity_mask_token_id] * len(entity_spans)
                else:
                    first_entity_ids = [self.entity_vocab.get(entity, self.entity_unk_token_id) for entity in entities]

            if text_pair is not None:
                if entity_spans_pair is None:
                    second_ids = get_input_ids(text_pair)
                else:
                    self._check_entity_input_format(entities_pair, entity_spans_pair)

                    second_ids, second_entity_token_spans = get_input_ids_and_entity_token_spans(
                        text_pair, entity_spans_pair
                    )
                    if entities_pair is None:
                        second_entity_ids = [self.entity_mask_token_id] * len(entity_spans_pair)
                    else:
                        second_entity_ids = [
                            self.entity_vocab.get(entity, self.entity_unk_token_id) for entity in entities_pair
                        ]

        elif self.task == "entity_classification":
            if not (isinstance(entity_spans, list) and len(entity_spans) == 1 and isinstance(entity_spans[0], tuple)):
                raise ValueError(
                    "Entity spans should be a list containing a single tuple "
                    "containing the start and end character indices of an entity"
                )
            first_entity_ids = [self.entity_mask_token_id]
            first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(text, entity_spans)

            # add special tokens to input ids
            entity_token_start, entity_token_end = first_entity_token_spans[0]
            first_ids = (
                first_ids[:entity_token_end] + [self.additional_special_tokens_ids[0]] + first_ids[entity_token_end:]
            )
            first_ids = (
                first_ids[:entity_token_start]
                + [self.additional_special_tokens_ids[0]]
                + first_ids[entity_token_start:]
            )
            first_entity_token_spans = [(entity_token_start, entity_token_end + 2)]

        elif self.task == "entity_pair_classification":
            if not (
                isinstance(entity_spans, list)
                and len(entity_spans) == 2
                and isinstance(entity_spans[0], tuple)
                and isinstance(entity_spans[1], tuple)
            ):
                raise ValueError(
                    "Entity spans should be provided as a list of two tuples, "
                    "each tuple containing the start and end character indices of an entity"
                )

            head_span, tail_span = entity_spans
            first_entity_ids = [self.entity_mask_token_id, self.entity_mask2_token_id]
            first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(text, entity_spans)

            head_token_span, tail_token_span = first_entity_token_spans
            token_span_with_special_token_ids = [
                (head_token_span, self.additional_special_tokens_ids[0]),
                (tail_token_span, self.additional_special_tokens_ids[1]),
            ]
            if head_token_span[0] < tail_token_span[0]:
                first_entity_token_spans[0] = (head_token_span[0], head_token_span[1] + 2)
                first_entity_token_spans[1] = (tail_token_span[0] + 2, tail_token_span[1] + 4)
                token_span_with_special_token_ids = reversed(token_span_with_special_token_ids)
            else:
                first_entity_token_spans[0] = (head_token_span[0] + 2, head_token_span[1] + 4)
                first_entity_token_spans[1] = (tail_token_span[0], tail_token_span[1] + 2)

            for (entity_token_start, entity_token_end), special_token_id in token_span_with_special_token_ids:
                first_ids = first_ids[:entity_token_end] + [special_token_id] + first_ids[entity_token_end:]
                first_ids = first_ids[:entity_token_start] + [special_token_id] + first_ids[entity_token_start:]

        elif self.task == "entity_span_classification":
            if not (isinstance(entity_spans, list) and len(entity_spans) > 0 and isinstance(entity_spans[0], tuple)):
                raise ValueError(
                    "Entity spans should be provided as a list of tuples, "
                    "each tuple containing the start and end character indices of an entity"
                )

            first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(text, entity_spans)
            first_entity_ids = [self.entity_mask_token_id] * len(entity_spans)

        else:
            raise ValueError(f"Task {self.task} not supported")

        return (
            first_ids,
            second_ids,
            first_entity_ids,
            second_entity_ids,
            first_entity_token_spans,
            second_entity_token_spans,
        )

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer._batch_prepare_for_model
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Tuple[List[int], None]],
        batch_entity_ids_pairs: List[Tuple[Optional[List[int]], Optional[List[int]]]],
        batch_entity_token_spans_pairs: List[Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens


        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
            batch_entity_ids_pairs: list of entity ids or entity ids pairs
            batch_entity_token_spans_pairs: list of entity spans or entity spans pairs
            max_entity_length: The maximum length of the entity sequence.
        """

        batch_outputs = {}
        for input_ids, entity_ids, entity_token_span_pairs in zip(
            batch_ids_pairs, batch_entity_ids_pairs, batch_entity_token_spans_pairs
        ):
            first_ids, second_ids = input_ids
            first_entity_ids, second_entity_ids = entity_ids
            first_entity_token_spans, second_entity_token_spans = entity_token_span_pairs
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                entity_ids=first_entity_ids,
                pair_entity_ids=second_entity_ids,
                entity_token_spans=first_entity_token_spans,
                pair_entity_token_spans=second_entity_token_spans,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                max_entity_length=max_entity_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                padding_side=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer.prepare_for_model
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        entity_ids: Optional[List[int]] = None,
        pair_entity_ids: Optional[List[int]] = None,
        entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        pair_entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
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
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, entity id and entity span, or a pair of sequences of inputs ids, entity ids,
        entity spans so that it can be used by the model. It adds special tokens, truncates sequences if overflowing
        while taking into account the special tokens and manages a moving window (with user defined stride) for
        overflowing tokens. Please Note, for *pair_ids* different than `None` and *truncation_strategy = longest_first*
        or `True`, it is not possible to return overflowing tokens. Such a combination of arguments will raise an
        error.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence.
            entity_ids (`List[int]`, *optional*):
                Entity ids of the first sequence.
            pair_entity_ids (`List[int]`, *optional*):
                Entity ids of the second sequence.
            entity_token_spans (`List[Tuple[int, int]]`, *optional*):
                Entity spans of the first sequence.
            pair_entity_token_spans (`List[Tuple[int, int]]`, *optional*):
                Entity spans of the second sequence.
            max_entity_length (`int`, *optional*):
                The maximum length of the entity sequence.
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # Compute lengths
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )
        if (
            return_overflowing_tokens
            and truncation_strategy == TruncationStrategy.LONGEST_FIRST
            and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned word encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length and max_entity_length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            # truncate words up to max_length
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
            entity_token_offset = 1  # 1 * <s> token
            pair_entity_token_offset = len(ids) + 3  # 1 * <s> token & 2 * <sep> tokens
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])
            entity_token_offset = 0
            pair_entity_token_offset = len(ids)

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Set max entity length
        if not max_entity_length:
            max_entity_length = self.max_entity_length

        if entity_ids is not None:
            total_entity_len = 0
            num_invalid_entities = 0
            valid_entity_ids = [ent_id for ent_id, span in zip(entity_ids, entity_token_spans) if span[1] <= len(ids)]
            valid_entity_token_spans = [span for span in entity_token_spans if span[1] <= len(ids)]

            total_entity_len += len(valid_entity_ids)
            num_invalid_entities += len(entity_ids) - len(valid_entity_ids)

            valid_pair_entity_ids, valid_pair_entity_token_spans = None, None
            if pair_entity_ids is not None:
                valid_pair_entity_ids = [
                    ent_id
                    for ent_id, span in zip(pair_entity_ids, pair_entity_token_spans)
                    if span[1] <= len(pair_ids)
                ]
                valid_pair_entity_token_spans = [span for span in pair_entity_token_spans if span[1] <= len(pair_ids)]
                total_entity_len += len(valid_pair_entity_ids)
                num_invalid_entities += len(pair_entity_ids) - len(valid_pair_entity_ids)

            if num_invalid_entities != 0:
                logger.warning(
                    f"{num_invalid_entities} entities are ignored because their entity spans are invalid due to the"
                    " truncation of input tokens"
                )

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and total_entity_len > max_entity_length:
                # truncate entities up to max_entity_length
                valid_entity_ids, valid_pair_entity_ids, overflowing_entities = self.truncate_sequences(
                    valid_entity_ids,
                    pair_ids=valid_pair_entity_ids,
                    num_tokens_to_remove=total_entity_len - max_entity_length,
                    truncation_strategy=truncation_strategy,
                    stride=stride,
                )
                valid_entity_token_spans = valid_entity_token_spans[: len(valid_entity_ids)]
                if valid_pair_entity_token_spans is not None:
                    valid_pair_entity_token_spans = valid_pair_entity_token_spans[: len(valid_pair_entity_ids)]

            if return_overflowing_tokens:
                encoded_inputs["overflowing_entities"] = overflowing_entities
                encoded_inputs["num_truncated_entities"] = total_entity_len - max_entity_length

            final_entity_ids = valid_entity_ids + valid_pair_entity_ids if valid_pair_entity_ids else valid_entity_ids
            encoded_inputs["entity_ids"] = list(final_entity_ids)
            entity_position_ids = []
            entity_start_positions = []
            entity_end_positions = []
            for token_spans, offset in (
                (valid_entity_token_spans, entity_token_offset),
                (valid_pair_entity_token_spans, pair_entity_token_offset),
            ):
                if token_spans is not None:
                    for start, end in token_spans:
                        start += offset
                        end += offset
                        position_ids = list(range(start, end))[: self.max_mention_length]
                        position_ids += [-1] * (self.max_mention_length - end + start)
                        entity_position_ids.append(position_ids)
                        entity_start_positions.append(start)
                        entity_end_positions.append(end - 1)

            encoded_inputs["entity_position_ids"] = entity_position_ids
            if self.task == "entity_span_classification":
                encoded_inputs["entity_start_positions"] = entity_start_positions
                encoded_inputs["entity_end_positions"] = entity_end_positions

            if return_token_type_ids:
                encoded_inputs["entity_token_type_ids"] = [0] * len(encoded_inputs["entity_ids"])

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                max_entity_length=max_entity_length,
                padding=padding_strategy.value,
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

    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer.pad
    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch. Padding side (left/right) padding token ids are defined at the tokenizer level (with
        `self.padding_side`, `self.pad_token_id` and `self.pad_token_type_id`) .. note:: If the `encoded_inputs` passed
        are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the result will use the same type unless
        you provide a different tensor type with `return_tensors`. In the case of PyTorch tensors, you will lose the
        specific device of your tensors however.

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function. Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or
                TensorFlow tensors), see the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            max_entity_length (`int`, *optional*):
                The maximum length of the entity sequence.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            padding_side:
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute. [What are attention
                masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_tensor(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        if max_entity_length is None:
            max_entity_length = self.max_entity_length

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                max_entity_length=max_entity_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        if any(len(v) != batch_size for v in encoded_inputs.values()):
            raise ValueError("Some items in the output dictionary have a different batch size than others.")

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            max_entity_length = (
                max(len(inputs) for inputs in encoded_inputs["entity_ids"]) if "entity_ids" in encoded_inputs else 0
            )
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items()}
            outputs = self._pad(
                inputs,
                max_length=max_length,
                max_entity_length=max_entity_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    # Copied from transformers.models.luke.tokenization_luke.LukeTokenizer._pad
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
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
            max_entity_length: The maximum length of the entity sequence.
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
        entities_provided = bool("entity_ids" in encoded_inputs)

        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(encoded_inputs["input_ids"])
            if entities_provided:
                max_entity_length = len(encoded_inputs["entity_ids"])

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        if (
            entities_provided
            and max_entity_length is not None
            and pad_to_multiple_of is not None
            and (max_entity_length % pad_to_multiple_of != 0)
        ):
            max_entity_length = ((max_entity_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and (
            len(encoded_inputs["input_ids"]) != max_length
            or (entities_provided and len(encoded_inputs["entity_ids"]) != max_entity_length)
        )

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])
        if entities_provided and return_attention_mask and "entity_attention_mask" not in encoded_inputs:
            encoded_inputs["entity_attention_mask"] = [1] * len(encoded_inputs["entity_ids"])

        if needs_to_be_padded:
            difference = max_length - len(encoded_inputs["input_ids"])
            padding_side = padding_side if padding_side is not None else self.padding_side
            if entities_provided:
                entity_difference = max_entity_length - len(encoded_inputs["entity_ids"])
            if padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                    if entities_provided:
                        encoded_inputs["entity_attention_mask"] = (
                            encoded_inputs["entity_attention_mask"] + [0] * entity_difference
                        )
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"] + [0] * difference
                    if entities_provided:
                        encoded_inputs["entity_token_type_ids"] = (
                            encoded_inputs["entity_token_type_ids"] + [0] * entity_difference
                        )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
                if entities_provided:
                    encoded_inputs["entity_ids"] = (
                        encoded_inputs["entity_ids"] + [self.entity_pad_token_id] * entity_difference
                    )
                    encoded_inputs["entity_position_ids"] = (
                        encoded_inputs["entity_position_ids"] + [[-1] * self.max_mention_length] * entity_difference
                    )
                    if self.task == "entity_span_classification":
                        encoded_inputs["entity_start_positions"] = (
                            encoded_inputs["entity_start_positions"] + [0] * entity_difference
                        )
                        encoded_inputs["entity_end_positions"] = (
                            encoded_inputs["entity_end_positions"] + [0] * entity_difference
                        )

            elif padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                    if entities_provided:
                        encoded_inputs["entity_attention_mask"] = [0] * entity_difference + encoded_inputs[
                            "entity_attention_mask"
                        ]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [0] * difference + encoded_inputs["token_type_ids"]
                    if entities_provided:
                        encoded_inputs["entity_token_type_ids"] = [0] * entity_difference + encoded_inputs[
                            "entity_token_type_ids"
                        ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + encoded_inputs["input_ids"]
                if entities_provided:
                    encoded_inputs["entity_ids"] = [self.entity_pad_token_id] * entity_difference + encoded_inputs[
                        "entity_ids"
                    ]
                    encoded_inputs["entity_position_ids"] = [
                        [-1] * self.max_mention_length
                    ] * entity_difference + encoded_inputs["entity_position_ids"]
                    if self.task == "entity_span_classification":
                        encoded_inputs["entity_start_positions"] = [0] * entity_difference + encoded_inputs[
                            "entity_start_positions"
                        ]
                        encoded_inputs["entity_end_positions"] = [0] * entity_difference + encoded_inputs[
                            "entity_end_positions"
                        ]
            else:
                raise ValueError("Invalid padding strategy:" + str(padding_side))

        return encoded_inputs

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
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

        entity_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["entity_vocab_file"]
        )

        with open(entity_vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.entity_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return out_vocab_file, entity_vocab_file

    # Copied from transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLM-RoBERTa sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

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
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    # Copied from transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer.get_special_tokens_mask
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

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    # Copied from transformers.models.xlm_roberta.tokenization_xlm_roberta.XLMRobertaTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
        not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.

        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]


__all__ = ["MLukeTokenizer"]

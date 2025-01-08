# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""
Fast tokenization class for LayoutLMv3. It overwrites 2 methods of the slow tokenizer class, namely _batch_encode_plus
and _encode_plus, in which the Rust tokenizer is used.
"""

import json
from typing import Dict, List, Optional, Tuple, Union

from tokenizers import pre_tokenizers, processors

from ...tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    PreTokenizedInput,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import add_end_docstrings, logging
from .tokenization_layoutlmv3 import (
    LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING,
    LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    LayoutLMv3Tokenizer,
)


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class LayoutLMv3TokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" LayoutLMv3 tokenizer (backed by HuggingFace's *tokenizers* library). Based on BPE.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
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
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
        cls_token_box (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`):
            The bounding box to use for the special [CLS] token.
        sep_token_box (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`):
            The bounding box to use for the special [SEP] token.
        pad_token_box (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`):
            The bounding box to use for the special [PAD] token.
        pad_token_label (`int`, *optional*, defaults to -100):
            The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch's
            CrossEntropyLoss.
        only_label_first_subword (`bool`, *optional*, defaults to `True`):
            Whether or not to only label the first subword, in case word labels are provided.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = LayoutLMv3Tokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=True,
        trim_offsets=True,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[0, 0, 0, 0],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            cls_token_box=cls_token_box,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            **kwargs,
        )

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            # The lists 'sep' and 'cls' must be cased in tuples for the object `post_processor_class`
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            changes_to_apply = False

            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)

        # additional properties
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # Copied from transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.__call__
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
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
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences with word-level normalized bounding boxes and optional labels.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
                (words of a single example or questions of a batch of examples) or a list of list of strings (batch of
                words).
            text_pair (`List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
                (pretokenized string).
            boxes (`List[List[int]]`, `List[List[List[int]]]`):
                Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.
            word_labels (`List[int]`, `List[List[int]]`, *optional*):
                Word-level integer labels (for token classification tasks such as FUNSD, CORD).
        """

        # Input type checking for clearer error
        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return True
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False

        if text_pair is not None:
            # in case text + text_pair are provided, text = questions, text_pair = words
            if not _is_valid_text_input(text):
                raise ValueError("text input must of type `str` (single example) or `List[str]` (batch of examples). ")
            if not isinstance(text_pair, (list, tuple)):
                raise ValueError(
                    "Words must be of type `List[str]` (single pretokenized example), "
                    "or `List[List[str]]` (batch of pretokenized examples)."
                )
        else:
            # in case only text is provided => must be words
            if not isinstance(text, (list, tuple)):
                raise ValueError(
                    "Words must be of type `List[str]` (single pretokenized example), "
                    "or `List[List[str]]` (batch of pretokenized examples)."
                )

        if text_pair is not None:
            is_batched = isinstance(text, (list, tuple))
        else:
            is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))

        words = text if text_pair is None else text_pair
        if boxes is None:
            raise ValueError("You must provide corresponding bounding boxes")
        if is_batched:
            if len(words) != len(boxes):
                raise ValueError("You must provide words and boxes for an equal amount of examples")
            for words_example, boxes_example in zip(words, boxes):
                if len(words_example) != len(boxes_example):
                    raise ValueError("You must provide as many words as there are bounding boxes")
        else:
            if len(words) != len(boxes):
                raise ValueError("You must provide as many words as there are bounding boxes")

        if is_batched:
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    f"batch length of `text`: {len(text)} does not match batch length of `text_pair`:"
                    f" {len(text_pair)}."
                )
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            is_pair = bool(text_pair is not None)
            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                is_pair=is_pair,
                boxes=boxes,
                word_labels=word_labels,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
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
                boxes=boxes,
                word_labels=word_labels,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
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

    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # Copied from transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.batch_encode_plus
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,
        boxes: Optional[List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
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
        **kwargs,
    ) -> BatchEncoding:
        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
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

    # Copied from transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.tokenize
    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        batched_input = [(text, pair)] if pair else [text]
        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )

        return encodings[0].tokens

    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # Copied from transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.encode_plus
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
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
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences. .. warning:: This method is deprecated,
        `__call__` should be used instead.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.
            text_pair (`List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a list of strings (words of a single example) or a
                list of list of strings (words of a batch of examples).
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

        return self._encode_plus(
            text=text,
            boxes=boxes,
            text_pair=text_pair,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
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

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,
        boxes: Optional[List[List[List[int]]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        if not isinstance(batch_text_or_text_pairs, list):
            raise TypeError(f"batch_text_or_text_pairs has to be a list (got {type(batch_text_or_text_pairs)})")

        # Set the truncation and padding strategy and restore the initial configuration
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
        )

        if is_pair:
            batch_text_or_text_pairs = [(text.split(), text_pair) for text, text_pair in batch_text_or_text_pairs]

        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=True,  # we set this to True as LayoutLMv3 always expects pretokenized inputs
        )

        # Convert encoding to dict
        # `Tokens` has type: Tuple[
        #                       List[Dict[str, List[List[int]]]] or List[Dict[str, 2D-Tensor]],
        #                       List[EncodingFast]
        #                    ]
        # with nested dimensions corresponding to batch, overflows, sequence length
        tokens_and_encodings = [
            self._convert_encoding(
                encoding=encoding,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=True
                if word_labels is not None
                else return_offsets_mapping,  # we use offsets to create the labels
                return_length=return_length,
                verbose=verbose,
            )
            for encoding in encodings
        ]

        # Convert the output to have dict[list] from list[dict] and remove the additional overflows dimension
        # From (variable) shape (batch, overflows, sequence length) to ~ (batch * overflows, sequence length)
        # (we say ~ because the number of overflow varies with the example in the batch)
        #
        # To match each overflowing sample with the original sample in the batch
        # we add an overflow_to_sample_mapping array (see below)
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0].keys():
            stack = [e for item, _ in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
        sanitized_encodings = [e for _, item in tokens_and_encodings for e in item]

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for i, (toks, _) in enumerate(tokens_and_encodings):
                overflow_to_sample_mapping += [i] * len(toks["input_ids"])
            sanitized_tokens["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        for input_ids in sanitized_tokens["input_ids"]:
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)

        # create the token boxes
        token_boxes = []
        for batch_index in range(len(sanitized_tokens["input_ids"])):
            if return_overflowing_tokens:
                original_index = sanitized_tokens["overflow_to_sample_mapping"][batch_index]
            else:
                original_index = batch_index
            token_boxes_example = []
            for id, sequence_id, word_id in zip(
                sanitized_tokens["input_ids"][batch_index],
                sanitized_encodings[batch_index].sequence_ids,
                sanitized_encodings[batch_index].word_ids,
            ):
                if word_id is not None:
                    if is_pair and sequence_id == 0:
                        token_boxes_example.append(self.pad_token_box)
                    else:
                        token_boxes_example.append(boxes[original_index][word_id])
                else:
                    if id == self.cls_token_id:
                        token_boxes_example.append(self.cls_token_box)
                    elif id == self.sep_token_id:
                        token_boxes_example.append(self.sep_token_box)
                    elif id == self.pad_token_id:
                        token_boxes_example.append(self.pad_token_box)
                    else:
                        raise ValueError("Id not recognized")
            token_boxes.append(token_boxes_example)

        sanitized_tokens["bbox"] = token_boxes

        # optionally, create the labels
        if word_labels is not None:
            labels = []
            for batch_index in range(len(sanitized_tokens["input_ids"])):
                if return_overflowing_tokens:
                    original_index = sanitized_tokens["overflow_to_sample_mapping"][batch_index]
                else:
                    original_index = batch_index
                labels_example = []
                previous_token_empty = False
                for id, offset, word_id in zip(
                    sanitized_tokens["input_ids"][batch_index],
                    sanitized_tokens["offset_mapping"][batch_index],
                    sanitized_encodings[batch_index].word_ids,
                ):
                    if word_id is not None:
                        if self.only_label_first_subword:
                            if offset[0] == 0 and not previous_token_empty:
                                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                                labels_example.append(word_labels[original_index][word_id])
                            else:
                                labels_example.append(self.pad_token_label)
                            if offset == (0, 0):
                                previous_token_empty = True
                            else:
                                previous_token_empty = False
                        else:
                            labels_example.append(word_labels[original_index][word_id])
                    else:
                        labels_example.append(self.pad_token_label)
                labels.append(labels_example)

            sanitized_tokens["labels"] = labels
            # finally, remove offsets if the user didn't want them
            if not return_offsets_mapping:
                del sanitized_tokens["offset_mapping"]

        return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

    # Copied from transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast._encode_plus
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[bool] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # make it a batched input
        # 2 options:
        # 1) only text, in case text must be a list of str
        # 2) text + text_pair, in which case text = str and text_pair a list of str
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_boxes = [boxes]
        batched_word_labels = [word_labels] if word_labels is not None else None
        batched_output = self._batch_encode_plus(
            batched_input,
            is_pair=bool(text_pair is not None),
            boxes=batched_boxes,
            word_labels=batched_word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
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

        # Return tensor is None, then we can remove the leading batch axis
        # Overflowing tokens are returned as a batch of output so we keep them in this case
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        return batched_output

    # Copied from transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast._pad
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

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            padding_side = padding_side if padding_side is not None else self.padding_side
            if padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "bbox" in encoded_inputs:
                    encoded_inputs["bbox"] = encoded_inputs["bbox"] + [self.pad_token_box] * difference
                if "labels" in encoded_inputs:
                    encoded_inputs["labels"] = encoded_inputs["labels"] + [self.pad_token_label] * difference
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
            elif padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "bbox" in encoded_inputs:
                    encoded_inputs["bbox"] = [self.pad_token_box] * difference + encoded_inputs["bbox"]
                if "labels" in encoded_inputs:
                    encoded_inputs["labels"] = [self.pad_token_label] * difference + encoded_inputs["labels"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(padding_side))

        return encoded_inputs

    # Copied from transformers.models.layoutlmv2.tokenization_layoutlmv2_fast.LayoutLMv2TokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Args:
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not:
        make use of token type ids, therefore a list of zeros is returned.
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


__all__ = ["LayoutLMv3TokenizerFast"]

# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Tokenization classes for UDOP model."""

import os
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union

from ...tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, is_sentencepiece_available, logging


if is_sentencepiece_available():
    from .tokenization_udop import UdopTokenizer
else:
    UdopTokenizer = None


VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


logger = logging.get_logger(__name__)

UDOP_ENCODE_KWARGS_DOCSTRING = r"""
            add_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to encode the sequences with the special tokens relative to their model.
            padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*, defaults to `False`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_length`, the overflowing tokens returned when
                `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping tokens.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
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

            - **bbox** -- List of bounding boxes to be fed to a model.

            - **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
              if *"token_type_ids"* is in `self.model_input_names`).

              [What are token type IDs?](../glossary#token-type-ids)

            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

              [What are attention masks?](../glossary#attention-mask)

            - **labels** -- List of labels to be fed to a model. (when `word_labels` is specified).
            - **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
              `return_overflowing_tokens=True`).
            - **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
              `return_overflowing_tokens=True`).
            - **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
              regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
            - **length** -- The length of the inputs (when `return_length=True`).
"""


class UdopTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" UDOP tokenizer (backed by HuggingFace's *tokenizers* library). Adapted from
    [`LayoutXLMTokenizer`] and [`T5Tokenizer`]. Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.

        tokenizer_file (`str`, *optional*):
            Path to the tokenizer file.
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
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        sep_token_box (`List[int]`, *optional*, defaults to `[1000, 1000, 1000, 1000]`):
            The bounding box to use for the special [SEP] token.
        pad_token_box (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`):
            The bounding box to use for the special [PAD] token.
        pad_token_label (`int`, *optional*, defaults to -100):
            The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch's
            CrossEntropyLoss.
        only_label_first_subword (`bool`, *optional*, defaults to `True`):
            Whether or not to only label the first subword, in case word labels are provided.
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = UdopTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        eos_token="</s>",
        sep_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        additional_special_tokens=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file

        # additional properties
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    @add_end_docstrings(UDOP_ENCODE_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        **kwargs,
    ) -> BatchEncoding:
        if text is None and text_target is None:
            raise ValueError("You need to specify either `text` or `text_target`.")
        if text is not None:
            # The context manager will send the inputs as normal texts and not text_target, but we shouldn't change the
            # input mode in this case.
            if not self._in_target_context_manager:
                self._switch_to_input_mode()
            encodings = self.call_boxes(text=text, text_pair=text_pair, boxes=boxes, word_labels=word_labels, **kwargs)
        if text_target is not None:
            self._switch_to_target_mode()
            target_encodings = self._call_one(text=text_target, text_pair=text_pair_target, **kwargs)
        # Leave back tokenizer in input mode
        self._switch_to_input_mode()

        if text_target is None:
            return encodings
        elif text is None:
            return target_encodings
        else:
            encodings["labels"] = target_encodings["input_ids"]
            return encodings

    @add_end_docstrings(UDOP_ENCODE_KWARGS_DOCSTRING)
    def call_boxes(
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
                    "words must of type `List[str]` (single pretokenized example), "
                    "or `List[List[str]]` (batch of pretokenized examples)."
                )
        else:
            # in case only text is provided => must be words
            if not isinstance(text, (list, tuple)):
                raise ValueError(
                    "Words must of type `List[str]` (single pretokenized example), "
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
            return self.batch_encode_plus_boxes(
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
            return self.encode_plus_boxes(
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

    # Copied from transformers.models.layoutxlm.tokenization_layoutxlm_fast.LayoutXLMTokenizerFast.tokenize
    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        batched_input = [(text, pair)] if pair else [text]

        self._tokenizer.encode_special_tokens = kwargs.pop(
            "split_special_tokens", self._tokenizer.encode_special_tokens
        )

        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )

        return encodings[0].tokens

    def batch_encode_plus_boxes(
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
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
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
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            batch_text_or_text_pairs (`List[str]`, `List[Tuple[str, str]]`, `List[List[str]]`, `List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also `List[List[int]]`, `List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
                details in `encode_plus`).
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

        return self._batch_encode_plus_boxes(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
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

    def _batch_encode_plus_boxes(
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
        **kwargs,
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
            is_pretokenized=True,  # we set this to True as LayoutLMv2 always expects pretokenized inputs
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
                    if id == self.sep_token_id:
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
                        else:
                            labels_example.append(word_labels[original_index][word_id])
                        if self.decode(id) == "":
                            previous_token_empty = True
                        else:
                            previous_token_empty = False
                    else:
                        labels_example.append(self.pad_token_label)
                labels.append(labels_example)

            sanitized_tokens["labels"] = labels
            # finally, remove offsets if the user didn't want them
            if not return_offsets_mapping:
                del sanitized_tokens["offset_mapping"]

        return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

    def _encode_plus_boxes(
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
        batched_output = self._batch_encode_plus_boxes(
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

    def encode_boxes(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[int]:
        """
        Args:
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary. Same as doing
        `self.convert_tokens_to_ids(self.tokenize(text))`.
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
        encoded_inputs = self.encode_plus_boxes(
            text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]

    def encode_plus_boxes(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
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
        Tokenize and prepare for the model a sequence or a pair of sequences.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            text (`str`, `List[str]` or (for non-fast tokenizers) `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
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

        return self._encode_plus_boxes(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
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

    # Copied from transformers.models.layoutxlm.tokenization_layoutxlm_fast.LayoutXLMTokenizerFast._pad
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
            padding_side (`str`, *optional*):
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
            return token_ids_0 + [self.sep_token_id]
        sep = [self.sep_token_id]
        return token_ids_0 + sep + token_ids_1 + sep

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

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        return len(token_ids_0 + sep + token_ids_1 + sep) * [0]

    # Copied from transformers.models.layoutxlm.tokenization_layoutxlm_fast.LayoutXLMTokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)


__all__ = ["UdopTokenizerFast"]

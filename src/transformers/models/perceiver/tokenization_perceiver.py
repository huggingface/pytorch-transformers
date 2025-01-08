# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Tokenization class for Perceiver."""

from typing import Dict, List, Optional, Tuple

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


class PerceiverTokenizer(PreTrainedTokenizer):
    """
    Construct a Perceiver tokenizer. The Perceiver simply uses raw bytes utf-8 encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        bos_token (`str`, *optional*, defaults to `"[BOS]"`):
            The BOS token (reserved in the vocab, but not actually used).
        eos_token (`str`, *optional*, defaults to `"[EOS]"`):
            The end of sequence token (reserved in the vocab, but not actually used).

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The MASK token, useful for masked language modeling.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The CLS token (reserved in the vocab, but not actually used).
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from two sequences.

    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=2048,
        **kwargs,
    ) -> None:
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        mask_token = AddedToken(mask_token, lstrip=False, rstrip=False) if isinstance(mask_token, str) else mask_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token

        self._utf_vocab_size = 2**8  # utf is 8 bits

        # Since these tokens are not part of the vocabulary, we manually add them
        self._added_tokens_decoder: Dict[str, int] = {
            0: pad_token,
            1: bos_token,
            2: eos_token,
            3: mask_token,
            4: cls_token,
            5: sep_token,
        }
        self._num_special_tokens = len(self._added_tokens_decoder)
        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sep_token=sep_token,
            model_max_length=model_max_length,
            **kwargs,
        )

    def get_vocab(self) -> Dict[str, int]:
        vocab = {}
        for i in range(self._utf_vocab_size):
            token = chr(i)
            vocab[token] = i + self._num_special_tokens
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def vocab_size(self):
        return self._utf_vocab_size

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

        # normal case: some special tokens
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks. A sequence has the
        following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

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
        else:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if len(token) != 1:
            token_id = self.unk_token_id
        else:
            token_id = ord(token) + self._num_special_tokens
        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = chr(index - self._num_special_tokens)
        return token

    # TODO @ArthurZ refactor this as well....
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        bstring = b""
        for token in tokens:
            if token in self.added_tokens_encoder:
                tok_string = str(token).encode("utf-8")
            else:
                tok_string = bytes([ord(token)])
            bstring += tok_string
        string = bstring.decode("utf-8", errors="replace")
        return string

    # PerceiverTokenizer has no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()


__all__ = ["PerceiverTokenizer"]

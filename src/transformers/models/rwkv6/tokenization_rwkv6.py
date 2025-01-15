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
# limitations under the License.
"""Tokenization classes for RWKV6."""

import os
from typing import TYPE_CHECKING, List, Optional, Tuple

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "rwkv_vocab_v20230424.txt",
}


class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to: list
    values: set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while fr is not None:
            if fr.ch is not None:
                ret.append(fr.ch)
            fr = fr.front
        return f"<TRIE {ret[::-1]} {self.values}>"

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u: TRIE = self
        ch: int = key[idx]
        ret = None

        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = idx, u, u.values
            if idx == len(key):
                break
            ch = key[idx]

        if ret is None:
            raise ValueError("No match found")
        return ret


class RwkvTokenizer:
    def __init__(self, file_name):
        self.idx2token = {}
        sorted_token = []  # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)

            assert len(x) == int(l[l.rindex(" ") :])
            sorted_token += [x]
            self.idx2token[idx] = x

        self.idx2token[0] = b"<|endoftext|>"  # use a special token
        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encode_bytes(self, src: bytes):
        idx: int = 0
        tokens = []
        while idx < len(src):
            _idx: int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert idx != _idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decode_bytes(self, tokens):
        return b"".join(self.idx2token[i] for i in tokens)

    def encode(self, src):
        if isinstance(src, str):
            return [self.encode_bytes(src.encode("utf-8"))]
        elif isinstance(src, list):
            return [self.encode_bytes(s.encode("utf-8")) for s in src]

    def decode(self, tokens):
        return [self.decode_bytes(batch).decode("utf-8", errors="replace") for batch in tokens]

    def print_tokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode("utf-8")
            except BaseException:
                pass
            print(f"{repr(s)}{i}", end=" ")
        print()


class Rwkv6Tokenizer(PreTrainedTokenizer):
    r"""
    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        bos_token (:obj:`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining.
            Can be used as sequence classifier token.
        eos_token (:obj:`str`, *optional*, defaults to `"<s>"`):
            The end of sequence token.
        unk_token (:obj:`str`, *optional*, defaults to `"<s>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.

    Example::
        >>> from transformers import Rwkv6Tokenizer
        >>> tokenizer = Rwkv6Tokenizer(
        ...     vocab_file='vocab.txt',
        ...     bos_token='<s>',
        ...     eos_token='</s>',
        ...     unk_token='<unk>'
        ... )
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, bos_token="<s>", eos_token="<s>", unk_token="<s>", **kwargs):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        if "add_bos_token" in kwargs:
            self.add_bos_token = kwargs["add_bos_token"]
        else:
            self.add_bos_token = False
        self.trie_tokenizer = RwkvTokenizer(vocab_file)
        vocab = self.trie_tokenizer.token2idx
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self._added_tokens_decoder = {0: AddedToken(str(bos_token))}
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        vocab = {str(self.convert_ids_to_tokens(i)): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, split_special_tokens=False):
        # return self.wordpiece_tokenizer.tokenize(text.encode("utf-8"))
        return self.trie_tokenizer.encode(text)[0]

    def _convert_token_to_id(self, token):
        return token

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (byte) using the vocab."""
        token = self.decoder.get(index, self.unk_token)
        if isinstance(token, (bytes)):
            token = token.decode("utf-8", errors="replace")
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (bytes) in a single string. Additional tokens are encoded to bytes"""
        out_string = b"".join([k.encode(errors="replace") if isinstance(k, str) else k for k in tokens]).decode(
            "utf-8"
        )
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + "vocab.txt",
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.encoder.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(str(token) + "\n")
                index += 1
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
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
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=False,
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

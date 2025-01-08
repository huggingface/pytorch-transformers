# coding=utf-8
# Copyright 2022 ABEJA, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for GPTNeoXJapanese."""

import collections
import json
import os
import re
from typing import Optional, Tuple

import numpy as np

from ...tokenization_utils_fast import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "emoji_file": "emoji.json"}


def load_vocab_and_emoji(vocab_file, emoji_file):
    """Loads a vocabulary file and emoji file into a dictionary."""
    with open(emoji_file, "r", encoding="utf-8") as f:
        emoji = json.loads(f.read())

    vocab = collections.OrderedDict()
    raw_vocab = collections.OrderedDict()
    ids_to_tokens = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as f:
        token = f.readlines()
    token = [[t.rstrip("\n")] if (t == "," or "," not in t) else t.rstrip("\n").split(",") for t in token]
    for idx, b in enumerate(token):
        ids_to_tokens[idx] = b
        raw_vocab[",".join(b)] = idx
        for wd in b:
            vocab[wd] = idx

    return vocab, raw_vocab, ids_to_tokens, emoji


class GPTNeoXJapaneseTokenizer(PreTrainedTokenizer):
    """
    This tokenizer inherits from [`PreTrainedTokenizer`] and is based on Japanese special Sub-Word-Encoding that is
    used in this repository (https://github.com/tanreinama/Japanese-BPEEncoder_V2). Check the repository for details.
    Japanese has a relatively large vocabulary and there is no separation between words. Furthermore, the language is a
    combination of hiragana, katakana, and kanji, and variants such as "1" and "①" are often used. In order to cope
    with these, this tokenizer has the following features
    - Subword-by-subword segmentation, which is intermediate between byte strings and morphological analysis.
    - BPEs are created for each Kanji, Hiragana, and Katakana character, and there are no BPEs that cross character
        types, such as Kanji + Hiragana or Hiragana + Katakana.
    - All-byte encoding that does not require <unk>.
    - Independent of UTF codes such as 2-byte and 3-byte characters
    - Conversion of heterographs to the same token_id
    - Emoji and Emoticon are grouped into 12 types as special tags.

    Example:

    ```python
    >>> from transformers import GPTNeoXJapaneseTokenizer

    >>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
    >>> # You can confirm both 慶応 and 慶應 are encoded to 17749
    >>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
    [30014, 26883, 26638, 27228, 25, 26650, 31732, 31679, 27809, 26638, 17749, 31592, 17749, 31593, 321, 1281]

    >>> # Both 慶応 and 慶應 are decoded to 慶応
    >>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
    '吾輩は猫である🐯。実は慶応(慶応)大学出身'
    ```

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        emoji_file (`str`):
            File containing the emoji.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        do_clean_text (`bool`, *optional*, defaults to `False`):
            Whether or not to clean text for URL, EMAIL, TEL, Japanese DATE and Japanese PRICE.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        emoji_file,
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        do_clean_text=False,
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = GPTNeoXJapaneseokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        if not os.path.isfile(emoji_file):
            raise ValueError(
                f"Can't find a emoji file at path '{emoji_file}'. To load the emoji information from a Google"
                " pretrained model use `tokenizer = GPTNeoXJapaneseokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.do_clean_text = do_clean_text
        self.vocab, self.raw_vocab, self.ids_to_tokens, self.emoji = load_vocab_and_emoji(vocab_file, emoji_file)
        self.subword_tokenizer = SubWordJapaneseTokenizer(
            vocab=self.vocab, ids_to_tokens=self.ids_to_tokens, emoji=self.emoji
        )
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            do_clean_text=do_clean_text,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # self.vocab contains support for character fluctuation unique to Japanese, and has a large number of vocab
        return len(self.raw_vocab)

    def get_vocab(self):
        return dict(self.raw_vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        return self.subword_tokenizer.tokenize(text, clean=self.do_clean_text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.subword_tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = "".join(tokens).strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
            emoji_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["emoji_file"]
            )
        else:
            vocab_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["vocab_file"]
            )
            emoji_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["emoji_file"]
            )
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token_index, token in self.ids_to_tokens.items():
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(",".join(token) + "\n")
                index += 1
        with open(emoji_file, "w", encoding="utf-8") as writer:
            json.dump(self.emoji, writer)
        return vocab_file, emoji_file


class SubWordJapaneseTokenizer:
    """
    https://github.com/tanreinama/Japanese-BPEEncoder_V2 This tokenizer class is under MIT Lisence according to the
    original repository.

    MIT License

    Copyright (c) 2020 tanreinama

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, vocab, ids_to_tokens, emoji):
        self.vocab = vocab  # same as swe
        self.ids_to_tokens = ids_to_tokens  # same as bpe
        self.emoji = emoji
        self.maxlen = np.max([len(w) for w in self.vocab.keys()])
        self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.content_repatter3 = re.compile(r"[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}")
        self.content_repatter4 = re.compile(
            r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )
        self.content_repatter5 = re.compile(
            r"(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|\u32ff)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )
        self.content_repatter6 = re.compile(
            r"((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*"
        )
        keisen = "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
        blocks = "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟"
        self.content_trans1 = str.maketrans({k: "<BLOCK>" for k in keisen + blocks})

    def __len__(self):
        return len(self.ids_to_tokens)

    def clean_text(self, content):
        content = self.content_repatter1.sub("<URL>", content)
        content = self.content_repatter2.sub("<EMAIL>", content)
        content = self.content_repatter3.sub("<TEL>", content)
        content = self.content_repatter4.sub("<DATE>", content)
        content = self.content_repatter5.sub("<DATE>", content)
        content = self.content_repatter6.sub("<PRICE>", content)
        content = content.translate(self.content_trans1)
        while "<BLOCK><BLOCK>" in content:
            content = content.replace("<BLOCK><BLOCK>", "<BLOCK>")
        return content

    def tokenize(self, text, clean=False):
        text = text.replace(" ", "<SP>")
        text = text.replace("　", "<SP>")
        text = text.replace("\r\n", "<BR>")
        text = text.replace("\n", "<BR>")
        text = text.replace("\r", "<BR>")
        text = text.replace("\t", "<TAB>")
        text = text.replace("—", "ー")
        text = text.replace("−", "ー")
        for k, v in self.emoji["emoji"].items():
            if k in text:
                text = text.replace(k, v)
        if clean:
            text = self.clean_text(text)

        def check_simbol(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 2:
                c = (int(e[0]) << 8) + int(e[1])
                if (
                    (c >= 0xC2A1 and c <= 0xC2BF)
                    or (c >= 0xC780 and c <= 0xC783)
                    or (c >= 0xCAB9 and c <= 0xCBBF)
                    or (c >= 0xCC80 and c <= 0xCDA2)
                ):
                    return True
            return False

        def checku2e(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 3:
                c = (int(e[0]) << 16) + (int(e[1]) << 8) + int(e[2])
                if c >= 0xE28080 and c <= 0xE2B07F:
                    return True
            return False

        pos = 0
        result = []
        while pos < len(text):
            end = min(len(text), pos + self.maxlen + 1) if text[pos] == "<" else pos + 3
            candidates = []  # (token_id, token, pos)
            for e in range(end, pos, -1):
                wd = text[pos:e]
                if wd in self.vocab:
                    if wd[0] == "<" and len(wd) > 2:
                        candidates = [(self.vocab[wd], wd, e)]
                        break
                    else:
                        candidates.append((self.vocab[wd], wd, e))
            if len(candidates) > 0:
                # the smallest token_id is adopted
                _, wd, e = sorted(candidates, key=lambda x: x[0])[0]
                result.append(wd)
                pos = e
            else:
                end = pos + 1
                wd = text[pos:end]
                if check_simbol(wd):
                    result.append("<KIGOU>")
                elif checku2e(wd):
                    result.append("<U2000U2BFF>")
                else:
                    for i in wd.encode("utf-8"):
                        result.append("<|byte%d|>" % i)
                pos = end
        return result

    def convert_id_to_token(self, index, breakline="\n"):
        words = []
        byte_tokens = []
        word = self.ids_to_tokens[index][0]
        if word[:6] == "<|byte" and word[-2:] == "|>":
            byte_tokens.append(int(word[6:-2]))
        else:
            if len(byte_tokens) > 0:
                words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
                byte_tokens = []
            if word[:7] == "<|emoji" and word[-2:] == "|>":
                words.append(self.emoji["emoji_inv"][word])
            elif word == "<SP>":
                words.append(" ")
            elif word == "<BR>":
                words.append(breakline)
            elif word == "<TAB>":
                words.append("\t")
            elif word == "<BLOCK>":
                words.append("▀")
            elif word == "<KIGOU>":
                words.append("ǀ")
            elif word == "<U2000U2BFF>":
                words.append("‖")
            else:
                words.append(word)
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
        text = "".join(words)
        return text


__all__ = ["GPTNeoXJapaneseTokenizer"]

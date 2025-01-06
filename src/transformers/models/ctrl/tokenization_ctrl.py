# coding=utf-8
# Copyright 2018 Salesforce and The HuggingFace Inc. team.
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
"""Tokenization classes for Salesforce CTRL."""

import json
import os
from typing import Optional, Tuple

import regex as re

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


CONTROL_CODES = {
    "Pregnancy": 168629,
    "Christianity": 7675,
    "Explain": 106423,
    "Fitness": 63440,
    "Saving": 63163,
    "Ask": 27171,
    "Ass": 95985,
    "Joke": 163509,
    "Questions": 45622,
    "Thoughts": 49605,
    "Retail": 52342,
    "Feminism": 164338,
    "Writing": 11992,
    "Atheism": 192263,
    "Netflix": 48616,
    "Computing": 39639,
    "Opinion": 43213,
    "Alone": 44967,
    "Funny": 58917,
    "Gaming": 40358,
    "Human": 4088,
    "India": 1331,
    "Joker": 77138,
    "Diet": 36206,
    "Legal": 11859,
    "Norman": 4939,
    "Tip": 72689,
    "Weight": 52343,
    "Movies": 46273,
    "Running": 23425,
    "Science": 2090,
    "Horror": 37793,
    "Confession": 60572,
    "Finance": 12250,
    "Politics": 16360,
    "Scary": 191985,
    "Support": 12654,
    "Technologies": 32516,
    "Teenage": 66160,
    "Event": 32769,
    "Learned": 67460,
    "Notion": 182770,
    "Wikipedia": 37583,
    "Books": 6665,
    "Extract": 76050,
    "Confessions": 102701,
    "Conspiracy": 75932,
    "Links": 63674,
    "Narcissus": 150425,
    "Relationship": 54766,
    "Relationships": 134796,
    "Reviews": 41671,
    "News": 4256,
    "Translation": 26820,
    "multilingual": 128406,
}


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs


class CTRLTokenizer(PreTrainedTokenizer):
    """
    Construct a CTRL tokenizer. Based on Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    control_codes = CONTROL_CODES

    def __init__(self, vocab_file, merges_file, unk_token="<unk>", **kwargs):
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        super().__init__(unk_token=unk_token, **kwargs)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = "@@ ".join(word)
        word = word[:-4]
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        split_tokens = []

        words = re.findall(r"\S+\n?", text)

        for token in words:
            split_tokens.extend(list(self.bpe(token).split(" ")))
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    # def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
    #     filtered_tokens = ' '.join(self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens))
    #     tokens_generated_so_far = re.sub('(@@ )', '', string=filtered_tokens)
    #     tokens_generated_so_far = re.sub('(@@ ?$)', '', string=tokens_generated_so_far)
    #     return ''.join(tokens_generated_so_far)


__all__ = ["CTRLTokenizer"]

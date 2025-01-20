# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Whisper."""

import json
import os
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_file": "tokenizer.json",
    "merges_file": "merges.txt",
    "normalizer_file": "normalizer.json",
}


MAX_MODEL_INPUT_SIZES = {
    "openai/whisper-base": 448,
}


# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


logger = logging.get_logger(__name__)


# Copied from transformers.models.gpt2.tokenization_gpt2.get_pairs
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
    return pairs


LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}

TASK_IDS = ["translate", "transcribe"]


class WhisperTokenizer(PreTrainedTokenizer):
    """
    Construct a Whisper tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        normalizer_file (`str`, *optional*):
            Path to the normalizer_file file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token. The `decoder_start_token_id` is used to set the first token as
            `"<|startoftranscript|>"` when generating.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word.
        language (`str`, *optional*):
            The language of the transcription text. The corresponding language id token is appended to the start of the
            sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
            `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
        task (`str`, *optional*):
            Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
            fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
        predict_timestamps (`bool`, *optional*, defaults to `False`):
            Whether to omit the `<|notimestamps|>` token at the start of the sequence.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        normalizer_file=None,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        language=None,
        task=None,
        predict_timestamps=False,
        **kwargs,
    ):
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(pad_token, str)
            else pad_token
        )

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        if normalizer_file is not None:
            with open(normalizer_file, encoding="utf-8") as vocab_handle:
                self.english_spelling_normalizer = json.load(vocab_handle)
        else:
            self.english_spelling_normalizer = None

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")

        self.language = language
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.task = task
        self.predict_timestamps = predict_timestamps

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe with GPT2 -> Whisper
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
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
        word = " ".join(word)
        self.cache[token] = word
        return word

    def set_prefix_tokens(self, language: str = None, task: str = None, predict_timestamps: bool = None):
        """
        Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to
        update the prefix tokens as required when fine-tuning. Example:

        ```python
        >>> # instantiate the tokenizer and set the prefix token to Spanish
        >>> tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="spanish")
        >>> # now switch the prefix token from Spanish to French
        >>> tokenizer.set_prefix_tokens(language="french")
        ```

        Args:
            language (`str`, *optional*, defaults to `None`):
                The language of the transcription text.
            task (`str`, *optional*, defaults to `None`):
                Task identifier to append at the start of sequence (if any).
            predict_timestamps (`bool`, *optional*, defaults to `None`):
                Whether to omit the `<|notimestamps|>` token at the start of the sequence.
        """
        self.language = language if language is not None else self.language
        self.task = task if task is not None else self.task
        self.predict_timestamps = predict_timestamps if predict_timestamps is not None else self.predict_timestamps

    @property
    def prefix_tokens(self) -> List[int]:
        bos_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
        translate_token_id = self.convert_tokens_to_ids("<|translate|>")
        transcribe_token_id = self.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_token_id = self.convert_tokens_to_ids("<|notimestamps|>")
        langs = tuple(LANGUAGES.keys())

        if self.language is not None:
            self.language = self.language.lower()
            if self.language in TO_LANGUAGE_CODE:
                language_id = TO_LANGUAGE_CODE[self.language]
            elif self.language in TO_LANGUAGE_CODE.values():
                language_id = self.language
            else:
                is_language_code = len(self.language) == 2
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )

        if self.task is not None:
            if self.task not in TASK_IDS:
                raise ValueError(f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}")

        bos_sequence = [bos_token_id]
        if self.language is not None:
            bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
        if self.task is not None:
            bos_sequence.append(transcribe_token_id if self.task == "transcribe" else translate_token_id)
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence

    # Copied from transformers.models.speech_to_text.tokenization_speech_to_text.Speech2TextTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

    # Copied from transformers.models.speech_to_text.tokenization_speech_to_text.Speech2TextTokenizer.get_special_tokens_mask
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

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1]
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._tokenize with GPT2 -> Whisper
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_token_to_id with GPT2 -> Whisper
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) in a token (str) using the vocab. Whisper's base tokenizer always decodes OOV
        tokens as "", thus we do not use the `unk_token` here.
        """
        return self.decoder.get(index, "")

    def _normalize(self, text):
        warnings.warn(
            "The private method `_normalize` is deprecated and will be removed in v5 of Transformers."
            "You can normalize an input string using the Whisper English normalizer using the `normalize` method."
        )
        return self.normalize(text)

    def _basic_normalize(self, text, remove_diacritics=False):
        warnings.warn(
            "The private method `_basic_normalize` is deprecated and will be removed in v5 of Transformers."
            "You can normalize an input string using the Whisper basic normalizer using the `basic_normalize` method."
        )
        return self.basic_normalize(text, remove_diacritics=remove_diacritics)

    def normalize(self, text):
        """
        Normalize a given string using the `EnglishTextNormalizer` class, which preforms commons transformation on
        english text.
        """
        normalizer = EnglishTextNormalizer(self.english_spelling_normalizer)
        return normalizer(text)

    @staticmethod
    def basic_normalize(text, remove_diacritics=False):
        """
        Normalize a given string using the `BasicTextNormalizer` class, which preforms commons transformation on
        multilingual text.
        """
        normalizer = BasicTextNormalizer(remove_diacritics=remove_diacritics)
        return normalizer(text)

    def _decode_with_timestamps(
        self, token_ids, skip_special_tokens=False, time_precision=0.02, segment_size=1500
    ) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
        given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        timestamp_begin = self.all_special_ids[-1] + 1
        outputs = [[]]

        cur_max_timestamp = 0.0
        prev_segments_len = 0.0
        penultimate_timestamp = 0.0

        for i, token in enumerate(token_ids):
            if token >= timestamp_begin:
                timestamp = float((token - timestamp_begin) * time_precision)

                if timestamp < cur_max_timestamp:
                    # next segment has started
                    last_was_single_ending = i >= 2 and not (
                        token_ids[i - 1] >= timestamp_begin and token_ids[i - 2] >= timestamp_begin
                    )
                    if last_was_single_ending:
                        prev_segments_len += time_precision * segment_size
                    else:
                        cur_max_timestamp = penultimate_timestamp
                        prev_segments_len += penultimate_timestamp
                        outputs = outputs[:-2]

                penultimate_timestamp = cur_max_timestamp
                cur_max_timestamp = timestamp

                outputs.append(f"<|{(timestamp + prev_segments_len):.2f}|>")
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = [
            s if isinstance(s, str) else self.decode(s, skip_special_tokens=skip_special_tokens) for s in outputs
        ]
        return "".join(outputs)

    def _compute_offsets(self, token_ids, time_precision=0.02, segment_size=1500):
        """
        Compute offsets for a given tokenized input

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            time_precision (`float`, *optional*, defaults to 0.02):
                The time ratio to convert from token to time.
            segment_size (`int`, *optional*, defaults to 1500):
                The number of features in the input mel spectrogram.
        """
        offsets = []
        # ensure torch tensor of token ids is placed on cpu
        if "torch" in str(type(token_ids)) and (hasattr(token_ids, "cpu") and callable(token_ids.cpu)):
            token_ids = token_ids.cpu()
        token_ids = np.array(token_ids)
        if token_ids.shape[0] > 1 and len(token_ids.shape) > 1:
            raise ValueError("Can only process a single input at a time")
        timestamp_begin = self.all_special_ids[-1] + 1
        timestamp_tokens = token_ids >= timestamp_begin

        consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
        if consecutive.shape[0] == 0 and timestamp_tokens.sum() <= 1:
            # either there are no timestamps or there are no consecutive ones
            return []
        elif np.where(timestamp_tokens)[0][-1] + 1 not in consecutive:
            # we add the final timestamp if it is not already in the list
            consecutive = np.append(consecutive, np.where(timestamp_tokens)[0][-1] + 1)

        last_slice = np.where(timestamp_tokens)[0][0]
        cur_max_timestamp = 0
        prev_segments_len = 0
        for current_slice in consecutive:
            sliced_tokens = token_ids[last_slice:current_slice]
            if len(sliced_tokens) > 1:
                start_timestamp_position = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_position = sliced_tokens[-1].item() - timestamp_begin

                if start_timestamp_position < cur_max_timestamp:
                    # next segment has started
                    is_single_ending = last_slice >= 2 and not (
                        token_ids[last_slice - 2] >= timestamp_begin and token_ids[last_slice - 1] >= timestamp_begin
                    )
                    if is_single_ending:
                        prev_segments_len += segment_size
                    else:
                        prev_segments_len += cur_max_timestamp

                cur_max_timestamp = end_timestamp_position

                # strip timestamp tokens from the text output
                sliced_tokens = self._preprocess_token_ids(sliced_tokens)
                text = self._decode(sliced_tokens)
                text = self._filter_timestamp_ids(text)
                offsets.append(
                    {
                        "text": text,
                        "timestamp": (
                            start_timestamp_position * time_precision + prev_segments_len * time_precision,
                            end_timestamp_position * time_precision + prev_segments_len * time_precision,
                        ),
                    }
                )
            last_slice = current_slice

        return offsets

    @lru_cache
    def timestamp_ids(self, time_precision=0.02):
        """
        Compute the timestamp token ids for a given precision and save to least-recently used (LRU) cache.

        Args:
            time_precision (`float`, *optional*, defaults to 0.02):
                The time ratio to convert from token to time.
        """
        return self.convert_tokens_to_ids([("<|%.2f|>" % (i * time_precision)) for i in range(1500 + 1)])

    def _preprocess_token_ids(self, token_ids, skip_special_tokens: bool = False):
        """
        Pre-process the token ids for decoding by removing the prompt tokens ids and timestamp token ids.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Typically, obtained using the `__call__` method of the tokenizer.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens from the token ids. If `True`, the prompt token ids will be
                removed.
        """
        if skip_special_tokens:
            prompt_token_id = self.convert_tokens_to_ids("<|startofprev|>")
            decoder_start_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
            token_ids = self._strip_prompt(token_ids, prompt_token_id, decoder_start_token_id)

        return token_ids

    def _filter_timestamp_ids(self, token_ids):
        return re.sub(self.timestamp_pat, "", token_ids)

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_offsets: bool = False,
        time_precision: float = 0.02,
        decode_with_timestamps: bool = False,
        normalize: bool = False,
        basic_normalize: bool = False,
        remove_diacritics: bool = False,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding. Will remove the previous tokens (pre-prompt)
                if present.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            output_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output the offsets of the tokens. This should only be set if the model predicted
                timestamps. If there are previous tokens (pre-prompt) to decode, they will only appear in the decoded
                text if they contain timestamp tokens.
            time_precision (`float`, *optional*, defaults to 0.02):
                The time ratio to convert from token to time.
            decode_with_timestamps (`bool`, *optional*, defaults to `False`):
                Whether or not to decode with timestamps included in the raw text.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether or not to apply the English text normalizer to the decoded text. Only applicable when the
                target text is in English. Otherwise, the basic text normalizer should be applied.
            basic_normalize (`bool`, *optional*, defaults to `False`):
                Whether or not to apply the Basic text normalizer to the decoded text. Applicable to multilingual
                target text.
            remove_diacritics (`bool`, *optional*, defaults to `False`):
                Whether or not to remove diacritics when applying the Basic text normalizer. Removing diacritics may
                destroy information in the decoded text, hence it should be used with caution.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.
        Returns:
            `str`: The decoded sentence.
        """
        filtered_ids = self._preprocess_token_ids(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

        text = super().decode(
            filtered_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            normalize=normalize,
            basic_normalize=basic_normalize,
            remove_diacritics=remove_diacritics,
            **kwargs,
        )
        if decode_with_timestamps:
            # legacy method to decode timestamps when not included in the tokenizer vocabulary
            text = self._decode_with_timestamps(
                filtered_ids, time_precision=time_precision, skip_special_tokens=skip_special_tokens
            )
        else:
            text = self._filter_timestamp_ids(text)

        # retrieve offsets
        if output_offsets:
            offsets = self._compute_offsets(token_ids, time_precision=time_precision)
            return {"text": text, "offsets": offsets}
        return text

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        normalize: bool = False,
        basic_normalize: bool = False,
        remove_diacritics: bool = False,
        **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        text = "".join(sub_texts)

        if normalize:
            clean_text = self.normalize(text)
            return clean_text
        elif basic_normalize:
            clean_text = self.basic_normalize(text, remove_diacritics=remove_diacritics)
            return clean_text
        else:
            return text

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.convert_tokens_to_string with GPT2 -> Whisper
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

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
        normalizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["normalizer_file"]
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

        if self.english_spelling_normalizer is not None:
            with open(normalizer_file, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(self.english_spelling_normalizer, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                )

        return vocab_file, merge_file, normalizer_file

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.prepare_for_tokenization with GPT2 -> Whisper
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        self.set_prefix_tokens(task=task, language=language, predict_timestamps=not no_timestamps)
        # prefix tokens are of the form: <|startoftranscript|> <|lang_id|> <|task|> <|notimestamps|>
        # we don't want to force the bos token at position 1, as this is the starting token
        # when we generate, so we slice the prefix tokens to: <|lang_id|> <|task|> <|notimestamps|>
        # to get the forced tokens
        forced_tokens = self.prefix_tokens[1:]
        forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_tokens)]
        return forced_decoder_ids

    def _decode_asr(self, model_outputs, *, return_timestamps, return_language, time_precision):
        return _decode_asr(
            self,
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )

    def get_prompt_ids(self, text: str, return_tensors="np"):
        """Converts prompt text to IDs that can be passed to [`~WhisperForConditionalGeneration.generate`]."""
        batch_encoding = self("<|startofprev|>", " " + text.strip(), add_special_tokens=False)

        # Check for special tokens
        prompt_text_ids = batch_encoding["input_ids"][1:]
        special_token_id = next((x for x in prompt_text_ids if x >= self.all_special_ids[0]), None)
        if special_token_id is not None:
            token = self.convert_ids_to_tokens(special_token_id)
            raise ValueError(f"Encountered text in the prompt corresponding to disallowed special token: {token}.")

        batch_encoding.convert_to_tensors(tensor_type=return_tensors)
        return batch_encoding["input_ids"]

    def _strip_prompt(self, token_ids: List[int], prompt_token_id: int, decoder_start_token_id: int):
        if not isinstance(token_ids, list):
            token_ids = self._convert_to_list(token_ids)

        # handle case of empty token_ids for decoding with timestamps.
        # at this point token_ids is a list, so it is safe to use if not check.
        if not token_ids:
            return token_ids

        has_prompt = token_ids[0] == prompt_token_id
        if has_prompt:
            if decoder_start_token_id in token_ids:
                return token_ids[token_ids.index(decoder_start_token_id) :]
            else:
                return []

        return token_ids

    @staticmethod
    def _convert_to_list(token_ids):
        # convert type to ndarray if necessary
        if hasattr(token_ids, "numpy"):
            if "torch" in str(type(token_ids)):
                token_ids = token_ids.cpu().numpy()
            elif "tensorflow" in str(type(token_ids)):
                token_ids = token_ids.numpy()
        elif "jaxlib" in str(type(token_ids)):
            token_ids = token_ids.tolist()
        # now the token ids are either a numpy array, or a list of lists
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        return token_ids


def _decode_asr(tokenizer, model_outputs, *, return_timestamps, return_language, time_precision, segment_size=1500):
    """
    Internal method meant to only be used by asr pipeline. Handles all the little quirks specific to whisper to handle
    the various options not allowed in other seq2seq models
    """

    # =========== Overview ============
    # - iterate over all outputs
    # - all tokens within output
    # - Each token can be
    #   - language token
    #   - special token
    #   - timestamp token
    #   - text token
    # - We accumulate the text tokens.
    # - We split on end timestamps
    # - Lots of complexity comes from stride and timestamps

    last_language = None

    def new_chunk():
        return {"language": last_language, "timestamp": [None, None], "text": ""}

    # Welcome to the state machine !
    chunks = []
    chunk = new_chunk()
    time_offset = 0.0
    timestamp_begin = tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1
    previous_tokens = []
    previous_token_timestamps = []
    skip = False
    right_stride_start = None

    all_special_ids = set(tokenizer.all_special_ids)
    prompt_token_id = tokenizer.convert_tokens_to_ids("<|startofprev|>")
    decoder_start_token_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    # - iterate over all outputs
    for chunk_id, output in enumerate(model_outputs):
        # We can drop everything to Python list, it's going to make
        # our lives easier
        token_ids = output["tokens"][0].tolist()
        # (possibly) remove the prompt from the token ids
        token_ids = tokenizer._strip_prompt(token_ids, prompt_token_id, decoder_start_token_id)
        if return_timestamps == "word":
            token_timestamps = output["token_timestamps"][0].tolist()

        # Those keep track of timestamps within strides
        # Which need to be skipped and resolve all tokens in a single
        # chunk.
        last_timestamp = None
        first_timestamp = timestamp_begin

        # long form generation: we need to handle the case where the call to generate returns concatenated segments,
        # with underlying multiple calls to generate
        cur_max_timestamp = 0.0
        prev_segments_len = 0.0
        penultimate_timestamp = 0.0

        if "stride" in output:
            chunk_len, stride_left, stride_right = output["stride"]
            # Offset the timings to account for the other `model_outputs`.
            time_offset -= stride_left
            right_stride_start = chunk_len - stride_right

            # Keeping track of timestamps within strides
            # We're going to NOT split on those, and delay until we're
            # out of BOTH stride. Otherwise lots of issues occur and
            # corner cases
            if stride_left:
                first_timestamp = stride_left / time_precision + timestamp_begin
            if stride_right:
                for token in reversed(token_ids):
                    if token >= timestamp_begin:
                        # There can be several token in the right stride
                        # But the last one is ALWAYS going to be skipped
                        if (
                            last_timestamp is not None
                            and (token - timestamp_begin) * time_precision < right_stride_start
                        ):
                            break
                        last_timestamp = token

        current_tokens = []
        current_token_timestamps = []

        # - all tokens within output
        for i, token in enumerate(token_ids):
            # 4 possible states for each token
            # - 1/ Language code
            # - 2/ all other special tokens (which we ignore)
            # - 3/ Timestamp
            # - 4/ Regular text
            if token in all_special_ids:
                # Either language code or other
                text = tokenizer.decode([token])
                # Removing outer shell <|XX|>
                text = text[2:-2]
                language = LANGUAGES.get(text, None)
                if language is not None:
                    # 1/ Indeed some language
                    # TODO Handle when language is different from the previous
                    # one, and we cannot use timestamped tokens to create chunks
                    if last_language and language != last_language and not return_timestamps:
                        previous_tokens.append(current_tokens)
                        resolved_tokens = _find_longest_common_sequence(previous_tokens)
                        resolved_text = tokenizer.decode(resolved_tokens)
                        chunk["text"] = resolved_text
                        chunks.append(chunk)

                        # Flush all our temporary context
                        previous_tokens = []
                        current_tokens = []
                        chunk = new_chunk()
                    chunk["language"] = language
                    last_language = language
                else:
                    # 2/ This is a regular special token, ignoring it
                    pass
            elif token >= timestamp_begin:
                # 3/ Timestamp token

                timestamp = float((token - timestamp_begin) * time_precision)
                if timestamp < cur_max_timestamp:
                    # next segment has started
                    last_was_single_ending = i >= 2 and not (
                        token_ids[i - 1] >= timestamp_begin and token_ids[i - 2] >= timestamp_begin
                    )
                    if last_was_single_ending:
                        prev_segments_len += time_precision * segment_size
                    else:
                        cur_max_timestamp = penultimate_timestamp
                        prev_segments_len += penultimate_timestamp

                penultimate_timestamp = cur_max_timestamp
                cur_max_timestamp = timestamp

                time = (token - timestamp_begin) * time_precision + time_offset + prev_segments_len

                time = round(time, 2)
                if last_timestamp and token >= last_timestamp:
                    # Whisper outputted a timestamp token, but it falls within
                    # our stride, so we're going to skip it for the time being
                    # and resolve this later
                    # Skip is necessary because timestamp tokens always come
                    # by pair, so we need to skip the next one too (which would mark the start of another chunk).
                    skip = True
                elif skip or (previous_tokens and token < first_timestamp):
                    skip = False
                elif chunk["timestamp"][0] is None:
                    chunk["timestamp"][0] = time
                else:
                    # This is the end of the timestamp chunk
                    if time == chunk["timestamp"][0]:
                        # This is a bug in timestamp token output
                        # where we're taking the duplicate token
                        # as a stop where it should be a start.
                        # This is an issue in the underlying model output
                        # Let's just skip it so it becomes de-factor
                        # a start agin
                        pass
                    else:
                        chunk["timestamp"][1] = time
                        # Handling merges.
                        previous_tokens.append(current_tokens)
                        if return_timestamps == "word":
                            previous_token_timestamps.append(current_token_timestamps)
                        resolved_tokens, resolved_token_timestamps = _find_longest_common_sequence(
                            previous_tokens, previous_token_timestamps
                        )
                        resolved_text = tokenizer.decode(resolved_tokens)
                        chunk["text"] = resolved_text
                        if return_timestamps == "word":
                            chunk["words"] = _collate_word_timestamps(
                                tokenizer, resolved_tokens, resolved_token_timestamps, last_language, return_language
                            )
                        chunks.append(chunk)

                        # Flush all our temporary context
                        previous_tokens = []
                        current_tokens = []
                        previous_token_timestamps = []
                        current_token_timestamps = []
                        chunk = new_chunk()
            else:
                # 4/ Regular token
                # We just append to the list of all tokens so we can handle
                # merges later and decode into text.
                current_tokens.append(token)
                if return_timestamps == "word":
                    start_time = round(token_timestamps[i] + time_offset, 2)
                    if i + 1 < len(token_timestamps):
                        end_time = round(token_timestamps[i + 1] + time_offset, 2)
                    else:
                        end_time = None  # should never happen
                    current_token_timestamps.append((start_time, end_time))

        if "stride" in output:
            time_offset += chunk_len - stride_right

        # Leftover tokens
        if current_tokens:
            previous_tokens.append(current_tokens)
            if return_timestamps == "word":
                previous_token_timestamps.append(current_token_timestamps)
        elif not (any(p for p in previous_tokens)):
            chunk = new_chunk()
            previous_tokens = []
            current_tokens = []
            previous_token_timestamps = []
            current_token_timestamps = []

    if previous_tokens:
        if return_timestamps:
            logger.warning(
                "Whisper did not predict an ending timestamp, which can happen if audio is cut off in the middle of a word. "
                "Also make sure WhisperTimeStampLogitsProcessor was used during generation."
            )
        # Happens when we don't use timestamps
        resolved_tokens, resolved_token_timestamps = _find_longest_common_sequence(
            previous_tokens, previous_token_timestamps
        )
        resolved_text = tokenizer.decode(resolved_tokens)
        chunk["text"] = resolved_text
        if return_timestamps == "word":
            chunk["words"] = _collate_word_timestamps(
                tokenizer, resolved_tokens, resolved_token_timestamps, last_language, return_language
            )
        chunks.append(chunk)

    # Preparing and cleaning up the pipeline output
    full_text = "".join(chunk["text"] for chunk in chunks)
    if return_timestamps or return_language:
        for chunk in chunks:
            if not return_timestamps:
                chunk.pop("timestamp")
            else:
                chunk["timestamp"] = tuple(chunk["timestamp"])
            if not return_language:
                chunk.pop("language")

        if return_timestamps == "word":
            new_chunks = []
            for chunk in chunks:
                new_chunks.extend(chunk["words"])
            optional = {"chunks": new_chunks}
        else:
            optional = {"chunks": chunks}
    else:
        optional = {}
    return full_text, optional


def _find_longest_common_sequence(sequences, token_timestamp_sequences=None):
    # It would be much harder to do O(n) because of fault tolerance.
    # We actually have a really good property which is that the total sequence
    # MUST be those subsequences in order.
    # If token_timestamp_sequences is provided, will split those sequences in
    # exactly the same way.

    left_sequence = sequences[0]
    left_length = len(left_sequence)
    total_sequence = []

    if token_timestamp_sequences:
        left_token_timestamp_sequence = token_timestamp_sequences[0]
        total_token_timestamp_sequence = []

    for seq_idx, right_sequence in enumerate(sequences[1:]):
        # index = 0
        max_ = 0.0
        max_indices = (left_length, left_length, 0, 0)
        # Here we're sliding matches
        # [a, b, c, d]
        #          [c, d, f]
        # =        [c] == [d]
        #
        # [a, b, c, d]
        #       [c, d, f]
        # =     [c, d] == [c, d]
        #
        #
        # [a, b, c, d]
        #    [c, d, f]
        #
        # =  [b, c, d] == [c, d, f]
        #
        # [a, b, c, d]
        # [c, d, f]
        #
        # [a, b, c] == [c, d, f]
        #
        # [a, b, c, d]
        # [d, f]
        #
        # [a, b] == [d, f]
        #
        # [a, b, c, d]
        # [f]
        #
        # [a] == [f]
        right_length = len(right_sequence)
        for i in range(1, left_length + right_length):
            # epsilon to favor long perfect matches
            eps = i / 10000.0

            # Slightly convoluted because we don't want out of bound indices
            # This will be necessary for a small conflict resolution optimization
            # later
            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = np.array(left_sequence[left_start:left_stop])

            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = np.array(right_sequence[right_start:right_stop])

            # We can only match subsequences of the same size.
            if len(left) != len(right):
                raise RuntimeError(
                    "There is a bug within whisper `decode_asr` function, please report it. Dropping to prevent bad inference."
                )

            if token_timestamp_sequences:
                # Get length of longest subsequence of tokens that match
                # and have timestamps that are in order
                matches = sum(
                    1
                    for idx, elem in enumerate(left)
                    if (
                        elem == right[idx]
                        and left_token_timestamp_sequence[left_start + idx]
                        <= token_timestamp_sequences[seq_idx + 1][right_start + idx]
                    )
                )

            else:
                matches = np.sum(left == right)

            matching = matches / i + eps
            if matches > 1 and matching > max_:
                max_ = matching
                max_indices = (left_start, left_stop, right_start, right_stop)

        (left_start, left_stop, right_start, right_stop) = max_indices

        # This is a small conflict optimization since those sequences overlap
        # in audio.
        # We're going to give more confidence to the left sequence
        # for the left of the overlap,
        # and to the right of the sequence, for the right of the overlap
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2
        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)

        if token_timestamp_sequences:
            total_token_timestamp_sequence.extend(left_token_timestamp_sequence[:left_mid])
            left_token_timestamp_sequence = token_timestamp_sequences[seq_idx + 1][right_mid:]

    total_sequence.extend(left_sequence)

    if token_timestamp_sequences is None:
        return total_sequence

    if len(token_timestamp_sequences) > 0:
        total_token_timestamp_sequence.extend(left_token_timestamp_sequence)
        return total_sequence, total_token_timestamp_sequence
    else:
        return total_sequence, []


def _collate_word_timestamps(tokenizer, tokens, token_timestamps, language, return_language):
    words, _, token_indices = _combine_tokens_into_words(tokenizer, tokens, language)

    optional_language_field = {"language": language} if return_language else {}

    timings = [
        {
            "text": word,
            "timestamp": (token_timestamps[indices[0]][0], token_timestamps[indices[-1]][1]),
            **optional_language_field,
        }
        for word, indices in zip(words, token_indices)
    ]
    return timings


def _combine_tokens_into_words(
    tokenizer,
    tokens: List[int],
    language: str = None,
    prepend_punctuations: str = "\"'“¡¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
):
    """
    Groups tokens by word. Returns a tuple containing a list of strings with the words, and a list of `token_id`
    sequences with the tokens making up each word.
    """
    if language is None:
        language = tokenizer.language
    if language is None:
        language = "english"

    if language in {"chinese", "japanese", "thai", "lao", "myanmar", "cantonese"}:
        # These languages don't typically use spaces.
        words, word_tokens, token_indices = _split_tokens_on_unicode(tokenizer, tokens)
    else:
        words, word_tokens, token_indices = _split_tokens_on_spaces(tokenizer, tokens)

    _merge_punctuations(words, word_tokens, token_indices, prepend_punctuations, append_punctuations)
    return words, word_tokens, token_indices


def _split_tokens_on_unicode(tokenizer, tokens: List[int]):
    """Combine tokens into words by splitting at any position where the tokens are decoded as valid unicode points."""
    decoded_full = tokenizer.decode(tokens, decode_with_timestamps=True)
    replacement_char = "\ufffd"

    words = []
    word_tokens = []
    token_indices = []
    current_tokens = []
    current_indices = []
    unicode_offset = 0

    for token_idx, token in enumerate(tokens):
        current_tokens.append(token)
        current_indices.append(token_idx)
        decoded = tokenizer.decode(current_tokens, decode_with_timestamps=True)

        if (
            replacement_char not in decoded
            or decoded_full[unicode_offset + decoded.index(replacement_char)] == replacement_char
        ):
            words.append(decoded)
            word_tokens.append(current_tokens)
            token_indices.append(current_indices)
            current_tokens = []
            current_indices = []
            unicode_offset += len(decoded)

    return words, word_tokens, token_indices


def _split_tokens_on_spaces(tokenizer, tokens: List[int]):
    """Combine tokens into words by splitting at whitespace and punctuation tokens."""
    subwords, subword_tokens_list, subword_indices_list = _split_tokens_on_unicode(tokenizer, tokens)
    words = []
    word_tokens = []
    token_indices = []

    for subword, subword_tokens, subword_indices in zip(subwords, subword_tokens_list, subword_indices_list):
        special = subword_tokens[0] >= tokenizer.eos_token_id
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

        if special or with_space or punctuation or len(words) == 0:
            words.append(subword)
            word_tokens.append(subword_tokens)
            token_indices.append(subword_indices)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)
            token_indices[-1].extend(subword_indices)

    return words, word_tokens, token_indices


def _merge_punctuations(words, tokens, indices, prepended, appended):
    """Merges punctuation tokens with neighboring words."""
    # prepend punctuations
    i = len(words) - 2
    j = len(words) - 1
    while i >= 0:
        if words[i].startswith(" ") and words[i].strip() in prepended:
            words[j] = words[i] + words[j]
            tokens[j] = tokens[i] + tokens[j]
            indices[j] = indices[i] + indices[j]
            words[i] = ""
            tokens[i] = []
            indices[i] = []
        else:
            j = i
        i -= 1

    # append punctuations
    i = 0
    j = 1
    while j < len(words):
        if not words[i].endswith(" ") and words[j] in appended:
            words[i] += words[j]
            tokens[i] += tokens[j]
            indices[i] += indices[j]
            words[j] = ""
            tokens[j] = []
            indices[j] = []
        else:
            i = j
        j += 1

    # remove elements that are now empty
    words[:] = [word for word in words if word]
    tokens[:] = [token for token in tokens if token]
    indices[:] = [idx for idx in indices if idx]


__all__ = ["WhisperTokenizer"]

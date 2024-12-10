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
import re
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from tokenizers import AddedToken, pre_tokenizers, processors

from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from .tokenization_whisper import LANGUAGES, TASK_IDS, TO_LANGUAGE_CODE, WhisperTokenizer, _decode_asr


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_file": "tokenizer.json",
    "merges_file": "merges.txt",
    "normalizer_file": "normalizer.json",
}


class WhisperTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Whisper tokenizer (backed by HuggingFace's *tokenizers* library).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        normalizer_file (`str`, *optional*):
            Path to the normalizer_file file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token. The `decoder_start_token_id` is used to set the first token as
            `"<|startoftranscript|>"` when generating.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Whisper tokenizer detect beginning of words by the preceding space).
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
    slow_tokenizer_class = WhisperTokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        normalizer_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
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

        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.add_bos_token = kwargs.pop("add_bos_token", False)

        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        if normalizer_file is not None:
            with open(normalizer_file, encoding="utf-8") as vocab_handle:
                self.english_spelling_normalizer = json.load(vocab_handle)
        else:
            self.english_spelling_normalizer = None

        self.add_prefix_space = add_prefix_space
        self.timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")

        self.language = language
        self.task = task
        self.predict_timestamps = predict_timestamps

    # Copied from transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast._batch_encode_plus
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    # Copied from transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast._encode_plus
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._decode_with_timestamps
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

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._compute_offsets
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
    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.timestamp_ids
    def timestamp_ids(self, time_precision=0.02):
        """
        Compute the timestamp token ids for a given precision and save to least-recently used (LRU) cache.

        Args:
            time_precision (`float`, *optional*, defaults to 0.02):
                The time ratio to convert from token to time.
        """
        return self.convert_tokens_to_ids([("<|%.2f|>" % (i * time_precision)) for i in range(1500 + 1)])

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._preprocess_token_ids
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

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._filter_timestamp_ids
    def _filter_timestamp_ids(self, token_ids):
        return re.sub(self.timestamp_pat, "", token_ids)

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.decode
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
        self, *args, normalize: bool = False, basic_normalize: bool = False, remove_diacritics: bool = False, **kwargs
    ) -> str:
        text = super()._decode(*args, **kwargs)

        if normalize:
            clean_text = self._normalize(text)
            return clean_text
        elif basic_normalize:
            clean_text = self._basic_normalize(text, remove_diacritics=remove_diacritics)
            return clean_text
        else:
            return text

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._normalize
    def _normalize(self, text):
        warnings.warn(
            "The private method `_normalize` is deprecated and will be removed in v5 of Transformers."
            "You can normalize an input string using the Whisper English normalizer using the `normalize` method."
        )
        return self.normalize(text)

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._basic_normalize
    def _basic_normalize(self, text, remove_diacritics=False):
        warnings.warn(
            "The private method `_basic_normalize` is deprecated and will be removed in v5 of Transformers."
            "You can normalize an input string using the Whisper basic normalizer using the `basic_normalize` method."
        )
        return self.basic_normalize(text, remove_diacritics=remove_diacritics)

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.normalize
    def normalize(self, text):
        """
        Normalize a given string using the `EnglishTextNormalizer` class, which preforms commons transformation on
        english text.
        """
        normalizer = EnglishTextNormalizer(self.english_spelling_normalizer)
        return normalizer(text)

    @staticmethod
    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.basic_normalize
    def basic_normalize(text, remove_diacritics=False):
        """
        Normalize a given string using the `BasicTextNormalizer` class, which preforms commons transformation on
        multilingual text.
        """
        normalizer = BasicTextNormalizer(remove_diacritics=remove_diacritics)
        return normalizer(text)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)

        normalizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["normalizer_file"]
        )

        if self.english_spelling_normalizer is not None:
            with open(normalizer_file, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(self.english_spelling_normalizer, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                )

        return tuple(files) + (normalizer_file,)

    def set_prefix_tokens(self, language: str = None, task: str = None, predict_timestamps: bool = None):
        """
        Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to
        update the prefix tokens as required when fine-tuning. Example:

        ```python
        >>> # instantiate the tokenizer and set the prefix token to Spanish
        >>> tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny", language="spanish")
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

        prefix_token_ids = self.prefix_tokens
        prefixes = self.convert_ids_to_tokens(prefix_token_ids)
        eos = self.eos_token
        eos_token_id = self.eos_token_id
        prefix_template = " ".join([f"{token}:0" for token in prefixes])
        self.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{prefix_template} $A:0 {eos}:0",
            pair=f"{prefix_template} $A:0 $B:1 {eos}:1",
            special_tokens=[
                (eos, eos_token_id),
                *zip(prefixes, prefix_token_ids),
            ],
        )

    @property
    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.prefix_tokens
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

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.get_special_tokens_mask
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

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.get_decoder_prompt_ids
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

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.get_prompt_ids
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

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._strip_prompt
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
    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._convert_to_list
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

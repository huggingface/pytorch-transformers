# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import json
import os
import re
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "source_spm": "source.spm",
    "target_spm": "target.spm",
    "vocab": "vocab.json",
    "target_vocab_file": "target_vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}


SPIECE_UNDERLINE = "▁"

# Example URL https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/vocab.json


class MarianTokenizer(PreTrainedTokenizer):
    r"""
    Construct a Marian tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        source_spm (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary for the source language.
        target_spm (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary for the target language.
        source_lang (`str`, *optional*):
            A string representing the source language.
        target_lang (`str`, *optional*):
            A string representing the target language.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        model_max_length (`int`, *optional*, defaults to 512):
            The maximum sentence length the model accepts.
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<eop>", "<eod>"]`):
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

    Examples:

    ```python
    >>> from transformers import MarianForCausalLM, MarianTokenizer

    >>> model = MarianForCausalLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    >>> tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    >>> src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
    >>> tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
    >>> inputs = tokenizer(src_texts, text_target=tgt_texts, return_tensors="pt", padding=True)

    >>> outputs = model(**inputs)  # should work
    ```"""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    language_code_re = re.compile(">>.+<<")  # type: re.Pattern

    def __init__(
        self,
        source_spm,
        target_spm,
        vocab,
        target_vocab_file=None,
        source_lang=None,
        target_lang=None,
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=512,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        separate_vocabs=False,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        assert Path(source_spm).exists(), f"cannot find spm source {source_spm}"

        self.separate_vocabs = separate_vocabs
        self.encoder = load_json(vocab)
        if str(unk_token) not in self.encoder:
            raise KeyError("<unk> token must be in the vocab")
        assert str(pad_token) in self.encoder

        if separate_vocabs:
            self.target_encoder = load_json(target_vocab_file)
            self.decoder = {v: k for k, v in self.target_encoder.items()}
            self.supported_language_codes = []
        else:
            self.decoder = {v: k for k, v in self.encoder.items()}
            self.supported_language_codes: list = [k for k in self.encoder if k.startswith(">>") and k.endswith("<<")]

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.spm_files = [source_spm, target_spm]

        # load SentencePiece model for pre-processing
        self.spm_source = load_spm(source_spm, self.sp_model_kwargs)
        self.spm_target = load_spm(target_spm, self.sp_model_kwargs)
        self.current_spm = self.spm_source
        self.current_encoder = self.encoder

        # Multilingual target side: default to using first supported language code.

        self._setup_normalizer()

        super().__init__(
            # bos_token=bos_token,  unused. Start decoding with config.decoder_start_token_id
            source_lang=source_lang,
            target_lang=target_lang,
            unk_token=unk_token,
            eos_token=eos_token,
            pad_token=pad_token,
            model_max_length=model_max_length,
            sp_model_kwargs=self.sp_model_kwargs,
            target_vocab_file=target_vocab_file,
            separate_vocabs=separate_vocabs,
            **kwargs,
        )

    def _setup_normalizer(self):
        try:
            from sacremoses import MosesPunctNormalizer

            self.punc_normalizer = MosesPunctNormalizer(self.source_lang).normalize
        except (ImportError, FileNotFoundError):
            warnings.warn("Recommended: pip install sacremoses.")
            self.punc_normalizer = lambda x: x

    def normalize(self, x: str) -> str:
        """Cover moses empty string edge case. They return empty list for '' input!"""
        return self.punc_normalizer(x) if x else ""

    def _convert_token_to_id(self, token):
        return self.current_encoder.get(token, self.current_encoder[self.unk_token])

    def remove_language_code(self, text: str):
        """Remove language codes like >>fr<< before sentencepiece"""
        match = self.language_code_re.match(text)
        code: list = [match.group(0)] if match else []
        return code, self.language_code_re.sub("", text)

    def _tokenize(self, text: str) -> List[str]:
        code, text = self.remove_language_code(text)
        pieces = self.current_spm.encode(text, out_type=str)
        return code + pieces

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        return self.decoder.get(index, self.unk_token)

    def batch_decode(self, sequences, **kwargs):
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            use_source_tokenizer (`bool`, *optional*, defaults to `False`):
                Whether or not to use the source tokenizer to decode sequences (only applicable in sequence-to-sequence
                problems).
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        return super().batch_decode(sequences, **kwargs)

    def decode(self, token_ids, **kwargs):
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            use_source_tokenizer (`bool`, *optional*, defaults to `False`):
                Whether or not to use the source tokenizer to decode sequences (only applicable in sequence-to-sequence
                problems).
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        return super().decode(token_ids, **kwargs)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Uses source spm if _decode_use_source_tokenizer is True, and target spm otherwise"""
        sp_model = self.spm_source if self._decode_use_source_tokenizer else self.spm_target
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += sp_model.decode_pieces(current_sub_tokens) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += sp_model.decode_pieces(current_sub_tokens)
        out_string = out_string.replace(SPIECE_UNDERLINE, " ")
        return out_string.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def _switch_to_input_mode(self):
        self.current_spm = self.spm_source
        self.current_encoder = self.encoder

    def _switch_to_target_mode(self):
        self.current_spm = self.spm_target
        if self.separate_vocabs:
            self.current_encoder = self.target_encoder

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        saved_files = []

        if self.separate_vocabs:
            out_src_vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"],
            )
            out_tgt_vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["target_vocab_file"],
            )
            save_json(self.encoder, out_src_vocab_file)
            save_json(self.target_encoder, out_tgt_vocab_file)
            saved_files.append(out_src_vocab_file)
            saved_files.append(out_tgt_vocab_file)
        else:
            out_vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"]
            )
            save_json(self.encoder, out_vocab_file)
            saved_files.append(out_vocab_file)

        for spm_save_filename, spm_orig_path, spm_model in zip(
            [VOCAB_FILES_NAMES["source_spm"], VOCAB_FILES_NAMES["target_spm"]],
            self.spm_files,
            [self.spm_source, self.spm_target],
        ):
            spm_save_path = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + spm_save_filename
            )
            if os.path.abspath(spm_orig_path) != os.path.abspath(spm_save_path) and os.path.isfile(spm_orig_path):
                copyfile(spm_orig_path, spm_save_path)
                saved_files.append(spm_save_path)
            elif not os.path.isfile(spm_orig_path):
                with open(spm_save_path, "wb") as fi:
                    content_spiece_model = spm_model.serialized_model_proto()
                    fi.write(content_spiece_model)
                saved_files.append(spm_save_path)

        return tuple(saved_files)

    def get_vocab(self) -> Dict:
        return self.get_src_vocab()

    def get_src_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def get_tgt_vocab(self):
        return dict(self.target_encoder, **self.added_tokens_decoder)

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state.update(
            {k: None for k in ["spm_source", "spm_target", "current_spm", "punc_normalizer", "target_vocab_file"]}
        )
        return state

    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.spm_source, self.spm_target = (load_spm(f, self.sp_model_kwargs) for f in self.spm_files)
        self.current_spm = self.spm_source
        self._setup_normalizer()

    def num_special_tokens_to_add(self, *args, **kwargs):
        """Just EOS"""
        return 1

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # call it once instead of inside list comp
        all_special_ids.remove(self.unk_token_id)  # <unk> is only sometimes special
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]


def load_spm(path: str, sp_model_kwargs: Dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    spm = sentencepiece.SentencePieceProcessor(**sp_model_kwargs)
    spm.Load(path)
    return spm


def save_json(data, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Union[Dict, List]:
    with open(path, "r") as f:
        return json.load(f)


__all__ = ["MarianTokenizer"]

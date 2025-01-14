# coding=utf-8
# Copyright 2022 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI Jukebox."""

import json
import os
import re
import unicodedata
from json.encoder import INFINITY
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import regex

from ....tokenization_utils import AddedToken, PreTrainedTokenizer
from ....tokenization_utils_base import BatchEncoding
from ....utils import TensorType, is_flax_available, is_tf_available, is_torch_available, logging
from ....utils.generic import _is_jax, _is_numpy


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "artists_file": "artists.json",
    "lyrics_file": "lyrics.json",
    "genres_file": "genres.json",
}


class JukeboxTokenizer(PreTrainedTokenizer):
    """
    Constructs a Jukebox tokenizer. Jukebox can be conditioned on 3 different inputs :
        - Artists, unique ids are associated to each artist from the provided dictionary.
        - Genres, unique ids are associated to each genre from the provided dictionary.
        - Lyrics, character based tokenization. Must be initialized with the list of characters that are inside the
        vocabulary.

    This tokenizer does not require training. It should be able to process a different number of inputs:
    as the conditioning of the model can be done on the three different queries. If None is provided, defaults values will be used.:

    Depending on the number of genres on which the model should be conditioned (`n_genres`).
    ```python
    >>> from transformers import JukeboxTokenizer

    >>> tokenizer = JukeboxTokenizer.from_pretrained("openai/jukebox-1b-lyrics")
    >>> tokenizer("Alan Jackson", "Country Rock", "old town road")["input_ids"]
    [tensor([[   0,    0,    0, 6785,  546,   41,   38,   30,   76,   46,   41,   49,
               40,   76,   44,   41,   27,   30]]), tensor([[  0,   0,   0, 145,   0]]), tensor([[  0,   0,   0, 145,   0]])]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    If nothing is provided, the genres and the artist will either be selected randomly or set to None

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to:
    this superclass for more information regarding those methods.

    However the code does not allow that and only supports composing from various genres.

    Args:
        artists_file (`str`):
            Path to the vocabulary file which contains a mapping between artists and ids. The default file supports
            both "v2" and "v3"
        genres_file (`str`):
            Path to the vocabulary file which contain a mapping between genres and ids.
        lyrics_file (`str`):
            Path to the vocabulary file which contains the accepted characters for the lyrics tokenization.
        version (`List[str]`, `optional`, default to `["v3", "v2", "v2"]`) :
            List of the tokenizer versions. The `5b-lyrics`'s top level prior model was trained using `v3` instead of
            `v2`.
        n_genres (`int`, `optional`, defaults to 1):
            Maximum number of genres to use for composition.
        max_n_lyric_tokens (`int`, `optional`, defaults to 512):
            Maximum number of lyric tokens to keep.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        artists_file,
        genres_file,
        lyrics_file,
        version=["v3", "v2", "v2"],
        max_n_lyric_tokens=512,
        n_genres=5,
        unk_token="<|endoftext|>",
        **kwargs,
    ):
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        self.version = version
        self.max_n_lyric_tokens = max_n_lyric_tokens
        self.n_genres = n_genres
        self._added_tokens_decoder = {0: unk_token}

        with open(artists_file, encoding="utf-8") as vocab_handle:
            self.artists_encoder = json.load(vocab_handle)

        with open(genres_file, encoding="utf-8") as vocab_handle:
            self.genres_encoder = json.load(vocab_handle)

        with open(lyrics_file, encoding="utf-8") as vocab_handle:
            self.lyrics_encoder = json.load(vocab_handle)

        oov = r"[^A-Za-z0-9.,:;!?\-'\"()\[\] \t\n]+"
        # In v2, we had a n_vocab=80 and in v3 we missed + and so n_vocab=79 of characters.
        if len(self.lyrics_encoder) == 79:
            oov = oov.replace(r"\-'", r"\-+'")

        self.out_of_vocab = regex.compile(oov)
        self.artists_decoder = {v: k for k, v in self.artists_encoder.items()}
        self.genres_decoder = {v: k for k, v in self.genres_encoder.items()}
        self.lyrics_decoder = {v: k for k, v in self.lyrics_encoder.items()}
        super().__init__(
            unk_token=unk_token,
            n_genres=n_genres,
            version=version,
            max_n_lyric_tokens=max_n_lyric_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.artists_encoder) + len(self.genres_encoder) + len(self.lyrics_encoder)

    def get_vocab(self):
        return {
            "artists_encoder": self.artists_encoder,
            "genres_encoder": self.genres_encoder,
            "lyrics_encoder": self.lyrics_encoder,
        }

    def _convert_token_to_id(self, list_artists, list_genres, list_lyrics):
        """Converts the artist, genre and lyrics tokens to their index using the vocabulary.
        The total_length, offset and duration have to be provided in order to select relevant lyrics and add padding to
        the lyrics token sequence.
        """
        artists_id = [self.artists_encoder.get(artist, 0) for artist in list_artists]
        for genres in range(len(list_genres)):
            list_genres[genres] = [self.genres_encoder.get(genre, 0) for genre in list_genres[genres]]
            list_genres[genres] = list_genres[genres] + [-1] * (self.n_genres - len(list_genres[genres]))

        lyric_ids = [[self.lyrics_encoder.get(character, 0) for character in list_lyrics[0]], [], []]
        return artists_id, list_genres, lyric_ids

    def _tokenize(self, lyrics):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens. Only the lyrics are split into character for the character-based vocabulary.
        """
        # only lyrics are not tokenized, but character based is easily handled
        return list(lyrics)

    def tokenize(self, artist, genre, lyrics, **kwargs):
        """
        Converts three strings in a 3 sequence of tokens using the tokenizer
        """
        artist, genre, lyrics = self.prepare_for_tokenization(artist, genre, lyrics)
        lyrics = self._tokenize(lyrics)
        return artist, genre, lyrics

    def prepare_for_tokenization(
        self, artists: str, genres: str, lyrics: str, is_split_into_words: bool = False
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        Args:
            artist (`str`):
                The artist name to prepare. This will mostly lower the string
            genres (`str`):
                The genre name to prepare. This will mostly lower the string.
            lyrics (`str`):
                The lyrics to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
        """
        for idx in range(len(self.version)):
            if self.version[idx] == "v3":
                artists[idx] = artists[idx].lower()
                genres[idx] = [genres[idx].lower()]
            else:
                artists[idx] = self._normalize(artists[idx]) + ".v2"
                genres[idx] = [
                    self._normalize(genre) + ".v2" for genre in genres[idx].split("_")
                ]  # split is for the full dictionary with combined genres

        if self.version[0] == "v2":
            self.out_of_vocab = regex.compile(r"[^A-Za-z0-9.,:;!?\-'\"()\[\] \t\n]+")
            vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;!?-+'\"()[] \t\n"
            self.vocab = {vocab[index]: index + 1 for index in range(len(vocab))}
            self.vocab["<unk>"] = 0
            self.n_vocab = len(vocab) + 1
            self.lyrics_encoder = self.vocab
            self.lyrics_decoder = {v: k for k, v in self.vocab.items()}
            self.lyrics_decoder[0] = ""
        else:
            self.out_of_vocab = regex.compile(r"[^A-Za-z0-9.,:;!?\-+'\"()\[\] \t\n]+")

        lyrics = self._run_strip_accents(lyrics)
        lyrics = lyrics.replace("\\", "\n")
        lyrics = self.out_of_vocab.sub("", lyrics), [], []
        return artists, genres, lyrics

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _normalize(self, text: str) -> str:
        """
        Normalizes the input text. This process is for the genres and the artist

        Args:
            text (`str`):
                Artist or Genre string to normalize
        """

        accepted = (
            [chr(i) for i in range(ord("a"), ord("z") + 1)]
            + [chr(i) for i in range(ord("A"), ord("Z") + 1)]
            + [chr(i) for i in range(ord("0"), ord("9") + 1)]
            + ["."]
        )
        accepted = frozenset(accepted)
        pattern = re.compile(r"_+")
        text = "".join([c if c in accepted else "_" for c in text.lower()])
        text = pattern.sub("_", text).strip("_")
        return text

    def convert_lyric_tokens_to_string(self, lyrics: List[str]) -> str:
        return " ".join(lyrics)

    def convert_to_tensors(
        self, inputs, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
    ):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                unset, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.TENSORFLOW:
            if not is_tf_available():
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )
            import tensorflow as tf

            as_tensor = tf.constant
            is_tensor = tf.is_tensor
        elif tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch

            as_tensor = torch.tensor
            is_tensor = torch.is_tensor
        elif tensor_type == TensorType.JAX:
            if not is_flax_available():
                raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
            import jax.numpy as jnp  # noqa: F811

            as_tensor = jnp.array
            is_tensor = _is_jax
        else:
            as_tensor = np.asarray
            is_tensor = _is_numpy

        # Do the tensor conversion in batch

        try:
            if prepend_batch_axis:
                inputs = [inputs]

            if not is_tensor(inputs):
                inputs = as_tensor(inputs)
        except NameError as e:
            if 'NumPy' in str(e):
                logger.warning("Ensure that the correct version of NumPy is installed.")
            raise NameError(f"A NameError has ocurred: {e}") from e
        except RuntimeError as e:
                raise RuntimeError(f"""
                A runtime error has ocurred: {e}
                This could be due to numpy version mismatches, check your version
                against the transformers accepted numpy verions.
                """) from e
        except ValueError as e:   # noqa E722
            raise ValueError(
                "Unable to create tensor, you should probably activate truncation and/or padding "
                "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
            ) from e
        except Exception as e:
            raise Exception(f"Something went wrong that was not a runtime or value error: {e}") from e

        return inputs

    def __call__(self, artist, genres, lyrics="", return_tensors="pt") -> BatchEncoding:
        """Convert the raw string to a list of token ids

        Args:
            artist (`str`):
                Name of the artist.
            genres (`str`):
                List of genres that will be mixed to condition the audio
            lyrics (`str`, *optional*, defaults to `""`):
                Lyrics used to condition the generation
        """
        input_ids = [0, 0, 0]
        artist = [artist] * len(self.version)
        genres = [genres] * len(self.version)

        artists_tokens, genres_tokens, lyrics_tokens = self.tokenize(artist, genres, lyrics)
        artists_id, genres_ids, full_tokens = self._convert_token_to_id(artists_tokens, genres_tokens, lyrics_tokens)

        attention_masks = [-INFINITY] * len(full_tokens[-1])
        input_ids = [
            self.convert_to_tensors(
                [input_ids + [artists_id[i]] + genres_ids[i] + full_tokens[i]], tensor_type=return_tensors
            )
            for i in range(len(self.version))
        ]
        return BatchEncoding({"input_ids": input_ids, "attention_masks": attention_masks})

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the tokenizer's vocabulary dictionary to the provided save_directory.

        Args:
            save_directory (`str`):
                A path to the directory where to saved. It will be created if it doesn't exist.

            filename_prefix (`Optional[str]`, *optional*):
                A prefix to add to the names of the files saved by the tokenizer.

        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        artists_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["artists_file"]
        )
        with open(artists_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.artists_encoder, ensure_ascii=False))

        genres_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["genres_file"]
        )
        with open(genres_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.genres_encoder, ensure_ascii=False))

        lyrics_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["lyrics_file"]
        )
        with open(lyrics_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.lyrics_encoder, ensure_ascii=False))

        return (artists_file, genres_file, lyrics_file)

    def _convert_id_to_token(self, artists_index, genres_index, lyric_index):
        """
        Converts an index (integer) in a token (str) using the vocab.

        Args:
            artists_index (`int`):
                Index of the artist in its corresponding dictionary.
            genres_index (`Union[List[int], int]`):
               Index of the genre in its corresponding dictionary.
            lyric_index (`List[int]`):
                List of character indices, which each correspond to a character.
        """
        artist = self.artists_decoder.get(artists_index)
        genres = [self.genres_decoder.get(genre) for genre in genres_index]
        lyrics = [self.lyrics_decoder.get(character) for character in lyric_index]
        return artist, genres, lyrics

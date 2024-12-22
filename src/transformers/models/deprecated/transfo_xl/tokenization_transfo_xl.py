# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
Tokenization classes for Transformer XL model. Adapted from https://github.com/kimiyoung/transformer-xl.
"""

import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple

import numpy as np

from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
    cached_file,
    is_sacremoses_available,
    is_torch_available,
    logging,
    requires_backends,
    strtobool,
    torch_only_method,
)


if is_sacremoses_available():
    import sacremoses as sm


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "pretrained_vocab_file": "vocab.pkl",
    "pretrained_vocab_file_torch": "vocab.bin",
    "vocab_file": "vocab.txt",
}


PRETRAINED_CORPUS_ARCHIVE_MAP = {
    "transfo-xl/transfo-xl-wt103": "https://huggingface.co/transfo-xl/transfo-xl-wt103/resolve/main/corpus.bin",
}
CORPUS_NAME = "corpus.bin"

MATCH_NUMBERS = r"(?<=\d)[,.](?=\d)", r" @\g<0>@ "
DETOKENIZE_NUMBERS = [(r" @\,@ ", r","), (r" @\.@ ", r".")]


def tokenize_numbers(text_array: List[str]) -> List[str]:
    """
    Splits large comma-separated numbers and floating point values. This is done by replacing commas with ' @,@ ' and
    dots with ' @.@ '.

    Args:
        text_array: An already tokenized text as list.

    Returns:
        A list of strings with tokenized numbers.

    Example:

    ```python
    >>> tokenize_numbers(["$", "5,000", "1.73", "m"])
    ['$', '5', '@,@', '000', '1', '@.@', '73', 'm']
    ```"""
    tokenized = []
    for i in range(len(text_array)):
        reg, sub = MATCH_NUMBERS
        replaced = re.sub(reg, sub, text_array[i]).split()
        tokenized.extend(replaced)

    return tokenized


def detokenize_numbers(text: str) -> str:
    """
    Inverts the operation of *tokenize_numbers*. This is replacing ' @,@ ' and ' @.@' by ',' and '.'.

    Args:
        text: A string where the number should be detokenized.

    Returns:
        A detokenized string.

    Example:

    ```python
    >>> detokenize_numbers("$ 5 @,@ 000 1 @.@ 73 m")
    '$ 5,000 1.73 m'
    ```"""
    for reg, sub in DETOKENIZE_NUMBERS:
        text = re.sub(reg, sub, text)
    return text


class TransfoXLTokenizer(PreTrainedTokenizer):
    """
    Construct a Transformer-XL tokenizer adapted from Vocab class in [the original
    code](https://github.com/kimiyoung/transformer-xl). The Transformer-XL tokenizer is a word-level tokenizer (no
    sub-word tokenization).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        special (`List[str]`, *optional*):
            A list of special tokens (to be treated by the original implementation of this tokenizer).
        min_freq (`int`, *optional*, defaults to 0):
            The minimum number of times a token has to be present in order to be kept in the vocabulary (otherwise it
            will be mapped to `unk_token`).
        max_size (`int`, *optional*):
            The maximum size of the vocabulary. If left unset, it will default to the size of the vocabulary found
            after excluding the tokens according to the `min_freq` rule.
        lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        delimiter (`str`, *optional*):
            The delimiter used between tokens.
        vocab_file (`str`, *optional*):
            File containing the vocabulary (from the original implementation).
        pretrained_vocab_file (`str`, *optional*):
            File containing the vocabulary as saved with the `save_pretrained()` method.
        never_split (`List[str]`, *optional*):
            List of tokens that should never be split. If no list is specified, will simply use the existing special
            tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (`str`, *optional*, defaults to `"<eos>"`):
            The end of sequence token.
        additional_special_tokens (`List[str]`, *optional*, defaults to `['<formula>']`):
            A list of additional special tokens (for the HuggingFace functionality).
        language (`str`, *optional*, defaults to `"en"`):
            The language of this tokenizer (used for mose preprocessing).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids"]

    def __init__(
        self,
        special=None,
        min_freq=0,
        max_size=None,
        lower_case=False,
        delimiter=None,
        vocab_file=None,
        pretrained_vocab_file: str = None,
        never_split=None,
        unk_token="<unk>",
        eos_token="<eos>",
        additional_special_tokens=["<formula>"],
        language="en",
        **kwargs,
    ):
        logger.error(
            "`TransfoXL` was deprecated due to security issues linked to `pickle.load` in `TransfoXLTokenizer`. "
            "See more details on this model's documentation page: "
            "`https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/transfo-xl.md`."
        )

        requires_backends(self, "sacremoses")
        if special is None:
            special = []
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file
        self.punctuation_symbols = '!"#$%&()*+,-./\\:;<=>?@[\\]^_`{|}~'
        self.punction_without_space_before_pattern = re.compile(rf"[^\s][{self.punctuation_symbols}]")
        self.punctuation_with_space_around_pattern = self._compile_space_around_punctuation_pattern()
        self.language = language
        self.moses_punct_normalizer = sm.MosesPunctNormalizer(language)
        self.moses_tokenizer = sm.MosesTokenizer(language)
        self.moses_detokenizer = sm.MosesDetokenizer(language)
        self.idx2sym = []
        self.sym2idx = OrderedDict()
        # This try... catch... is not beautiful but honestly this tokenizer was not made to be used
        # in a library like ours, at all.
        try:
            vocab_dict = None
            if pretrained_vocab_file is not None:
                # Priority on pickle files (support PyTorch and TF)
                if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
                    raise ValueError(
                        "This part uses `pickle.load` which is insecure and will execute arbitrary code that is "
                        "potentially malicious. It's recommended to never unpickle data that could have come from an "
                        "untrusted source, or that could have been tampered with. If you already verified the pickle "
                        "data and decided to use it, you can set the environment variable "
                        "`TRUST_REMOTE_CODE` to `True` to allow it."
                    )
                with open(pretrained_vocab_file, "rb") as f:
                    vocab_dict = pickle.load(f)

                # Loading a torch-saved transfo-xl vocab dict with pickle results in an integer
                # Entering this if statement means that we tried to load a torch-saved file with pickle, and we failed.
                # We therefore load it with torch, if it's available.
                if isinstance(vocab_dict, int):
                    if not is_torch_available():
                        raise ImportError(
                            "Not trying to load dict with PyTorch as you need to install pytorch to load "
                            "from a PyTorch pretrained vocabulary, "
                            "or activate it with environment variables USE_TORCH=1 and USE_TF=0."
                        )
                    vocab_dict = torch.load(pretrained_vocab_file, weights_only=True)

            if vocab_dict is not None:
                for key, value in vocab_dict.items():
                    if key not in self.__dict__ or key in ["sym2idx", "idx2sym"]:
                        self.__dict__[key] = value
            elif vocab_file is not None:
                self.build_vocab()

        except Exception as e:
            raise ValueError(
                f"Unable to parse file {pretrained_vocab_file}. Unknown format. "
                "If you tried to load a model saved through TransfoXLTokenizerFast, "
                "please note they are not compatible."
            ) from e

        if vocab_file is not None:
            self.build_vocab()

        super().__init__(
            special=special,
            min_freq=min_freq,
            max_size=max_size,
            lower_case=lower_case,
            delimiter=delimiter,
            vocab_file=vocab_file,
            pretrained_vocab_file=pretrained_vocab_file,
            never_split=never_split,
            unk_token=unk_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens,
            language=language,
            **kwargs,
        )

        # these are not required to initialize the parent class as only used when tokenizing.
        if never_split is None:
            never_split = self.all_special_tokens
        self.never_split = never_split

    @property
    def do_lower_case(self):
        return self.lower_case

    def _compile_space_around_punctuation_pattern(self):
        look_ahead_for_special_token = f"(?=[{self.punctuation_symbols}])"
        look_ahead_to_match_all_except_space = r"(?=[^\s])"
        return re.compile(r"" + look_ahead_for_special_token + look_ahead_to_match_all_except_space)

    def count_file(self, path, verbose=False, add_eos=False):
        if verbose:
            logger.info(f"counting file {path} ...")
        assert os.path.exists(path), f"Input file {path} not found"

        sents = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    logger.info(f"    line {idx}")
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
        sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose:
            logger.info(f"counting {len(sents)} sents ...")
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                logger.info(f"    line {idx}")
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        if "<UNK>" in self.sym2idx:
            self.unk_idx = self.sym2idx["<UNK>"]
        elif "<unk>" in self.sym2idx:
            self.unk_idx = self.sym2idx["<unk>"]
        else:
            raise ValueError("Token not in vocabulary and no <unk> token in vocabulary for replacement.")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["pretrained_vocab_file"],
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "wb") as f:
            pickle.dump(self.__dict__, f)
        return (vocab_file,)

    def build_vocab(self):
        if self.vocab_file:
            logger.info(f"building vocab from {self.vocab_file}")
            self._build_from_file(self.vocab_file)
            logger.info(f"Final vocab size {len(self.sym2idx)}")
        else:
            logger.info(f"building vocab with min_freq={self.min_freq}, max_size={self.max_size}")
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq:
                    break
                self.add_symbol(sym)

            logger.info(f"Final vocab size {len(self.sym2idx)} from {len(self.counter)} unique tokens")

    @torch_only_method
    def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False):
        if verbose:
            logger.info(f"encoding file {path} ...")
        assert os.path.exists(path), f"Output file {path} not found"
        encoded = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    logger.info(f"    line {idx}")
                symbols = self.tokenize(line, add_eos=add_eos, add_double_eos=add_double_eos)
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    @torch_only_method
    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose:
            logger.info(f"encoding {len(sents)} sents ...")
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                logger.info(f"    line {idx}")
            encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, f"{sym.strip('<>')}_idx", self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def move_added_token(self, token: str, target_idx: int):
        """
        Moves an added token to a specific position in the vocab. This method should be used when resizing an embedding
        layer other than the last one in the `AdaptiveEmbedding` in order to move the token in the tokenizer from the
        default position (at the very end) to the desired one.

        Args:
            token: The token to move to a specific position in the vocab.
            target_idx: The position where the token should be moved to.
        """
        assert token in self.added_tokens_encoder, "Token which should be moved has to be an added token"
        assert token not in self.idx2sym, "Token which should be moved is already in vocab"

        # Insert sym into vocab
        self.idx2sym.insert(target_idx, token)
        self.sym2idx[token] = target_idx

        # Shift following indices in sym2idx
        for idx in range(target_idx + 1, len(self.idx2sym)):
            current_sym = self.idx2sym[idx]
            self.sym2idx[current_sym] = idx

        # Delete token from added_tokens
        old_index = self._added_tokens_encoder.pop(token)
        self._added_tokens_decoder.pop(old_index)

    def moses_punct_norm(self, text):
        return self.moses_punct_normalizer.normalize(text)

    def moses_tokenize(self, text):
        return self.moses_tokenizer.tokenize(
            text, aggressive_dash_splits=True, return_str=False, escape=False, protected_patterns=self.never_split
        )

    def moses_pipeline(self, text: str) -> List[str]:
        """
        Does basic tokenization using [`sacremoses.MosesPunctNormalizer`] and [`sacremoses.MosesTokenizer`] with
        *aggressive_dash_splits=True* (see [`sacremoses.tokenize.MosesTokenizer.tokenize`]). Additionally, large
        comma-separated numbers and floating point values are split. E.g. "23,000 people are 1.80m tall" -> "23 @,@ 000
        people are 1 @.@ 80m tall"

        Args:
            text: Text to be tokenize

        Returns:
            A list of tokenized string

        Example:

        ```python
        >>> tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl/transfo-xl-wt103")
        >>> tokenizer.moses_pipeline("23,000 people are 1.80 m tall")
        ['23', '@,@', '000', 'people', 'are', '1', '@.@', '80', 'm', 'tall']
        ```"""
        text = self.moses_punct_norm(text)
        text = self.moses_tokenize(text)
        text = tokenize_numbers(text)
        return text

    def _convert_id_to_token(self, idx):
        """Converts an id in a token (BPE) using the vocab."""
        assert 0 <= idx < len(self), f"Index {idx} out of vocabulary range"
        return self.idx2sym[idx]

    def _convert_token_to_id(self, sym):
        """Converts a token (str) in an id using the vocab."""
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # logger.info(f'encounter unk {sym}')
            # assert '<eos>' not in sym
            if hasattr(self, "unk_idx"):
                return self.sym2idx.get(sym, self.unk_idx)
            # Backward compatibility with pre-trained models
            elif "<unk>" in self.sym2idx:
                return self.sym2idx["<unk>"]
            elif "<UNK>" in self.sym2idx:
                return self.sym2idx["<UNK>"]
            else:
                raise ValueError("Token not in vocabulary and no <unk> token in vocabulary for replacement.")

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) in a single string. Additionally, the split numbers are converted back
        into it's original form.
        """
        out_string = self.moses_detokenizer.detokenize(tokens)
        return detokenize_numbers(out_string).strip()

    @torch_only_method
    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.convert_tokens_to_ids(symbols))

    @property
    def vocab_size(self):
        return len(self.idx2sym)

    def get_vocab(self):
        vocab = self.sym2idx.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, line, add_eos=False, add_double_eos=False):
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == "":
            symbols = line
        else:
            symbols = self.moses_pipeline(line)

        if add_double_eos:  # lm1b
            return ["<S>"] + symbols + ["<S>"]
        elif add_eos:
            return symbols + ["<eos>"]
        else:
            return symbols


class LMOrderedIterator:
    def __init__(self, data, bsz, bptt, device="cpu", ext_len=None):
        """
        data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i + 1 : i + 1 + seq_len]

        data_out = data.transpose(0, 1).contiguous().to(self.device)
        target_out = target.transpose(0, 1).contiguous().to(self.device)

        return data_out, target_out, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.0
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator:
    def __init__(self, data, bsz, bptt, device="cpu", ext_len=None, shuffle=False):
        """
        data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle else np.array(range(len(self.data)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    @torch_only_method
    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[n_retain + n_filled : n_retain + n_filled + n_new, i] = streams[i][:n_new]
                        target[n_filled : n_filled + n_new, i] = streams[i][1 : n_new + 1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data_out = data.transpose(0, 1).contiguous().to(self.device)
            target_out = target.transpose(0, 1).contiguous().to(self.device)

            yield data_out, target_out, self.bptt

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device="cpu", ext_len=None, shuffle=False):
        self.paths = paths
        self.vocab = vocab

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)

        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)

        for path in self.paths:
            # sent_stream is an iterator
            sent_stream = self.get_sent_stream(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch


class TransfoXLCorpus:
    @classmethod
    @torch_only_method
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a pre-processed corpus.
        """
        vocab = TransfoXLTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        # redirect to the cache, if necessary
        try:
            resolved_corpus_file = cached_file(pretrained_model_name_or_path, CORPUS_NAME, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                f"Corpus '{pretrained_model_name_or_path}' was not found in corpus list"
                f" ({', '.join(PRETRAINED_CORPUS_ARCHIVE_MAP.keys())}. We assumed '{pretrained_model_name_or_path}'"
                f" was a path or url but couldn't find files {CORPUS_NAME} at this path or url."
            )
            return None
        if is_local:
            logger.info(f"loading corpus file {resolved_corpus_file}")
        else:
            logger.info(f"loading corpus file {CORPUS_NAME} from cache at {resolved_corpus_file}")

        # Instantiate tokenizer.
        corpus = cls(*inputs, **kwargs)
        corpus_dict = torch.load(resolved_corpus_file, weights_only=True)
        for key, value in corpus_dict.items():
            corpus.__dict__[key] = value
        corpus.vocab = vocab
        if corpus.train is not None:
            corpus.train = torch.tensor(corpus.train, dtype=torch.long)
        if corpus.valid is not None:
            corpus.valid = torch.tensor(corpus.valid, dtype=torch.long)
        if corpus.test is not None:
            corpus.test = torch.tensor(corpus.test, dtype=torch.long)
        return corpus

    def __init__(self, *args, **kwargs):
        self.vocab = TransfoXLTokenizer(*args, **kwargs)
        self.dataset = None
        self.train = None
        self.valid = None
        self.test = None

    def build_corpus(self, path, dataset):
        self.dataset = dataset

        if self.dataset in ["ptb", "wt2", "enwik8", "text8"]:
            self.vocab.count_file(os.path.join(path, "train.txt"))
            self.vocab.count_file(os.path.join(path, "valid.txt"))
            self.vocab.count_file(os.path.join(path, "test.txt"))
        elif self.dataset == "wt103":
            self.vocab.count_file(os.path.join(path, "train.txt"))
        elif self.dataset == "lm1b":
            train_path_pattern = os.path.join(
                path,
                "1-billion-word-language-modeling-benchmark-r13output",
                "training-monolingual.tokenized.shuffled",
                "news.en-*",
            )
            train_paths = glob.glob(train_path_pattern)
            # the vocab will load from file when build_vocab() is called

        self.vocab.build_vocab()

        if self.dataset in ["ptb", "wt2", "wt103"]:
            self.train = self.vocab.encode_file(os.path.join(path, "train.txt"), ordered=True)
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"), ordered=True)
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"), ordered=True)
        elif self.dataset in ["enwik8", "text8"]:
            self.train = self.vocab.encode_file(os.path.join(path, "train.txt"), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"), ordered=True, add_eos=False)
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"), ordered=True, add_eos=False)
        elif self.dataset == "lm1b":
            self.train = train_paths
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"), ordered=False, add_double_eos=True)
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"), ordered=False, add_double_eos=True)

    def get_iterator(self, split, *args, **kwargs):
        if split == "train":
            if self.dataset in ["ptb", "wt2", "wt103", "enwik8", "text8"]:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == "lm1b":
                kwargs["shuffle"] = True
                data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
        elif split in ["valid", "test"]:
            data = self.valid if split == "valid" else self.test
            if self.dataset in ["ptb", "wt2", "wt103", "enwik8", "text8"]:
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == "lm1b":
                data_iter = LMShuffledIterator(data, *args, **kwargs)
        else:
            data_iter = None
            raise ValueError(f"Split not recognized: {split}")

        return data_iter


@torch_only_method
def get_lm_corpus(datadir, dataset):
    fn = os.path.join(datadir, "cache.pt")
    fn_pickle = os.path.join(datadir, "cache.pkl")
    if os.path.exists(fn):
        logger.info("Loading cached dataset...")
        corpus = torch.load(fn_pickle, weights_only=True)
    elif os.path.exists(fn):
        logger.info("Loading cached dataset from pickle...")
        if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
            raise ValueError(
                "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
                "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
                "that could have been tampered with. If you already verified the pickle data and decided to use it, "
                "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
            )
        with open(fn, "rb") as fp:
            corpus = pickle.load(fp)
    else:
        logger.info(f"Producing dataset {dataset}...")
        kwargs = {}
        if dataset in ["wt103", "wt2"]:
            kwargs["special"] = ["<eos>"]
            kwargs["lower_case"] = False
        elif dataset == "ptb":
            kwargs["special"] = ["<eos>"]
            kwargs["lower_case"] = True
        elif dataset == "lm1b":
            kwargs["special"] = []
            kwargs["lower_case"] = False
            kwargs["vocab_file"] = os.path.join(datadir, "1b_word_vocab.txt")
        elif dataset in ["enwik8", "text8"]:
            pass

        corpus = TransfoXLCorpus(datadir, dataset, **kwargs)
        torch.save(corpus, fn)

    return corpus

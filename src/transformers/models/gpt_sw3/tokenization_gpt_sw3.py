"""The tokenizer used by the GPT-SW3 models."""

import os
import re
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece as spm

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


class GPTSw3Tokenizer(PreTrainedTokenizer):
    """
    Construct an GPTSw3 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Example usage:
    ```python
    >>> from transformers import GPTSw3Tokenizer

    >>> tokenizer = GPTSw3Tokenizer.from_pretrained("AI-Sweden-Models/gpt-sw3-126m")
    >>> tokenizer("Svenska är kul!")["input_ids"]
    [1814, 377, 3617, 63504]
    ```

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        remove_space (`bool`, *optional*, defaults to `False`):
            Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
        keep_accents (`bool`, *optional*, defaults to `False`):
            Whether or not to keep accents when tokenizing.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths. If not provided, will
            default to '<pad>' or '<unk>' depending on model size.
        unk_token (`str`, *optional*):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. If not provided, will default to '<unk>'.
        eos_token (`str`, *optional*):
            The end of sequence token seen during pretraining. If not provided, will default to '<|endoftext|>'
        bos_token (`str`, *optional*):
            The beginning of sequence token that can be used for downstream task, was not seen during pretraining. If
            not provided, will default to '<s>' or '<|endoftext|>', depending on model size.
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

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
        whitespaces (`set`):
            The whitespaces that are replaced in the whitespace normalization in preprocessing.
        non_printing_characters_re (`Pattern`):
            The compiled regular expression to remove non-printing characters in preprocessing.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=False,
        keep_accents=False,
        pad_token=None,
        unk_token=None,
        eos_token=None,
        bos_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        name_or_path = kwargs.get("name_or_path")
        if name_or_path is None:
            logger.warning(
                "name_or_path not provided, will work for all GPTSw3 models except gpt-sw3-7b,"
                " you are testing the model, this can safely be ignored"
            )
            name_or_path = "None"

        # Default definitions for our 2 tokenizer versions, with None-checks to enable proper testing
        eos_token = "<|endoftext|>" if eos_token is None else eos_token
        unk_token = "<unk>" if unk_token is None else unk_token
        if "gpt-sw3-7b" in name_or_path:
            pad_token = unk_token if pad_token is None else pad_token
            bos_token = eos_token if bos_token is None else bos_token
        else:
            pad_token = "<pad>" if pad_token is None else pad_token
            bos_token = "<s>" if bos_token is None else bos_token

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # Used for whitespace normalization in input texts
        # fmt : off
        self.whitespaces = {" ", " ", " ", " ", " ", "　", " ", " ", " ", " ", "￼", ""}
        # fmt : on

        # Regular expression to remove non-printing characters (e.g. some unicode control chars) in preprocessing
        self.non_printing_characters_re = re.compile(
            f"[{''.join(map(chr, list(range(0, 9)) + list(range(11, 32)) + list(range(127, 160)) + [160, 173, 8203]))}]"
        )

        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.__getstate__
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.__setstate__
    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    @property
    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.vocab_size
    def vocab_size(self) -> int:
        return len(self.sp_model)

    def preprocess_text(self, text: str) -> str:
        """
        Returns the preprocessed text. This procedure is identical to what was used when training the tokenizer.
        """

        # Remove non-printing characters
        text = self.non_printing_characters_re.sub("", text)

        # Normalize whitespaces
        text = "".join([char if char not in self.whitespaces else " " for char in text])

        # NFC Unicode normalization
        text = unicodedata.normalize("NFC", text)
        return text

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        text = self.preprocess_text(text)
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id (int) using the vocab."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (int) to a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """Returns the input string, this function is overridden to remove the default clean up."""
        return out_string

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (strings) to a single string. Special tokens remain intact."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                # TODO: Check if this is needed, as it ensures that decode(encode(doc)) != doc by adding extra whitespace in the decoded document
                if not prev_is_special:
                    out_string += " "

                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)

        return out_string

    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.get_vocab
    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def encode_fast(
        self, text: Union[str, List[str]], return_tensors: Union[str, bool] = False
    ) -> Union[List[int], List[List[int]], "torch.Tensor"]:
        """
        Encodes a text or batch of texts to token ids using preprocessing and the raw SP tokenizer. This has reduced
        functionality but is often much faster.

        Does NOT handle special tokens correctly, these can manually be added as ids afterwards.

        Does NOT support padding, these can manually be added as ids afterwards.

        Use default HuggingFace tokenization methods for full functionality.

        Args:
            text (`str` or `List[str]`): One or several text(s) to convert to token ids.
            return_tensors (`str` or `bool`): Returns PyTorch tensors if set to True or "pt"

        Returns:
            `List[int]`, `List[List[int]]`, or `torch.Tensor`: The encoded text(s) as token ids.
        """

        if isinstance(text, str):
            text = self.preprocess_text(text)
            token_ids = self.sp_model.encode(text)
        else:
            text = [self.preprocess_text(t) for t in text]
            token_ids = self.sp_model.encode(text)

        if return_tensors is True or return_tensors == "pt":
            token_ids = torch.tensor(token_ids)

        return token_ids

    def decode_fast(self, token_ids: Union[int, List[int]]) -> str:
        """
        Encodes a text or batch of texts to token ids using preprocessing and the raw SP tokenizer. This has reduced
        functionality but is often much faster.

        Args:
            token_ids (`int` or `List[int]`): Encoded token or text as token id(s).

        Returns:
            `str`: Decoded text
        """

        return self.sp_model.decode(token_ids)


__all__ = ["GPTSw3Tokenizer"]

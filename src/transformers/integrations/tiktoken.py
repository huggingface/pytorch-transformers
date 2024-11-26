from pathlib import Path
from typing import Any

from transformers.convert_slow_tokenizer import TikTokenConverter
from transformers.tokenization_utils_fast import TIKTOKEN_VOCAB_FILE, TOKENIZER_FILE


def convert_tiktoken_to_fast(encoding: Any, output_dir: str):
    """
    Converts given `tiktoken` encoding to `PretrainedTokenizerFast` and saves the configuration of converted tokenizer
    on disk.

    Args:
        encoding (`str` or `tiktoken.Encoding`):
            Tokenizer from `tiktoken` library. If `encoding` is `str`, the tokenizer will be loaded with
            `tiktoken.get_encoding(encoding)`.
        output_dir (`str`):
            Save path for converted tokenizer configuration file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    save_file = output_dir / "tiktoken" / TIKTOKEN_VOCAB_FILE
    tokenizer_file = output_dir / TOKENIZER_FILE

    save_file_absolute = str(save_file.absolute())
    output_file_absolute = str(tokenizer_file.absolute())

    try:
        from tiktoken import get_encoding
        from tiktoken.load import dump_tiktoken_bpe

        if isinstance(encoding, str):
            encoding = get_encoding(encoding)

        dump_tiktoken_bpe(encoding._mergeable_ranks, save_file_absolute)
    except ImportError:
        raise ValueError(
            "`tiktoken` is required to save a `tiktoken` file. Install it with " "`pip install tiktoken`."
        )

    tokenizer = TikTokenConverter(
        vocab_file=save_file_absolute, pattern=encoding._pat_str, additional_special_tokens=encoding._special_tokens
    ).tokenizer()
    tokenizer.save(output_file_absolute)

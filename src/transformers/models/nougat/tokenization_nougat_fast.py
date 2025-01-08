# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Fast tokenizer class for Nougat.
"""

import re
from functools import partial
from multiprocessing import Pool
from typing import List, Union

import numpy as np

from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import add_end_docstrings

from ...utils import is_levenshtein_available, is_nltk_available, logging, requires_backends


if is_levenshtein_available():
    from Levenshtein import ratio

if is_nltk_available():
    import nltk


logger = logging.get_logger(__name__)


INIT_TOKENIZER_DOCSTRING += """
        tokenizer_object ([`tokenizers.Tokenizer`]):
            A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗
            tokenizers](../fast_tokenizers) for more information.
        tokenizer_file ([`str`]):
            A path to a local JSON file representing a previously serialized [`tokenizers.Tokenizer`] object from 🤗
            tokenizers.
"""


VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}


def markdown_compatible(text: str) -> str:
    """
    Make text compatible with Markdown formatting.

    This function makes various text formatting adjustments to make it compatible with Markdown.

    Args:
        text (`str`):
            The input text to be made Markdown-compatible.

    Returns:
        `str`: The Markdown-compatible text.
    """
    # equation tag
    # Replace lines that start with a pattern like (decimal) \[some text\] with \[[some text] \tag{decimal}\].
    text = re.sub(r"^\(([\d.]+[a-zA-Z]?)\) \\\[(.+?)\\\]$", r"\[\2 \\tag{\1}\]", text, flags=re.M)
    # Replace lines that start with a pattern like \[some text\] (decimal)  with \[[some text] \tag{decimal}\].
    text = re.sub(r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\)$", r"\[\1 \\tag{\2}\]", text, flags=re.M)
    # Replace lines that start with a pattern like \[some text\] (digits) \[another text\]  with \[[some text] \tag{digits}\] [another text].
    text = re.sub(
        r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\) (\\\[.+?\\\])$",
        r"\[\1 \\tag{\2}\] \3",
        text,
        flags=re.M,
    )
    # multi line
    text = text.replace(r"\. ", ". ")
    # bold formatting
    text = text.replace(r"\bm{", r"\mathbf{").replace(r"{\\bm ", r"\mathbf{")
    text = re.sub(r"\\mbox{ ?\\boldmath\$(.*?)\$}", r"\\mathbf{\1}", text)
    # Reformat urls (http, ftp and https only) to markdown [url](url) clickable format
    text = re.sub(
        r"((?:http|ftp|https):\/\/(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-]))",
        r"[\1](\1)",
        text,
    )
    # algorithms
    text = re.sub(r"```\s*(.+?)\s*```", r"```\n\1\n```", text, flags=re.S)

    return text


def normalize_list_like_lines(generation):
    """
    Normalize lines in the given text that resemble list items. The function looks for lines that start optionally with
    '-' or '*', possibly followed by Roman numerals or digits indicating nesting levels. The function reformats such
    lines to make them more structured.

    Args:
        generation (str): The input text containing lines that need to be normalized.

    Returns:
        str: The input text with the list-like lines normalized.

    Note:
        The function uses regular expressions to identify and reformat the list-like lines. The patterns capture
        optional bullet points, nesting levels indicated by numerals, and the actual list item content. The
        normalization adjusts the bullet point style and nesting levels based on the captured patterns.
    """

    # This matches lines starting with - or *, not followed by - or * (lists)
    # that are then numbered by digits \d or roman numerals (one or more)
    # and then, optional additional numbering of this line is captured
    # this is then fed to re.finditer.
    pattern = r"(?:^)(-|\*)?(?!-|\*) ?((?:\d|[ixv])+ )?.+? (-|\*) (((?:\d|[ixv])+)\.(\d|[ixv]) )?.*(?:$)"

    for match in reversed(list(re.finditer(pattern, generation, flags=re.I | re.M))):
        start, stop = match.span()
        delim = match.group(3) + " "
        splits = match.group(0).split(delim)
        replacement = ""

        if match.group(1) is not None:
            splits = splits[1:]
            delim1 = match.group(1) + " "
        else:
            delim1 = ""
            continue  # Skip false positives

        pre, post = generation[:start], generation[stop:]

        for i, item in enumerate(splits):
            level = 0
            potential_numeral, _, rest = item.strip().partition(" ")
            if not rest:
                continue
            # Infer current nesting level based on detected numbering
            if re.match(r"^[\dixv]+((?:\.[\dixv])?)+$", potential_numeral, flags=re.I | re.M):
                level = potential_numeral.count(".")

            replacement += (
                ("\n" if i > 0 else "") + ("\t" * level) + (delim if i > 0 or start == 0 else delim1) + item.strip()
            )

        if post == "":
            post = "\n"

        generation = pre + replacement + post

    return generation


def find_next_punctuation(text: str, start_idx=0):
    """
    Find the index of the next punctuation mark.

    Args:
        text (`str`):
            String to examine
        start_idx (`int`, *optional*)
            Index where to start
    """

    for i in range(start_idx, len(text)):
        if text[i] in [".", "?", "!", "\n"]:
            return i

    return None


def truncate_repetitions(text: str, min_len: int = 30) -> str:
    """
    Attempt to truncate repeating segments in the input string.

    This function looks for the longest repeating substring at the end of the input string and truncates it to appear
    only once. To be considered for removal, repetitions need to be continuous.

    Args:
        text (`str`):
            The input raw prediction to be truncated.
        min_len (int):
            The minimum length of the repeating segment.

    Returns:
        `str`: The input string with repeated segments truncated.
    """
    text_lower = text.lower()
    text_length = len(text_lower)

    if text_length < 2 * min_len:
        return text

    # try to find a length at which the tail is repeating
    max_repetition_length = None
    for repetition_length in range(min_len, int(text_length / 2)):
        # check if there is a repetition at the end
        same = True
        for i in range(0, repetition_length):
            if text_lower[text_length - repetition_length - i - 1] != text_lower[text_length - i - 1]:
                same = False
                break

        if same:
            max_repetition_length = repetition_length

    if max_repetition_length is None:
        return text

    lcs = text_lower[-max_repetition_length:]

    # remove all but the last repetition
    substituted_text = text
    substituted_text_lower = text_lower
    while substituted_text_lower.endswith(lcs):
        substituted_text = substituted_text[:-max_repetition_length]
        substituted_text_lower = substituted_text_lower[:-max_repetition_length]

    # this is the tail with the repetitions
    repeating_tail = text_lower[len(substituted_text_lower) :]

    # add until next punctuation and make sure last sentence is not repeating
    substituted_text_lower_out = substituted_text_lower
    while True:
        sentence_end = find_next_punctuation(text_lower, len(substituted_text_lower_out))
        sentence_start = find_next_punctuation(text_lower[::-1], len(substituted_text_lower_out))
        if sentence_end and sentence_start:
            sentence = text_lower[sentence_start:sentence_end]
            substituted_text_lower_out = text_lower[: sentence_end + 1]
            if sentence in repeating_tail:
                break
        else:
            break

    text_out = text[: len(substituted_text_lower_out)]

    return text_out


def remove_numbers(lines):
    def _clean(s):
        return re.sub(r"(?:[\d_]|\*\*)", "", s).strip()

    if isinstance(lines, str):
        return _clean(lines)
    out = []
    for l in lines:
        out.append(_clean(l))
    return out


def get_slices(lines, clean_lines):
    """
    Get slices of text based on specific criteria within the lines.

    This function identifies and returns slices of text from the input lines based on certain conditions.

    These conditions were chosen by the Nougat authors:
    - The slice is less than 200 characters long.
    - The slice is more than 3 characters long.
    - The slice does not start with "[MISSING_PAGE".
    - The slice is either the same as the next slice or the ratio of the two in terms of Levensthein distance is
      greater than 0.9.

    Args:
        lines (`List[str]`):
            The list of lines containing the text.
        clean_lines (`List[str]`):
            A cleaned version of the text (without numbers).

    Returns:
        `List[tuple]`: A list of tuples representing the start and end indices of text slices.
    """
    indices = np.zeros(len(lines))
    for i in range(len(lines) - 1):
        j = i + 1
        while not clean_lines[j] and j < len(lines) - 1:
            j += 1
        if (
            len(clean_lines[i]) < 200
            and len(clean_lines[i]) > 3
            and len(clean_lines[j]) < 200
            and len(clean_lines[j]) > 3
            and not clean_lines[i].startswith("[MISSING_PAGE")
            and (clean_lines[i] == clean_lines[j] or ratio(clean_lines[i], clean_lines[j]) > 0.9)
        ):
            indices[i:j] = 1
    ids = np.where(indices)[0]
    slices = []
    if len(ids) == 0:
        return slices
    j0 = 0
    for j, x in enumerate(np.diff(ids) > 3):
        if x:
            slices.append((ids[j0], ids[j] + 2))
            j0 = j + 1
    slices.append((ids[j0], ids[-1] + 2))
    return [sli for sli in slices if sli[1] - sli[0] > 15]


def remove_slice_from_lines(lines, clean_text, slice) -> str:
    """
    Remove a slice of text from the lines based on specific criteria.

    This function identifies a slice of text within the lines and removes it based on certain conditions.

    Args:
        lines (list of str): The list of lines containing the text.
        clean_text (list of str): A cleaned version of the text (without numbers).
        slice (tuple): A tuple representing the start and end indices of the slice to be removed.

    Returns:
        str: The removed slice of text as a single string.
    """
    base = clean_text[slice[0]]
    section = list(slice)
    check_start_flag = False
    # backwards pass, at most 5 lines
    for line_idx in range(max(0, slice[0] - 1), max(0, slice[0] - 5), -1):
        if not lines[line_idx]:
            continue
        if lines[line_idx] == "## References":
            section[0] = line_idx
            break
        elif ratio(base, remove_numbers(lines[line_idx])) < 0.9:
            section[0] = line_idx + 1
            potential_ref = remove_numbers(lines[max(0, line_idx - 1)].partition("* [")[-1])
            if len(potential_ref) >= 0.75 * len(base) and ratio(base, potential_ref) < 0.9:
                section[0] = line_idx
            check_start_flag = True
            break
    # forward pass, at most 5 lines
    for line_idx in range(min(len(lines), slice[1]), min(len(lines), slice[1] + 5)):
        if ratio(base, remove_numbers(lines[line_idx])) < 0.9:
            section[1] = line_idx
            break
    if len(lines) <= section[1]:
        section[1] = len(lines) - 1
    to_delete = "\n".join(lines[section[0] : section[1] + 1])
    # cut off next page content
    itera, iterb = enumerate(lines[section[1] - 1]), enumerate(lines[section[1]])
    while True:
        try:
            (ia, a) = next(itera)
            while a.isnumeric():
                (ia, a) = next(itera)
            (ib, b) = next(iterb)
            while b.isnumeric():
                (ib, b) = next(iterb)
            if a != b:
                break
        except StopIteration:
            break
    if check_start_flag and "* [" in to_delete:
        to_delete = "* [" + to_delete.partition("* [")[-1]
    try:
        delta = len(lines[section[1]]) - ib - 1
        if delta > 0:
            to_delete = to_delete[:-delta]
    except UnboundLocalError:
        pass

    return to_delete.strip()


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class NougatTokenizerFast(PreTrainedTokenizerFast):
    """
    Fast tokenizer for Nougat (backed by HuggingFace tokenizers library).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods. This class mainly adds Nougat-specific
    methods for postprocessing the generated text.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.

        clean_up_tokenization_spaces (`str`, *optional*, defaults to `False`):
            Wether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra
            spaces.

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.

        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )
        self.vocab_file = vocab_file

    def remove_hallucinated_references(self, text: str) -> str:
        """
        Remove hallucinated or missing references from the text.

        This function identifies and removes references that are marked as missing or hallucinated from the input text.

        Args:
            text (`str`):
                The input text containing references.

        Returns:
            `str`: The text with hallucinated references removed.
        """
        lines = text.split("\n")
        if len(lines) == 0:
            return ""
        clean_lines = remove_numbers(lines)
        slices = get_slices(lines, clean_lines)
        to_delete = []
        for slice in slices:
            to_delete.append(remove_slice_from_lines(lines, clean_lines, slice))
        for to_delete in reversed(to_delete):
            text = text.replace(to_delete, "\n\n[MISSING_PAGE_POST]\n\n")
        text = re.sub(
            r"## References\n+\[MISSING_PAGE_POST(:\d+)?\]",
            "\n\n[MISSING_PAGE_POST\\1]",
            text,
        )
        return text

    def correct_tables(self, generation: str) -> str:
        """
        Takes a generated string and fixes tables/tabulars to make them match the markdown format needed.

        Args:
            generation (str): The generated text to be postprocessed.

        Returns:
            str: The postprocessed text.

        Example:

        ```python
        correct_tables("\\begin{table} \\begin{tabular}{l l} & \\ \\end{tabular} \\end{table}")
        "\\begin{table}\n\\begin{tabular}{l l} & \\ \\end{tabular}\n\\end{table}"
        ```
        """
        # remove obvious wrong tables
        for l in generation.split("\n"):
            if l.count("\\begin{tabular}") > 15 or l.count("\\multicolumn") > 60 or l.count("&") > 400:
                generation = generation.replace(l, "")
        # whitespace corrections

        generation = generation.replace("\\begin{table} \\begin{tabular}", "\\begin{table}\n\\begin{tabular}")
        generation = generation.replace("\\end{tabular} \\end{table}", "\\end{tabular}\n\\end{table}")
        generation = generation.replace("\\end{table} Tab", "\\end{table}\nTab")

        generation = re.sub(r"(^.+)\\begin{tab", r"\1\n\\begin{tab", generation, flags=re.M)

        # Remove left-aligned empty LaTeX tabular blocks.
        generation = generation.replace(r"\begin{tabular}{l l}  & \\ \end{tabular}", "")
        # Remove tabulars with just 2 newline characters.
        generation = generation.replace("\\begin{tabular}{}\n\n\\end{tabular}", "")
        return generation

    def post_process_single(self, generation: str, fix_markdown: bool = True) -> str:
        """
        Postprocess a single generated text. Regular expressions used here are taken directly from the Nougat article
        authors. These expressions are commented for clarity and tested end-to-end in most cases.

        Args:
            generation (str): The generated text to be postprocessed.
            fix_markdown (bool, optional): Whether to perform Markdown formatting fixes. Default is True.

        Returns:
            str: The postprocessed text.
        """
        generation = re.sub(
            r"(?:\n|^)#+ \d*\W? ?(.{100,})", r"\n\1", generation
        )  # too long section titles probably are none
        generation = generation.strip()
        # Remove LaTeX left margin tag
        generation = generation.replace("\n* [leftmargin=*]\n", "\n")
        # Remove lines with markdown headings starting with #, with numerals,
        # and possibly roman numerals with trailing spaces and newlines
        generation = re.sub(r"^#+ (?:[\d+\.]+|[ixv\.]+)?\s*(?:$|\n\s*)", "", generation, flags=re.M)
        # most likely hallucinated titles
        lines = generation.split("\n")
        if lines[-1].startswith("#") and lines[-1].lstrip("#").startswith(" ") and len(lines) > 1:
            logger.info("Likely hallucinated title at the end of the page: " + lines[-1])
            generation = "\n".join(lines[:-1])
        # obvious repetition detection
        generation = truncate_repetitions(generation)
        # Reference corrections
        generation = self.remove_hallucinated_references(generation)
        # Remove lines starting with asterisks and numbers like "*[1]" and followed by capital letters and periods (ie too long references)
        generation = re.sub(r"^\* \[\d+\](\s?[A-W]\.+\s?){10,}.*$", "", generation, flags=re.M)
        # Remove empty brackets after a reference number in brackets. *[12][]ABC will become *[12]ABC
        generation = re.sub(r"^(\* \[\d+\])\[\](.*)$", r"\1\2", generation, flags=re.M)
        # Remove single characters before or after 2 new lines
        generation = re.sub(r"(^\w\n\n|\n\n\w$)", "", generation)
        # pmc math artifact correction
        generation = re.sub(
            r"([\s.,()])_([a-zA-Z0-9])__([a-zA-Z0-9]){1,3}_([\s.,:()])",
            r"\1\(\2_{\3}\)\4",
            generation,
        )
        generation = re.sub(r"([\s.,\d])_([a-zA-Z0-9])_([\s.,\d;])", r"\1\(\2\)\3", generation)
        # footnote mistakes
        generation = re.sub(
            r"(\nFootnote .*?:) (?:footnotetext|thanks):\W*(.*(?:\n\n|$))",
            r"\1 \2",
            generation,
        )
        # TODO Come up with footnote formatting inside a table
        generation = re.sub(r"\[FOOTNOTE:.+?\](.*?)\[ENDFOOTNOTE\]", "", generation)
        # itemize post processing
        generation = normalize_list_like_lines(generation)

        if generation.endswith((".", "}")):
            generation += "\n\n"
        if re.match(r"[A-Z0-9,;:]$", generation):
            # add space in case it there is a comma or word ending
            generation += " "
        elif generation.startswith(("#", "**", "\\begin")):
            generation = "\n\n" + generation
        elif generation.split("\n")[-1].startswith(("#", "Figure", "Table")):
            generation = generation + "\n\n"
        else:
            try:
                last_word = generation.split(" ")[-1]
                if last_word in nltk.corpus.words.words():
                    generation += " "
            except LookupError:
                # add space just in case. Will split words but better than concatenating them
                generation += " "

        # table corrections
        generation = self.correct_tables(generation)
        # Remove optional, empty square brackets after begin{array}
        generation = generation.replace("\\begin{array}[]{", "\\begin{array}{")
        # Remove empty or malformed LaTeX tabular blocks with 2 or more columns specified, with spaces and ampersands.
        generation = re.sub(
            r"\\begin{tabular}{([clr ]){2,}}\s*[& ]*\s*(\\\\)? \\end{tabular}",
            "",
            generation,
        )
        # Remove lines containing "S.A.B." one or more times. Was included in Nougat's code.
        generation = re.sub(r"(\*\*S\. A\. B\.\*\*\n+){2,}", "", generation)
        # Remove markdown-style headers that are incomplete or empty on multiple lines.
        generation = re.sub(r"^#+( [\[\d\w])?$", "", generation, flags=re.M)
        # Remove lines with just one period.
        generation = re.sub(r"^\.\s*$", "", generation, flags=re.M)
        # Replace instances of three or more newlines with just two newlines.
        generation = re.sub(r"\n{3,}", "\n\n", generation)
        if fix_markdown:
            return markdown_compatible(generation)
        else:
            return generation

    def post_process_generation(
        self,
        generation: Union[str, List[str]],
        fix_markdown: bool = True,
        num_workers: int = None,
    ) -> Union[str, List[str]]:
        """
        Postprocess a generated text or a list of generated texts.

        This function can be used to perform postprocessing on generated text, such as fixing Markdown formatting.

        Postprocessing is quite slow so it is recommended to use multiprocessing to speed up the process.

        Args:
            generation (Union[str, List[str]]):
                The generated text or a list of generated texts.
            fix_markdown (`bool`, *optional*, defaults to `True`):
                Whether to perform Markdown formatting fixes.
            num_workers (`int`, *optional*):
                Optional number of workers to pass to leverage multiprocessing (postprocessing several texts in
                parallel).

        Returns:
            Union[str, List[str]]: The postprocessed text or list of postprocessed texts.
        """
        requires_backends(self, ["nltk", "levenshtein"])

        if isinstance(generation, list):
            if num_workers is not None and isinstance(num_workers, int):
                with Pool(num_workers) as p:
                    return p.map(partial(self.post_process_single, fix_markdown=fix_markdown), generation)
            else:
                return [self.post_process_single(s, fix_markdown=fix_markdown) for s in generation]
        else:
            return self.post_process_single(generation, fix_markdown=fix_markdown)


__all__ = ["NougatTokenizerFast"]

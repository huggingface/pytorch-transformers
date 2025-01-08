# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
Utility that checks whether the copies defined in the library match the original or not. This includes:
- All code commented with `# Copied from` comments,
- The list of models in the main README.md matches the ones in the localized READMEs,
- Files that are registered as full copies of one another in the `FULL_COPIES` constant of this script.

This also checks the list of models in the README is complete (has all models) and add a line to complete if there is
a model missing.

Use from the root of the repo with:

```bash
python utils/check_copies.py
```

for a check that will error in case of inconsistencies (used by `make repo-consistency`) or

```bash
python utils/check_copies.py --fix_and_overwrite
```

for a check that will fix all inconsistencies automatically (used by `make fix-copies`).
"""

import argparse
import glob
import os
import re
import subprocess
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

from transformers.utils import direct_transformers_import


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_copies.py
TRANSFORMERS_PATH = "src/transformers"
MODEL_TEST_PATH = "tests/models"
PATH_TO_DOCS = "docs/source/en"
REPO_PATH = "."

# Mapping for files that are full copies of others (keys are copies, values the file to keep them up to data with)
FULL_COPIES = {
    "examples/tensorflow/question-answering/utils_qa.py": "examples/pytorch/question-answering/utils_qa.py",
    "examples/flax/question-answering/utils_qa.py": "examples/pytorch/question-answering/utils_qa.py",
}


LOCALIZED_READMES = {
    # If the introduction or the conclusion of the list change, the prompts may need to be updated.
    "README.md": {
        "start_prompt": "🤗 Transformers currently provides the following architectures",
        "end_prompt": "1. Want to contribute a new model?",
        "format_model_list": (
            "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by"
            " {paper_authors}.{supplements}"
        ),
    },
    "README_zh-hans.md": {
        "start_prompt": "🤗 Transformers 目前支持如下的架构",
        "end_prompt": "1. 想要贡献新的模型？",
        "format_model_list": (
            "**[{title}]({model_link})** (来自 {paper_affiliations}) 伴随论文 {paper_title_link} 由 {paper_authors}"
            " 发布。{supplements}"
        ),
    },
    "README_zh-hant.md": {
        "start_prompt": "🤗 Transformers 目前支援以下的架構",
        "end_prompt": "1. 想要貢獻新的模型？",
        "format_model_list": (
            "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by"
            " {paper_authors}.{supplements}"
        ),
    },
    "README_ko.md": {
        "start_prompt": "🤗 Transformers는 다음 모델들을 제공합니다",
        "end_prompt": "1. 새로운 모델을 올리고 싶나요?",
        "format_model_list": (
            "**[{title}]({model_link})** ({paper_affiliations} 에서 제공)은 {paper_authors}.{supplements}의"
            " {paper_title_link}논문과 함께 발표했습니다."
        ),
    },
    "README_es.md": {
        "start_prompt": "🤗 Transformers actualmente proporciona las siguientes arquitecturas",
        "end_prompt": "1. ¿Quieres aportar un nuevo modelo?",
        "format_model_list": (
            "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by"
            " {paper_authors}.{supplements}"
        ),
    },
    "README_ja.md": {
        "start_prompt": "🤗Transformersは現在、以下のアーキテクチャを提供しています",
        "end_prompt": "1. 新しいモデルを投稿したいですか？",
        "format_model_list": (
            "**[{title}]({model_link})** ({paper_affiliations} から) {paper_authors}.{supplements} から公開された研究論文"
            " {paper_title_link}"
        ),
    },
    "README_hd.md": {
        "start_prompt": "🤗 ट्रांसफॉर्मर वर्तमान में निम्नलिखित आर्किटेक्चर का समर्थन करते हैं",
        "end_prompt": "1. एक नए मॉडल में योगदान देना चाहते हैं?",
        "format_model_list": (
            "**[{title}]({model_link})** ({paper_affiliations} से) {paper_authors}.{supplements} द्वारा"
            "अनुसंधान पत्र {paper_title_link} के साथ जारी किया गया"
        ),
    },
    "README_ru.md": {
        "start_prompt": "🤗 В настоящее время Transformers предоставляет следующие архитектуры",
        "end_prompt": "1. Хотите внести новую модель?",
        "format_model_list": (
            "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by"
            " {paper_authors}.{supplements}"
        ),
    },
    "README_pt-br.md": {
        "start_prompt": "🤗 Transformers atualmente fornece as seguintes arquiteturas",
        "end_prompt": "1. Quer contribuir com um novo modelo?",
        "format_model_list": (
            "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by"
            " {paper_authors}.{supplements}"
        ),
    },
    "README_te.md": {
        "start_prompt": "🤗 ట్రాన్స్‌ఫార్మర్లు ప్రస్తుతం కింది ఆర్కిటెక్చర్‌లను అందజేస్తున్నాయి",
        "end_prompt": "1. కొత్త మోడల్‌ను అందించాలనుకుంటున్నారా?",
        "format_model_list": (
            "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by"
            " {paper_authors}.{supplements}"
        ),
    },
    "README_fr.md": {
        "start_prompt": "🤗 Transformers fournit actuellement les architectures suivantes",
        "end_prompt": "1. Vous souhaitez contribuer avec un nouveau modèle ?",
        "format_model_list": (
            "**[{title}]({model_link})** (de {paper_affiliations}) publié dans l'article {paper_title_link} par"
            "{paper_authors}.{supplements}"
        ),
    },
    "README_de.md": {
        "start_prompt": "🤗 Transformers bietet derzeit die folgenden Architekturen an",
        "end_prompt": "1. Möchten Sie ein neues Modell beitragen?",
        "format_model_list": (
            "**[{title}]({model_link})** (from {paper_affiliations}) released with the paper {paper_title_link} by"
            " {paper_authors}.{supplements}"
        ),
    },
    "README_vi.md": {
        "start_prompt": "🤗 Transformers hiện đang cung cấp các kiến trúc sau đây",
        "end_prompt": "1. Muốn đóng góp một mô hình mới?",
        "format_model_list": (
            "**[{title}]({model_link})** (từ {paper_affiliations}) được phát hành với bài báo {paper_title_link} by"
            " {paper_authors}.{supplements}"
        ),
    },
}

# This is to make sure the transformers module imported is the one in the repo.
transformers_module = direct_transformers_import(TRANSFORMERS_PATH)


def _is_definition_header_ending_line(line: str) -> bool:
    # Helper function. Returns `True` if `line` is the end parenthesis of a class/function definition
    return re.search(r"^\s*\)(\s*->.*:|:)\s*$", line) is not None


def _should_continue(line: str, indent: str) -> bool:
    # Helper function. Returns `True` if `line` is empty, starts with the `indent` or is the end parenthesis of a
    # class/function definition
    return line.startswith(indent) or len(line.strip()) == 0 or _is_definition_header_ending_line(line)


def _sanity_check_splits(splits_1, splits_2, is_class, filename):
    """Check the two (inner) block structures of the corresponding code block given by `split_code_into_blocks` match.

    For the case of `class`, they must be of one of the following 3 cases:

        - a single block without name:

            class foo:
                a = 1

        - a consecutive sequence of (1 or more) blocks with name

            class foo:

                def f(x):
                    return x

        - a block without name, followed by a consecutive sequence of (1 or more) blocks with name

            class foo:
                a = 1

                def f(x):
                    return x

                def g(x):
                    return None

    The 2 code snippets that give `splits_1` and `splits_2` have to be in the same case to pass this check, but the
    number of blocks with name in the consecutive sequence is not taken into account.

    For the case of `function or method`, we don't require it to be in one of the above 3 cases. However, the structure
    of`splits_1` and `splits_2` have to match exactly. In particular, the number of blocks with name in a consecutive
    sequence is taken into account.
    """
    block_names_1 = []
    block_names_2 = []

    for block in splits_1[1:]:
        if block[0].startswith("_block_without_name_"):
            block_names_1.append("block_without_name")
        elif not block[0].startswith("_empty_block_") and (
            not is_class or len(block_names_1) == 0 or block_names_1[-1].startswith("block_without_name")
        ):
            block_names_1.append("block_with_name")

    for block in splits_2[1:]:
        if block[0].startswith("_block_without_name_"):
            block_names_2.append("block_without_name")
        elif not block[0].startswith("_empty_block_") and (
            not is_class or len(block_names_2) == 0 or block_names_2[-1].startswith("block_without_name")
        ):
            block_names_2.append("block_with_name")

    if is_class:
        if block_names_1 not in [
            ["block_without_name"],
            ["block_with_name"],
            ["block_without_name", "block_with_name"],
        ]:
            raise ValueError(
                f"""Class defined in {filename} doesn't have the expected stucture.
                See the docstring of `_sanity_check_splits` in the file `utils/check_copies.py`""",
            )

    if block_names_1 != block_names_2:
        raise ValueError(f"In {filename}, two code blocks expected to be copies have different structures.")


def find_block_end(lines: List[str], start_index: int, indent: int) -> int:
    """
    Find the end of the class/func block starting at `start_index` in a source code (defined by `lines`).

    Args:
        lines (`List[str]`):
            The source code, represented by a list of lines.
        start_index (`int`):
            The starting index of the target class/func block.
        indent (`int`):
            The indent of the class/func body.

    Returns:
        `int`: The index of the block's ending line plus by 1 (i.e. exclusive).
    """
    indent = " " * indent
    # enter the block body
    line_index = start_index + 1

    while line_index < len(lines) and _should_continue(lines[line_index], indent):
        line_index += 1
    # Clean up empty lines at the end (if any).
    while len(lines[line_index - 1]) <= 1:
        line_index -= 1

    return line_index


def split_code_into_blocks(
    lines: List[str], start_index: int, end_index: int, indent: int, backtrace: bool = False
) -> List[Tuple[str, int, int]]:
    """
    Split the class/func block starting at `start_index` in a source code (defined by `lines`) into *inner blocks*.

    The block's header is included as the first element. The contiguous regions (without empty lines) that are not
    inside any inner block are included as blocks. The contiguous regions of empty lines that are not inside any inner
    block are also included as (dummy) blocks.

    Args:
        lines (`List[str]`):
            The source code, represented by a list of lines.
        start_index (`int`):
            The starting index of the target class/func block.
        end_index (`int`):
            The ending index of the target class/func block.
        indent (`int`):
            The indent of the class/func body.
        backtrace (`bool`, *optional*, defaults to `False`):
            Whether or not to include the lines before the inner class/func block's header (e.g. comments, decorators,
            etc.) until an empty line is encountered.

    Returns:
        `List[Tuple[str, int, int]]`: A list of elements with the form `(block_name, start_index, end_index)`.
    """
    splits = []
    # `indent - 4` is the indent level of the target class/func header
    try:
        target_block_name = re.search(
            rf"^{' ' * (indent - 4)}((class|def)\s+\S+)(\(|\:)", lines[start_index]
        ).groups()[0]
    except Exception:
        start_context = min(start_index - 10, 0)
        end_context = min(end_index + 10, len(lines))
        raise ValueError(
            f"Tried to split a class or function. It did not work. Error comes from line {start_index}: \n```\n"
            + "".join(lines[start_context:end_context])
            + "```\n"
        )

    # from now on, the `block` means inner blocks unless explicitly specified
    indent_str = " " * indent
    block_without_name_idx = 0
    empty_block_idx = 0

    # Find the lines for the definition header
    index = start_index
    if "(" in lines[start_index] and "):" not in lines[start_index] in lines[start_index]:
        while index < end_index:
            if _is_definition_header_ending_line(lines[index]):
                break
            index += 1

    # the first line outside the definition header
    index += 1
    splits.append((target_block_name, start_index, index))

    block_start_index, prev_block_end_index = index, index
    while index < end_index:
        # if found, it will be an inner block
        block_found = re.search(rf"^{indent_str}((class|def)\s+\S+)(\(|\:)", lines[index])
        if block_found:
            name = block_found.groups()[0]

            block_end_index = find_block_end(lines, index, indent + 4)

            # backtrace to include the lines before the found block's definition header (e.g. comments, decorators,
            # etc.) until an empty line is encountered.
            block_start_index = index
            if index > prev_block_end_index and backtrace:
                idx = index - 1
                for idx in range(index - 1, prev_block_end_index - 2, -1):
                    if not (len(lines[idx].strip()) > 0 and lines[idx].startswith(indent_str)):
                        break
                idx += 1
                if idx < index:
                    block_start_index = idx

            # between the current found block and the previous found block
            if block_start_index > prev_block_end_index:
                # give it a dummy name
                if len("".join(lines[prev_block_end_index:block_start_index]).strip()) == 0:
                    prev_block_name = f"_empty_block_{empty_block_idx}"
                    empty_block_idx += 1
                else:
                    prev_block_name = f"_block_without_name_{block_without_name_idx}"
                    block_without_name_idx += 1
                # Add it as a block
                splits.append((prev_block_name, prev_block_end_index, block_start_index))

            # Add the current found block
            splits.append((name, block_start_index, block_end_index))
            prev_block_end_index = block_end_index
            index = block_end_index - 1

        index += 1

    if index > prev_block_end_index:
        if len("".join(lines[prev_block_end_index:index]).strip()) == 0:
            prev_block_name = f"_empty_block_{empty_block_idx}"
        else:
            prev_block_name = f"_block_without_name_{block_without_name_idx}"
        splits.append((prev_block_name, prev_block_end_index, index))

    return splits


def find_code_in_transformers(
    object_name: str, base_path: str = None, return_indices: bool = False
) -> Union[str, Tuple[List[str], int, int]]:
    """
    Find and return the source code of an object.

    Args:
        object_name (`str`):
            The name of the object we want the source code of.
        base_path (`str`, *optional*):
            The path to the base folder where files are checked. If not set, it will be set to `TRANSFORMERS_PATH`.
        return_indices(`bool`, *optional*, defaults to `False`):
            If `False`, will only return the code (as a string), otherwise it will also return the whole lines of the
            file where the object specified by `object_name` is defined, together the start/end indices of the block in
            the file that defines the object.

    Returns:
        `Union[str, Tuple[List[str], int, int]]`: If `return_indices=False`, only the source code of the object will be
        returned. Otherwise, it also returns the whole lines of the file where the object specified by `object_name` is
        defined, together the start/end indices of the block in the file that defines the object.
    """
    parts = object_name.split(".")
    i = 0

    # We can't set this as the default value in the argument, otherwise `CopyCheckTester` will fail, as it uses a
    # patched temp directory.
    if base_path is None:
        base_path = TRANSFORMERS_PATH

    # Detail: the `Copied from` statement is originally designed to work with the last part of `TRANSFORMERS_PATH`,
    # (which is `transformers`). The same should be applied for `MODEL_TEST_PATH`. However, its last part is `models`
    # (to only check and search in it) which is a bit confusing. So we keep the copied statement staring with
    # `tests.models.` and change it to `tests` here.
    if base_path == MODEL_TEST_PATH:
        base_path = "tests"

    # First let's find the module where our object lives.
    module = parts[i]
    while i < len(parts) and not os.path.isfile(os.path.join(base_path, f"{module}.py")):
        i += 1
        if i < len(parts):
            module = os.path.join(module, parts[i])
    if i >= len(parts):
        raise ValueError(
            f"`object_name` should begin with the name of a module of transformers but got {object_name}."
        )

    with open(os.path.join(base_path, f"{module}.py"), "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()

    # Now let's find the class / func in the code!
    indent = ""
    line_index = 0
    for name in parts[i + 1 :]:
        while (
            line_index < len(lines) and re.search(rf"^{indent}(class|def)\s+{name}(\(|\:)", lines[line_index]) is None
        ):
            line_index += 1
        # find the target specified in the current level in `parts` -> increase `indent` so we can search the next
        indent += "    "
        # the index of the first line in the (currently found) block *body*
        line_index += 1

    if line_index >= len(lines):
        raise ValueError(f" {object_name} does not match any function or class in {module}.")

    # `indent` is already one level deeper than the (found) class/func block's definition header

    # We found the beginning of the class / func, now let's find the end (when the indent diminishes).
    # `start_index` is the index of the class/func block's definition header
    start_index = line_index - 1
    end_index = find_block_end(lines, start_index, len(indent))

    code = "".join(lines[start_index:end_index])
    return (code, (lines, start_index, end_index)) if return_indices else code


def replace_code(code: str, replace_pattern: str) -> str:
    """Replace `code` by a pattern of the form `with X1->X2,Y1->Y2,Z1->Z2`.

    Args:
        code (`str`): The code to be modified.
        replace_pattern (`str`): The pattern used to modify `code`.

    Returns:
        `str`: The modified code.
    """
    if len(replace_pattern) > 0:
        patterns = replace_pattern.replace("with", "").split(",")
        patterns = [_re_replace_pattern.search(p) for p in patterns]
        for pattern in patterns:
            if pattern is None:
                continue
            obj1, obj2, option = pattern.groups()
            code = re.sub(obj1, obj2, code)
            if option.strip() == "all-casing":
                code = re.sub(obj1.lower(), obj2.lower(), code)
                code = re.sub(obj1.upper(), obj2.upper(), code)

    return code


def find_code_and_splits(object_name: str, base_path: str, buffer: dict = None):
    """Find the code of an object (specified by `object_name`) and split it into blocks.

    Args:
        object_name (`str`):
            The name of the object, e.g. `transformers.models.bert.modeling_bert.BertAttention` or
            `tests.models.llama.test_modeling_llama.LlamaModelTest.test_config`.
        base_path (`str`):
            The path to the base directory within which the search will be performed. It could be either
            `TRANSFORMERS_PATH` or `MODEL_TEST_PATH`.
        buffer (`dict`, *optional*):
            The buffer used to store the previous results in order to speed up the process.

    Returns:
        lines (`List[str]`):
            The lines of the whole file where the object is defined.
        code (`str`):
            The object's code.
        code_splits (`List[Tuple[str, int, int]]`):
            `code` splitted into blocks. See `split_code_into_blocks`.
    """
    if buffer is None:
        buffer = {}

    if (object_name, base_path) in buffer:
        lines, code, code_splits = buffer[(object_name, base_path)]
    else:
        code, (lines, target_start_index, target_end_index) = find_code_in_transformers(
            object_name, base_path=base_path, return_indices=True
        )
        indent = get_indent(code)

        # Split the code into blocks
        # `indent` is the indent of the class/func definition header, but `code_splits` expects the indent level of the
        # block body.
        code_splits = split_code_into_blocks(
            lines, target_start_index, target_end_index, len(indent) + 4, backtrace=True
        )
        buffer[(object_name, base_path)] = lines, code, code_splits

    return lines, code, code_splits


_re_copy_warning = re.compile(r"^(\s*)#\s*Copied from\s+transformers\.(\S+\.\S+)\s*($|\S.*$)")
_re_copy_warning_for_test_file = re.compile(r"^(\s*)#\s*Copied from\s+tests\.(\S+\.\S+)\s*($|\S.*$)")
_re_replace_pattern = re.compile(r"^\s*(\S+)->(\S+)(\s+.*|$)")
_re_fill_pattern = re.compile(r"<FILL\s+[^>]*>")


def get_indent(code: str) -> str:
    """
    Find the indent in the first non empty line in a code sample.

    Args:
        code (`str`): The code to inspect.

    Returns:
        `str`: The indent looked at (as string).
    """
    lines = code.split("\n")
    idx = 0
    while idx < len(lines) and len(lines[idx]) == 0:
        idx += 1
    if idx < len(lines):
        return re.search(r"^(\s*)\S", lines[idx]).groups()[0]
    return ""


def run_ruff(code, check=False):
    if check:
        command = ["ruff", "check", "-", "--fix", "--exit-zero"]
    else:
        command = ["ruff", "format", "-", "--config", "pyproject.toml", "--silent"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    stdout, _ = process.communicate(input=code.encode())
    return stdout.decode()


def stylify(code: str) -> str:
    """
    Applies the ruff part of our `make style` command to some code. This formats the code using `ruff format`.
    As `ruff` does not provide a python api this cannot be done on the fly.

    Args:
        code (`str`): The code to format.

    Returns:
        `str`: The formatted code.
    """
    has_indent = len(get_indent(code)) > 0
    if has_indent:
        code = f"class Bla:\n{code}"
    formatted_code = run_ruff(code)
    return formatted_code[len("class Bla:\n") :] if has_indent else formatted_code


def check_codes_match(observed_code: str, theoretical_code: str) -> Optional[int]:
    """
    Checks if two version of a code match with the exception of the class/function name.

    Args:
        observed_code (`str`): The code found.
        theoretical_code (`str`): The code to match.

    Returns:
        `Optional[int]`: The index of the first line where there is a difference (if any) and `None` if the codes
        match.
    """
    observed_code_header = observed_code.split("\n")[0]
    theoretical_code_header = theoretical_code.split("\n")[0]

    # Catch the function/class name: it is expected that those do not match.
    _re_class_match = re.compile(r"class\s+([^\(:]+)(?:\(|:)")
    _re_func_match = re.compile(r"def\s+([^\(]+)\(")
    for re_pattern in [_re_class_match, _re_func_match]:
        if re_pattern.match(observed_code_header) is not None:
            try:
                observed_obj_name = re_pattern.search(observed_code_header).groups()[0]
            except Exception:
                raise ValueError(
                    "Tried to split a class or function. It did not work. Error comes from: \n```\n"
                    + observed_code_header
                    + "\n```\n"
                )

            try:
                theoretical_name = re_pattern.search(theoretical_code_header).groups()[0]
            except Exception:
                raise ValueError(
                    "Tried to split a class or function. It did not work. Error comes from: \n```\n"
                    + theoretical_code_header
                    + "\n```\n"
                )
            theoretical_code_header = theoretical_code_header.replace(theoretical_name, observed_obj_name)

    # Find the first diff. Line 0 is special since we need to compare with the function/class names ignored.
    diff_index = 0
    if theoretical_code_header != observed_code_header:
        return 0

    diff_index = 1
    for observed_line, theoretical_line in zip(observed_code.split("\n")[1:], theoretical_code.split("\n")[1:]):
        if observed_line != theoretical_line:
            return diff_index
        diff_index += 1


def is_copy_consistent(filename: str, overwrite: bool = False, buffer: dict = None) -> Optional[List[Tuple[str, int]]]:
    """
    Check if the code commented as a copy in a file matches the original.

    Args:
        filename (`str`):
            The name of the file to check.
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the copies when they don't match.
        buffer (`dict`, *optional*):
            The buffer used to store the previous results in order to speed up the process.

    Returns:
        `Optional[List[Tuple[str, int]]]`: If `overwrite=False`, returns the list of differences as tuples `(str, int)`
        with the name of the object having a diff and the line number where theere is the first diff.
    """
    base_path = TRANSFORMERS_PATH if not filename.startswith("tests") else MODEL_TEST_PATH

    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    diffs = []
    line_index = 0
    # Not a for loop cause `lines` is going to change (if `overwrite=True`).
    search_re = _re_copy_warning_for_test_file if filename.startswith("tests") else _re_copy_warning
    while line_index < len(lines):
        search = search_re.search(lines[line_index])
        if search is None:
            line_index += 1
            continue

        # There is some copied code here, let's retrieve the original.
        indent, object_name, replace_pattern = search.groups()

        # Find the file lines, the object's code, and its blocks
        try:
            target_lines, theoretical_code, theoretical_code_splits = find_code_and_splits(
                object_name, base_path, buffer=buffer
            )
        except Exception as exc:
            exc.args = (f"Error while trying to find source code for {filename}.\n\n" + str(exc),)
            raise

        # code replaced by the patterns
        theoretical_code_blocks = OrderedDict()
        for name, start, end in theoretical_code_splits:
            name = replace_code(name, replace_pattern)
            code = "".join(target_lines[start:end])
            code = replace_code(code, replace_pattern)
            theoretical_code_blocks[name] = code

        theoretical_indent = get_indent(theoretical_code)

        # `start_index` is the index of the first line (the definition header) after `# Copied from`.
        # (`indent != theoretical_indent` doesn't seem to occur so far, not sure what this case is for.)
        start_index = line_index + 1 if indent == theoretical_indent else line_index
        # enter the block body
        line_index = start_index + 1

        subcode = "\n".join(theoretical_code.split("\n")[1:])
        indent = get_indent(subcode)
        # Loop to check the observed code, stop when indentation diminishes or if we see a End copy comment.
        # We can't call `find_block_end` directly as there is sth. special `# End copy"` here.
        should_continue = True
        while line_index < len(lines) and should_continue:
            line_index += 1
            if line_index >= len(lines):
                break
            line = lines[line_index]
            # There is a special pattern `# End copy` to stop early. It's not documented cause it shouldn't really be
            # used.
            should_continue = _should_continue(line, indent) and re.search(f"^{indent}# End copy", line) is None
        # `line_index` is outside the block
        # Clean up empty lines at the end (if any).
        while len(lines[line_index - 1]) <= 1:
            line_index -= 1

        # Split the observed code into blocks
        observed_code_splits = split_code_into_blocks(lines, start_index, line_index, len(indent), backtrace=True)

        is_class = lines[start_index].startswith(f"{' ' * (len(indent) - 4)}class ")
        # sanity check
        _sanity_check_splits(theoretical_code_splits, observed_code_splits, is_class=is_class, filename=filename)

        # observed code in a structured way (a dict mapping block names to blocks' code)
        observed_code_blocks = OrderedDict()
        for name, start, end in observed_code_splits:
            code = "".join(lines[start:end])
            observed_code_blocks[name] = code

        # Below, we change some names in `theoretical_code_blocks` and `observed_code_blocks`. These mappings map the
        # original names to the modified names: this is used to restore the original order of the code blocks.
        name_mappings_1 = {k: k for k in theoretical_code_blocks.keys()}
        name_mappings_2 = {k: k for k in observed_code_blocks.keys()}

        # Update code blocks' name and content:
        #   If `"# Ignore copy"` is found in a block of the observed code:
        #     1. if it's a block only in the observed code --> add it to the theoretical code.
        #     2. if it's also in the theoretical code () --> put its content (body) to the corresponding block under the
        #        same name in the theoretical code.
        #   In both cases, we change the name to have a prefix `_ignored_` so we know if we can discard them during the
        #   comparison.
        ignored_existing_block_index = 0
        ignored_new_block_index = 0
        for name in list(observed_code_blocks.keys()):
            code = observed_code_blocks[name]
            if "# Ignore copy" in code:
                if name in theoretical_code_blocks:
                    # in the target --> just copy the content
                    del theoretical_code_blocks[name]
                    theoretical_code_blocks[f"_ignored_existing_block_{ignored_existing_block_index}"] = code
                    name_mappings_1[name] = f"_ignored_existing_block_{ignored_existing_block_index}"

                    del observed_code_blocks[name]
                    observed_code_blocks[f"_ignored_existing_block_{ignored_existing_block_index}"] = code
                    name_mappings_2[name] = f"_ignored_existing_block_{ignored_existing_block_index}"
                    ignored_existing_block_index += 1
                else:
                    # not in the target --> add it
                    theoretical_code_blocks[f"_ignored_new_block_{ignored_new_block_index}"] = code
                    name_mappings_1[f"_ignored_new_block_{ignored_new_block_index}"] = (
                        f"_ignored_new_block_{ignored_new_block_index}"
                    )

                    del observed_code_blocks[name]
                    observed_code_blocks[f"_ignored_new_block_{ignored_new_block_index}"] = code
                    name_mappings_2[name] = f"_ignored_new_block_{ignored_new_block_index}"
                    ignored_new_block_index += 1

        # Respect the original block order:
        #   1. in `theoretical_code_blocks`: the new blocks will follow the existing ones
        #   2. in `observed_code_blocks`: the original order are kept with names modified potentially. This is necessary
        #      to compute the correct `diff_index` if `overwrite=True` and there is a diff.
        theoretical_code_blocks = {
            name_mappings_1[orig_name]: theoretical_code_blocks[name_mappings_1[orig_name]]
            for orig_name in name_mappings_1
        }
        observed_code_blocks = {
            name_mappings_2[orig_name]: observed_code_blocks[name_mappings_2[orig_name]]
            for orig_name in name_mappings_2
        }

        # Ignore the blocks specified to be ignored. This is the version used to check if there is a mismatch
        theoretical_code_blocks_clean = {
            k: v
            for k, v in theoretical_code_blocks.items()
            if not (k.startswith(("_ignored_existing_block_", "_ignored_new_block_")))
        }
        theoretical_code = "".join(list(theoretical_code_blocks_clean.values()))

        # stylify `theoretical_code` before compare (this is needed only when `replace_pattern` is not empty)
        if replace_pattern:
            theoretical_code = stylify(theoretical_code)
        # Remove `\n\n` in `theoretical_code` before compare (so no empty line)
        while "\n\n" in theoretical_code:
            theoretical_code = theoretical_code.replace("\n\n", "\n")

        # Compute `observed_code` where we don't include any empty line + keep track the line index between the
        # original/processed `observed_code` so we can have the correct `diff_index`.
        idx_to_orig_idx_mapping_for_observed_code_lines = {}
        idx = -1
        orig_idx = -1
        observed_code = ""
        for name, code in observed_code_blocks.items():
            if code.endswith("\n"):
                code = code[:-1]
            for code_line in code.split("\n"):
                orig_idx += 1
                if code_line.strip() and not name.startswith(("_ignored_existing_block_", "_ignored_new_block_")):
                    idx += 1
                    observed_code += code_line + "\n"
                    idx_to_orig_idx_mapping_for_observed_code_lines[idx] = orig_idx

        # Test for a diff and act accordingly.
        diff_index = check_codes_match(observed_code, theoretical_code)
        if diff_index is not None:
            # switch to the index in the original `observed_code` (i.e. before removing empty lines)
            diff_index = idx_to_orig_idx_mapping_for_observed_code_lines[diff_index]
            diffs.append([object_name, diff_index + start_index + 1])
            if overwrite:
                # `theoretical_code_to_write` is a single string but may have several lines.
                theoretical_code_to_write = stylify("".join(list(theoretical_code_blocks.values())))
                lines = lines[:start_index] + [theoretical_code_to_write] + lines[line_index:]
                # Here we treat it as a single entry in `lines`.
                line_index = start_index + 1

    if overwrite and len(diffs) > 0:
        # Warn the user a file has been modified.
        print(f"Detected changes, rewriting {filename}.")
        with open(filename, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines)
    return diffs


def check_copies(overwrite: bool = False, file: str = None):
    """
    Check every file is copy-consistent with the original. Also check the model list in the main README and other
    READMEs are consistent.

    Args:
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the copies when they don't match.
        file (`bool`, *optional*):
            The path to a specific file to check and/or fix.
    """
    buffer = {}

    if file is None:
        all_files = glob.glob(os.path.join(TRANSFORMERS_PATH, "**/*.py"), recursive=True)
        all_test_files = glob.glob(os.path.join(MODEL_TEST_PATH, "**/*.py"), recursive=True)
        all_files = list(all_files) + list(all_test_files)
    else:
        all_files = [file]

    diffs = []
    for filename in all_files:
        new_diffs = is_copy_consistent(filename, overwrite, buffer)
        diffs += [f"- {filename}: copy does not match {d[0]} at line {d[1]}" for d in new_diffs]
    if not overwrite and len(diffs) > 0:
        diff = "\n".join(diffs)
        raise Exception(
            "Found the following copy inconsistencies:\n"
            + diff
            + "\nRun `make fix-copies` or `python utils/check_copies.py --fix_and_overwrite` to fix them."
        )


def check_full_copies(overwrite: bool = False):
    """
    Check the files that are full copies of others (as indicated in `FULL_COPIES`) are copy-consistent.

    Args:
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether or not to overwrite the copies when they don't match.
    """
    diffs = []
    for target, source in FULL_COPIES.items():
        with open(source, "r", encoding="utf-8") as f:
            source_code = f.read()
        with open(target, "r", encoding="utf-8") as f:
            target_code = f.read()
        if source_code != target_code:
            if overwrite:
                with open(target, "w", encoding="utf-8") as f:
                    print(f"Replacing the content of {target} by the one of {source}.")
                    f.write(source_code)
            else:
                diffs.append(f"- {target}: copy does not match {source}.")

    if not overwrite and len(diffs) > 0:
        diff = "\n".join(diffs)
        raise Exception(
            "Found the following copy inconsistencies:\n"
            + diff
            + "\nRun `make fix-copies` or `python utils/check_copies.py --fix_and_overwrite` to fix them."
        )


def get_model_list(filename: str, start_prompt: str, end_prompt: str) -> str:
    """
    Extracts the model list from a README.

    Args:
        filename (`str`): The name of the README file to check.
        start_prompt (`str`): The string to look for that introduces the model list.
        end_prompt (`str`): The string to look for that ends the model list.

    Returns:
        `str`: The model list.
    """
    with open(os.path.join(REPO_PATH, filename), "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    # Find the start of the list.
    start_index = 0
    while not lines[start_index].startswith(start_prompt):
        start_index += 1
    start_index += 1

    result = []
    current_line = ""
    end_index = start_index

    # Keep going until the end of the list.
    while not lines[end_index].startswith(end_prompt):
        if lines[end_index].startswith("1."):
            if len(current_line) > 1:
                result.append(current_line)
            current_line = lines[end_index]
        elif len(lines[end_index]) > 1:
            current_line = f"{current_line[:-1]} {lines[end_index].lstrip()}"
        end_index += 1
    if len(current_line) > 1:
        result.append(current_line)

    return "".join(result)


def convert_to_localized_md(model_list: str, localized_model_list: str, format_str: str) -> Tuple[bool, str]:
    """
    Compare the model list from the main README to the one in a localized README.

    Args:
        model_list (`str`): The model list in the main README.
        localized_model_list (`str`): The model list in one of the localized README.
        format_str (`str`):
            The template for a model entry in the localized README (look at the `format_model_list` in the entries of
            `LOCALIZED_READMES` for examples).

    Returns:
        `Tuple[bool, str]`: A tuple where the first value indicates if the READMEs match or not, and the second value
        is the correct localized README.
    """

    def _rep(match):
        title, model_link, paper_affiliations, paper_title_link, paper_authors, supplements = match.groups()
        return format_str.format(
            title=title,
            model_link=model_link,
            paper_affiliations=paper_affiliations,
            paper_title_link=paper_title_link,
            paper_authors=paper_authors,
            supplements=" " + supplements.strip() if len(supplements) != 0 else "",
        )

    # This regex captures metadata from an English model description, including model title, model link,
    # affiliations of the paper, title of the paper, authors of the paper, and supplemental data (see DistilBERT for
    # example).
    _re_capture_meta = re.compile(
        r"\*\*\[([^\]]*)\]\(([^\)]*)\)\*\* \(from ([^)]*)\)[^\[]*([^\)]*\)).*?by (.*?[A-Za-z\*]{2,}?)\. (.*)$"
    )
    # This regex is used to synchronize title link.
    _re_capture_title_link = re.compile(r"\*\*\[([^\]]*)\]\(([^\)]*)\)\*\*")
    # This regex is used to synchronize paper title and link.
    _re_capture_paper_link = re.compile(r" \[([^\]]*)\]\(([^\)]*)\)")

    if len(localized_model_list) == 0:
        localized_model_index = {}
    else:
        try:
            localized_model_index = {
                re.search(r"\*\*\[([^\]]*)", line).groups()[0]: line
                for line in localized_model_list.strip().split("\n")
            }
        except AttributeError:
            raise AttributeError("A model name in localized READMEs cannot be recognized.")

    model_keys = [re.search(r"\*\*\[([^\]]*)", line).groups()[0] for line in model_list.strip().split("\n")]

    # We exclude keys in localized README not in the main one.
    readmes_match = not any(k not in model_keys for k in localized_model_index)
    localized_model_index = {k: v for k, v in localized_model_index.items() if k in model_keys}

    for model in model_list.strip().split("\n"):
        title, model_link = _re_capture_title_link.search(model).groups()
        if title not in localized_model_index:
            readmes_match = False
            # Add an anchor white space behind a model description string for regex.
            # If metadata cannot be captured, the English version will be directly copied.
            localized_model_index[title] = _re_capture_meta.sub(_rep, model + " ")
        elif _re_fill_pattern.search(localized_model_index[title]) is not None:
            update = _re_capture_meta.sub(_rep, model + " ")
            if update != localized_model_index[title]:
                readmes_match = False
                localized_model_index[title] = update
        else:
            # Synchronize title link
            converted_model = _re_capture_title_link.sub(
                f"**[{title}]({model_link})**", localized_model_index[title], count=1
            )

            # Synchronize paper title and its link (if found)
            paper_title_link = _re_capture_paper_link.search(model)
            if paper_title_link is not None:
                paper_title, paper_link = paper_title_link.groups()
                converted_model = _re_capture_paper_link.sub(
                    f" [{paper_title}]({paper_link})", converted_model, count=1
                )

            if converted_model != localized_model_index[title]:
                readmes_match = False
                localized_model_index[title] = converted_model

    sorted_index = sorted(localized_model_index.items(), key=lambda x: x[0].lower())

    return readmes_match, "\n".join((x[1] for x in sorted_index)) + "\n"


def _find_text_in_file(filename: str, start_prompt: str, end_prompt: str) -> Tuple[str, int, int, List[str]]:
    """
    Find the text in a file between two prompts.

    Args:
        filename (`str`): The name of the file to look into.
        start_prompt (`str`): The string to look for that introduces the content looked for.
        end_prompt (`str`): The string to look for that ends the content looked for.

    Returns:
        Tuple[str, int, int, List[str]]: The content between the two prompts, the index of the start line in the
        original file, the index of the end line in the original file and the list of lines of that file.
    """
    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    # Find the start prompt.
    start_index = 0
    while not lines[start_index].startswith(start_prompt):
        start_index += 1
    start_index += 1

    end_index = start_index
    while not lines[end_index].startswith(end_prompt):
        end_index += 1
    end_index -= 1

    while len(lines[start_index]) <= 1:
        start_index += 1
    while len(lines[end_index]) <= 1:
        end_index -= 1
    end_index += 1
    return "".join(lines[start_index:end_index]), start_index, end_index, lines


# Map a model name with the name it has in the README for the check_readme check
SPECIAL_MODEL_NAMES = {
    "Bert Generation": "BERT For Sequence Generation",
    "BigBird": "BigBird-RoBERTa",
    "Data2VecAudio": "Data2Vec",
    "Data2VecText": "Data2Vec",
    "Data2VecVision": "Data2Vec",
    "DonutSwin": "Swin Transformer",
    "Marian": "MarianMT",
    "MaskFormerSwin": "Swin Transformer",
    "OpenAI GPT-2": "GPT-2",
    "OpenAI GPT": "GPT",
    "Perceiver": "Perceiver IO",
    "SAM": "Segment Anything",
    "ViT": "Vision Transformer (ViT)",
}

# Update this list with the models that shouldn't be in the README. This only concerns modular models or those who do
# not have an associated paper.
MODELS_NOT_IN_README = [
    "BertJapanese",
    "Encoder decoder",
    "FairSeq Machine-Translation",
    "HerBERT",
    "RetriBERT",
    "Speech Encoder decoder",
    "Speech2Text",
    "Speech2Text2",
    "TimmBackbone",
    "Vision Encoder decoder",
    "VisionTextDualEncoder",
    "CLIPVisionModel",
    "SiglipVisionModel",
    "ChineseCLIPVisionModel",
    "VitPoseBackbone",
]

# Template for new entries to add in the main README when we have missing models.
README_TEMPLATE = (
    "1. **[{model_name}](https://huggingface.co/docs/main/transformers/model_doc/{model_type})** (from "
    "<FILL INSTITUTION>) released with the paper [<FILL PAPER TITLE>](<FILL ARKIV LINK>) by <FILL AUTHORS>."
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="A specific file to check and/or fix")
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_copies(args.fix_and_overwrite, args.file)
    check_full_copies(args.fix_and_overwrite)

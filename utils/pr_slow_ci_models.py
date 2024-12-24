# Copyright 2024 The HuggingFace Team. All rights reserved.
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
This script is used to get the models for which to run slow CI.

A new model added in a pull request will be included, as well as models specified in a GitHub pull request's comment
with a prefix `run-slow`, `run_slow` or `run slow`. For example, the commit message `run_slow: bert, gpt2` will give
`bert` and `gpt2`.

Usage:

```bash
python utils/pr_slow_ci_models.py
```
"""

import argparse
import os.path
import re
import string
from pathlib import Path
from typing import List

from git import Repo


PATH_TO_REPO = Path(__file__).parent.parent.resolve()


def get_new_python_files_between_commits(base_commit: str, commits: List[str]) -> List[str]:
    """
    Get the list of added python files between a base commit and one or several commits.

    Args:
        repo (`git.Repo`):
            A git repository (for instance the Transformers repo).
        base_commit (`str`):
            The commit reference of where to compare for the diff. This is the current commit, not the branching point!
        commits (`List[str]`):
            The list of commits with which to compare the repo at `base_commit` (so the branching point).

    Returns:
        `List[str]`: The list of python files added between a base commit and one or several commits.
    """
    code_diff = []
    for commit in commits:
        for diff_obj in commit.diff(base_commit):
            # We always add new python files
            if diff_obj.change_type == "A" and diff_obj.b_path.endswith(".py"):
                code_diff.append(diff_obj.b_path)

    return code_diff


def get_new_python_files() -> List[str]:
    """
    Return a list of python files that have been added between the current head and the main branch.

    Returns:
        `List[str]`: The list of python files added.
    """
    repo = Repo(PATH_TO_REPO)

    try:
        # For the cases where the main branch exists locally
        main = repo.refs.main
    except AttributeError:
        # On GitHub Actions runners, it doesn't have local main branch
        main = repo.remotes.origin.refs.main

    print(f"main is at {main.commit}")
    print(f"Current head is at {repo.head.commit}")

    branching_commits = repo.merge_base(main, repo.head)
    for commit in branching_commits:
        print(f"Branching commit: {commit}")
    return get_new_python_files_between_commits(repo.head.commit, branching_commits)


def get_new_model():
    new_files = get_new_python_files()
    reg = re.compile(r"src/transformers/models/(.*)/modeling_.*\.py")

    new_model = ""
    for x in new_files:
        find_new_model = reg.findall(x)
        if len(find_new_model) > 0:
            new_model = find_new_model[0]
            # It's unlikely we have 2 new modeling files in a pull request.
            break
    return new_model


def parse_message(message: str) -> str:
    """
    Parses a GitHub pull request's comment to find the models specified in it to run slow CI.

    Args:
        message (`str`): The body of a GitHub pull request's comment.

    Returns:
        `str`: The substring in `message` after `run-slow`, run_slow` or run slow`. If no such prefix is found, the
        empty string is returned.
    """
    if message is None:
        return ""

    message = message.strip().lower()

    # run-slow: model_1, model_2
    if not message.startswith(("run-slow", "run_slow", "run slow")):
        return ""
    message = message[len("run slow") :]
    # remove leading `:`
    while message.strip().startswith(":"):
        message = message.strip()[1:]

    return message


def get_models(message: str):
    models = parse_message(message)
    return models.replace(",", " ").split()


def check_model_names(model_name: str):
    allowed = string.ascii_letters + string.digits + "_"
    return not (model_name.startswith("_") or model_name.endswith("_")) and all(c in allowed for c in model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="", help="The content of a comment.")
    parser.add_argument("--quantization", action="store_true", help="If we collect quantization tests")
    args = parser.parse_args()

    new_model = get_new_model()
    specified_models = get_models(args.message)
    models = ([] if new_model == "" else [new_model]) + specified_models
    # a guard for strange model names
    models = [model for model in models if check_model_names(model)]

    # Add prefix
    final_list = []
    for model in models:
        if not args.quantization:
            if os.path.isdir(f"tests/models/{model}"):
                final_list.append(f"models/{model}")
            elif os.path.isdir(f"tests/{model}") and model != "quantization":
                final_list.append(model)
        elif os.path.isdir(f"tests/quantization/{model}"):
            final_list.append(f"quantization/{model}")

    print(sorted(set(final_list)))

#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import argparse
import json
import os
import re
import subprocess

import requests


def create_script(target_test):
    """Create a python script to be run by `git bisect run` to determine if `target_test` passes or fails.
    If a test is not found in a commit, the script with exit code `0` (i.e. `Success`).

    Args:
        target_test (`str`): The test to check.

    Returns:
        `str`: The script to be run by `git bisect run`.
    """

    script = f"""
import os
import subprocess

result = subprocess.run(
    ["python3", "-m", "pytest", "-v", f"{target_test}"],
    capture_output = True,
    text=True,
)
print(result.stdout)

if len(result.stderr) > 0:
    if "ERROR: file or directory not found: " in result.stderr:
        print("test not found in this commit")
        exit(0)
    else:
        print(f"pytest failed to run: {{result.stderr}}")
        exit(-1)
elif f"{target_test} FAILED" in result.stdout:
    print("test failed")
    exit(2)

exit(0)
"""

    with open("target_script.py", "w") as fp:
        fp.write(script.strip())


def find_bad_commit(target_test, start_commit, end_commit):
    """Find (backward) the earliest commit between `start_commit` and `end_commit` at which `target_test` fails.

    Args:
        target_test (`str`): The test to check.
        start_commit (`str`): The latest commit.
        end_commit (`str`): The earliest commit.

    Returns:
        `str`: The earliest commit at which `target_test` fails.
    """

    if start_commit == end_commit:
        return start_commit

    create_script(target_test=target_test)

    bash = f"""
git bisect reset
git bisect start {start_commit} {end_commit}
git bisect run python3 target_script.py
"""

    with open("run_git_bisect.sh", "w") as fp:
        fp.write(bash.strip())

    result = subprocess.run(
        ["bash", "run_git_bisect.sh"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    if "error: bisect run failed" in result.stderr:
        index = result.stderr.find("error: bisect run failed")
        bash_error = result.stderr[index:]

        error_msg = f"Error when running git bisect:\nbash error: {bash_error}"

        pattern = "pytest failed to run: .+"
        pytest_errors = re.findall(pattern, result.stdout)
        if len(pytest_errors) > 0:
            pytest_error = pytest_errors[0]
            index = pytest_error.find("pytest failed to run: ")
            index += len("pytest failed to run: ")
            pytest_error = pytest_error[index:]
            error_msg += f"pytest error: {pytest_error}"

        raise ValueError(error_msg)

    pattern = r"(.+) is the first bad commit"
    commits = re.findall(pattern, result.stdout)

    bad_commit = None
    if len(commits) > 0:
        bad_commit = commits[0]

    print(f"Between `start_commit` {start_commit} and `end_commit` {end_commit}")
    print(f"bad_commit: {bad_commit}\n")

    return bad_commit


def get_commit_info(commit):
    """Get information for a commit via `api.github.com`."""
    pr_number = None
    author = None
    merged_author = None

    url = f"https://api.github.com/repos/huggingface/transformers/commits/{commit}/pulls"
    pr_info_for_commit = requests.get(url).json()

    if len(pr_info_for_commit) > 0:
        pr_number = pr_info_for_commit[0]["number"]

        url = f"https://api.github.com/repos/huggingface/transformers/pulls/{pr_number}"
        pr_for_commit = requests.get(url).json()
        author = pr_for_commit["user"]["login"]
        merged_author = pr_for_commit["merged_by"]["login"]

    if author is None:
        url = f"https://api.github.com/repos/huggingface/transformers/commits/{commit}"
        commit_info = requests.get(url).json()
        author = commit_info["author"]["login"]

    return {"commit": commit, "pr_number": pr_number, "author": author, "merged_by": merged_author}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_commit", type=str, required=True, help="The latest commit hash to check.")
    parser.add_argument("--end_commit", type=str, required=True, help="The earliest commit hash to check.")
    parser.add_argument("--test", type=str, help="The test to check.")
    parser.add_argument("--file", type=str, help="The report file.")
    parser.add_argument("--output_file", type=str, required=True, help="The path of the output file.")
    args = parser.parse_args()

    print(f"start_commit: {args.start_commit}")
    print(f"end_commit: {args.end_commit}")

    if len({args.test is None, args.file is None}) != 2:
        raise ValueError("Exactly one argument `test` or `file` must be specified.")

    if args.test is not None:
        commit = find_bad_commit(target_test=args.test, start_commit=args.start_commit, end_commit=args.end_commit)
        with open(args.output_file, "w", encoding="UTF-8") as fp:
            fp.write(f"{args.test}\n{commit}")
    elif os.path.isfile(args.file):
        with open(args.file, "r", encoding="UTF-8") as fp:
            reports = json.load(fp)

        for model in reports:
            # TODO: make this script able to deal with both `single-gpu` and `multi-gpu` via a new argument.
            reports[model].pop("multi-gpu", None)
            failed_tests = reports[model]["single-gpu"]

            failed_tests_with_bad_commits = []
            for test in failed_tests:
                commit = find_bad_commit(target_test=test, start_commit=args.start_commit, end_commit=args.end_commit)
                info = {"test": test, "commit": commit}
                info.update(get_commit_info(commit))
                failed_tests_with_bad_commits.append(info)

            # If no single-gpu test failures, remove the key
            if len(failed_tests_with_bad_commits) > 0:
                reports[model]["single-gpu"] = failed_tests_with_bad_commits
            else:
                reports[model].pop("single-gpu", None)

        # remove the models without any test failure
        reports = {k: v for k, v in reports.items() if len(v) > 0}

        with open(args.output_file, "w", encoding="UTF-8") as fp:
            json.dump(reports, fp, ensure_ascii=False, indent=4)

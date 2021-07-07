# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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

from contextlib import contextmanager
from git import Repo

@contextmanager
def checkout_commit(repo, commit_id):
    """
    Context manager that checks out a commit in the repo.
    """
    current_head = repo.head.commit if repo.head.is_detached else repo.head.ref 

    try:
        repo.git.checkout(commit_id)
        yield

    finally:
        repo.git.checkout(current_head)


def diff_is_docstring_only(repo, branching_point, filename):
    """
    Check if the diff is only a docstring in a filename renamed from filename to filename.
    """
    with checkout_commit(repo, branching_point):
        with open(filename, "r", encoding="utf-8") as f:
            old_content = f.read()
    
    with open(filename, "r", encoding="utf-8") as f:
        new_content = f.read()
    
    old_content_splits = old_content.split('"""')
    old_content_no_doc = "".join(old_content_splits[::2])

    new_content_splits = new_content.split('"""')
    new_content_no_doc = "".join(new_content_splits[::2])

    return old_content_no_doc == new_content_no_doc


def get_modified_files():
    """
    Return a list of files that have been modified between the current head and the master branch.
    """
    repo = Repo(".")

    print(f"Master is at {repo.refs.master.commit}")
    print(f"Current head is at {repo.head.commit}")

    branching_commits = repo.merge_base(repo.refs.master, repo.head)
    for commit in branching_commits:
        print(f"Branching commit: {commit}")

    print("### DIFF ###")
    code_diff = []
    for commit in branching_commits:
        for diff_obj in commit.diff(repo.head.commit):
            # We always add new python files
            if diff_obj.change_type == "A" and diff_obj.b_path.endswith(".py"):
                code_diff.append(diff_obj.b_path)
            # We check that deleted python files won't break correspondping tests.
            elif diff_obj.change_type == "D" and diff_obj.a_path.endswith(".py"):
                code_diff.append(diff_obj.a_path)
            # Now for modified files
            elif diff_obj.change_type == "M" and diff_obj.b_path.endswith(".py"):
                # In case of renames, we'll look at the tests using both the old and new name.
                if diff_obj.a_path != diff_obj.b_path:
                    code_diff.extend([diff_obj.a_path, diff_obj.b_path])
                else:
                    # Otherwise, we check modifications are in code and not docstrings.
                    if diff_is_docstring_only(repo, commit, diff_obj.b_path):
                        print(f"Ignoring diff in {diff_obj.b_path} as it only concerns docstrings.")
                    else:
                        code_diff.append(diff_obj.a_path)
        
    return code_diff

if __name__ == "__main__":
    modified_files = get_modified_files()
    print(f"Modified files: {modified_files}")

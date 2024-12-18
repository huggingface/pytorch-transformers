# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Utility that prepares the repository for releases (or patches) by updating all versions in the relevant places. It
also performs some post-release cleanup, by updating the links in the main README to respective model doc pages (from
main to stable).

To prepare for a release, use from the root of the repo on the release branch with:

```bash
python release.py
```

or use `make pre-release`.

To prepare for a patch release, use from the root of the repo on the release branch with:

```bash
python release.py --patch
```

or use `make pre-patch`.

To do the post-release cleanup, use from the root of the repo on the main branch with:

```bash
python release.py --post_release
```

or use `make post-release`.
"""

import argparse
import os
import re
from pathlib import Path

import packaging.version


# All paths are defined with the intent that this script should be run from the root of the repo.
PATH_TO_EXAMPLES = "examples/"
PATH_TO_MODELS = "src/transformers/models"
# This maps a type of file to the pattern to look for when searching where the version is defined, as well as the
# template to follow when replacing it with the new version.
REPLACE_PATTERNS = {
    "examples": (re.compile(r'^check_min_version\("[^"]+"\)\s*$', re.MULTILINE), 'check_min_version("VERSION")\n'),
    "init": (re.compile(r'^__version__\s+=\s+"([^"]+)"\s*$', re.MULTILINE), '__version__ = "VERSION"\n'),
    "setup": (re.compile(r'^(\s*)version\s*=\s*"[^"]+",', re.MULTILINE), r'\1version="VERSION",'),
}
# This maps a type of file to its path in Transformers
REPLACE_FILES = {
    "init": "src/transformers/__init__.py",
    "setup": "setup.py",
}
README_FILE = "README.md"


def update_version_in_file(fname: str, version: str, file_type: str):
    """
    Update the version of Transformers in one file.

    Args:
        fname (`str`): The path to the file where we want to update the version.
        version (`str`): The new version to set in the file.
        file_type (`str`): The type of the file (should be a key in `REPLACE_PATTERNS`).
    """
    with open(fname, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()
    re_pattern, replace = REPLACE_PATTERNS[file_type]
    replace = replace.replace("VERSION", version)
    code = re_pattern.sub(replace, code)
    with open(fname, "w", encoding="utf-8", newline="\n") as f:
        f.write(code)


def update_version_in_examples(version: str):
    """
    Update the version in all examples files.

    Args:
        version (`str`): The new version to set in the examples.
    """
    for folder, directories, fnames in os.walk(PATH_TO_EXAMPLES):
        # Removing some of the folders with non-actively maintained examples from the walk
        if "research_projects" in directories:
            directories.remove("research_projects")
        if "legacy" in directories:
            directories.remove("legacy")
        for fname in fnames:
            if fname.endswith(".py"):
                update_version_in_file(os.path.join(folder, fname), version, file_type="examples")


def global_version_update(version: str, patch: bool = False):
    """
    Update the version in all needed files.

    Args:
        version (`str`): The new version to set everywhere.
        patch (`bool`, *optional*, defaults to `False`): Whether or not this is a patch release.
    """
    for pattern, fname in REPLACE_FILES.items():
        update_version_in_file(fname, version, pattern)
    if not patch:
        # We don't update the version in the examples for patch releases.
        update_version_in_examples(version)


def remove_conversion_scripts():
    """
    Delete the scripts that convert models from older, unsupported formats. We don't want to include these
    in release wheels because they often have to open insecure file types (pickle, Torch .bin models). This results in
    vulnerability scanners flagging us and can cause compliance issues for users with strict security policies.
    """
    model_dir = Path(PATH_TO_MODELS)
    for conversion_script in list(model_dir.glob("**/convert*.py")):
        conversion_script.unlink()


def get_version() -> packaging.version.Version:
    """
    Reads the current version in the main __init__.
    """
    with open(REPLACE_FILES["init"], "r") as f:
        code = f.read()
    default_version = REPLACE_PATTERNS["init"][0].search(code).groups()[0]
    return packaging.version.parse(default_version)


def pre_release_work(patch: bool = False):
    """
    Do all the necessary pre-release steps:
    - figure out the next minor release version and ask confirmation
    - update the version everywhere
    - clean-up the model list in the main README

    Args:
        patch (`bool`, *optional*, defaults to `False`): Whether or not this is a patch release.
    """
    # First let's get the default version: base version if we are in dev, bump minor otherwise.
    default_version = get_version()
    if patch and default_version.is_devrelease:
        raise ValueError("Can't create a patch version from the dev branch, checkout a released version!")
    if default_version.is_devrelease:
        default_version = default_version.base_version
    elif patch:
        default_version = f"{default_version.major}.{default_version.minor}.{default_version.micro + 1}"
    else:
        default_version = f"{default_version.major}.{default_version.minor + 1}.0"

    # Now let's ask nicely if we have found the right version.
    version = input(f"Which version are you releasing? [{default_version}]")
    if len(version) == 0:
        version = default_version

    print(f"Updating version to {version}.")
    global_version_update(version, patch=patch)
    print("Deleting conversion scripts.")
    remove_conversion_scripts()


def post_release_work():
    """
    Do all the necessary post-release steps:
    - figure out the next dev version and ask confirmation
    - update the version everywhere
    - clean-up the model list in the main README
    """
    # First let's get the current version
    current_version = get_version()
    dev_version = f"{current_version.major}.{current_version.minor + 1}.0.dev0"
    current_version = current_version.base_version

    # Check with the user we got that right.
    version = input(f"Which version are we developing now? [{dev_version}]")
    if len(version) == 0:
        version = dev_version

    print(f"Updating version to {version}.")
    global_version_update(version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--post_release", action="store_true", help="Whether this is pre or post release.")
    parser.add_argument("--patch", action="store_true", help="Whether or not this is a patch release.")
    args = parser.parse_args()
    if not args.post_release:
        pre_release_work(patch=args.patch)
    elif args.patch:
        print("Nothing to do after a patch :-)")
    else:
        post_release_work()

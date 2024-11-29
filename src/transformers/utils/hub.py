# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Hub utilities: utilities related to download and cache models
"""

import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4

import huggingface_hub
import requests
from huggingface_hub import (
    _CACHED_NO_EXIST,
    CommitOperationAdd,
    ModelCard,
    ModelCardData,
    constants,
    create_branch,
    create_commit,
    create_repo,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    try_to_load_from_cache,
)
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    build_hf_headers,
    get_session,
    hf_raise_for_status,
    send_telemetry,
)
from huggingface_hub.utils._deprecation import _deprecate_method
from requests.exceptions import HTTPError

from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
    ENV_VARS_TRUE_VALUES,
    _tf_version,
    _torch_version,
    is_tf_available,
    is_torch_available,
    is_training_run_on_sagemaker,
)
from .logging import tqdm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_is_offline_mode = huggingface_hub.constants.HF_HUB_OFFLINE


def is_offline_mode():
    return _is_offline_mode


torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
default_cache_path = constants.default_cache_path
old_default_cache_path = os.path.join(torch_cache_home, "transformers")

# Determine default cache directory. Lots of legacy environment variables to ensure backward compatibility.
# The best way to set the cache path is with the environment variable HF_HOME. For more details, checkout this
# documentation page: https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables.
#
# In code, use `HF_HUB_CACHE` as the default cache path. This variable is set by the library and is guaranteed
# to be set to the right value.
#
# TODO: clean this for v5?
PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", constants.HF_HUB_CACHE)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

# Onetime move from the old location to the new one if no ENV variable has been set.
if (
    os.path.isdir(old_default_cache_path)
    and not os.path.isdir(constants.HF_HUB_CACHE)
    and "PYTORCH_PRETRAINED_BERT_CACHE" not in os.environ
    and "PYTORCH_TRANSFORMERS_CACHE" not in os.environ
    and "TRANSFORMERS_CACHE" not in os.environ
):
    logger.warning(
        "In Transformers v4.22.0, the default path to cache downloaded models changed from"
        " '~/.cache/torch/transformers' to '~/.cache/huggingface/hub'. Since you don't seem to have"
        " overridden and '~/.cache/torch/transformers' is a directory that exists, we're moving it to"
        " '~/.cache/huggingface/hub' to avoid redownloading models you have already in the cache. You should"
        " only see this message once."
    )
    shutil.move(old_default_cache_path, constants.HF_HUB_CACHE)

HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(constants.HF_HOME, "modules"))
TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules"
SESSION_ID = uuid4().hex

# Add deprecation warning for old environment variables.
for key in ("PYTORCH_PRETRAINED_BERT_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "TRANSFORMERS_CACHE"):
    if os.getenv(key) is not None:
        warnings.warn(
            f"Using `{key}` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.",
            FutureWarning,
        )


S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"

_staging_mode = os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
_default_endpoint = "https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co"

HUGGINGFACE_CO_RESOLVE_ENDPOINT = _default_endpoint
if os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None) is not None:
    warnings.warn(
        "Using the environment variable `HUGGINGFACE_CO_RESOLVE_ENDPOINT` is deprecated and will be removed in "
        "Transformers v5. Use `HF_ENDPOINT` instead.",
        FutureWarning,
    )
    HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None)
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", HUGGINGFACE_CO_RESOLVE_ENDPOINT)
HUGGINGFACE_CO_PREFIX = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/{model_id}/resolve/{revision}/{filename}"
HUGGINGFACE_CO_EXAMPLES_TELEMETRY = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/api/telemetry/examples"


def _get_cache_file_to_return(
    path_or_repo_id: str, full_filename: str, cache_dir: Union[str, Path, None] = None, revision: Optional[str] = None
):
    # We try to see if we have a cached version (not up to date):
    resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
        return resolved_file
    return None


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


# TODO: remove this once fully deprecated
# TODO? remove from './examples/research_projects/lxmert/utils.py' as well
# TODO? remove from './examples/research_projects/visual_bert/utils.py' as well
@_deprecate_method(version="4.39.0", message="This method is outdated and does not support the new cache system.")
def get_cached_models(cache_dir: Union[str, Path] = None) -> List[Tuple]:
    """
    Returns a list of tuples representing model binaries that are cached locally. Each tuple has shape `(model_url,
    etag, size_MB)`. Filenames in `cache_dir` are use to get the metadata for each model, only urls ending with *.bin*
    are added.

    Args:
        cache_dir (`Union[str, Path]`, *optional*):
            The cache directory to search for models within. Will default to the transformers cache if unset.

    Returns:
        List[Tuple]: List of tuples each with shape `(model_url, etag, size_MB)`
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    elif isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if not os.path.isdir(cache_dir):
        return []

    cached_models = []
    for file in os.listdir(cache_dir):
        if file.endswith(".json"):
            meta_path = os.path.join(cache_dir, file)
            with open(meta_path, encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
                url = metadata["url"]
                etag = metadata["etag"]
                if url.endswith(".bin"):
                    size_MB = os.path.getsize(meta_path.strip(".json")) / 1e6
                    cached_models.append((url, etag, size_MB))

    return cached_models


def define_sagemaker_information():
    try:
        instance_data = requests.get(os.environ["ECS_CONTAINER_METADATA_URI"]).json()
        dlc_container_used = instance_data["Image"]
        dlc_tag = instance_data["Image"].split(":")[1]
    except Exception:
        dlc_container_used = None
        dlc_tag = None

    sagemaker_params = json.loads(os.getenv("SM_FRAMEWORK_PARAMS", "{}"))
    runs_distributed_training = True if "sagemaker_distributed_dataparallel_enabled" in sagemaker_params else False
    account_id = os.getenv("TRAINING_JOB_ARN").split(":")[4] if "TRAINING_JOB_ARN" in os.environ else None

    sagemaker_object = {
        "sm_framework": os.getenv("SM_FRAMEWORK_MODULE", None),
        "sm_region": os.getenv("AWS_REGION", None),
        "sm_number_gpu": os.getenv("SM_NUM_GPUS", 0),
        "sm_number_cpu": os.getenv("SM_NUM_CPUS", 0),
        "sm_distributed_training": runs_distributed_training,
        "sm_deep_learning_container": dlc_container_used,
        "sm_deep_learning_container_tag": dlc_tag,
        "sm_account_id": account_id,
    }
    return sagemaker_object


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f"transformers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if is_tf_available():
        ua += f"; tensorflow/{_tf_version}"
    if constants.HF_HUB_DISABLE_TELEMETRY:
        return ua + "; telemetry/off"
    if is_training_run_on_sagemaker():
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in define_sagemaker_information().items())
    # CI will set this value to True
    if os.environ.get("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        ua += "; is_ci/true"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str]) -> Optional[str]:
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


def cached_file(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    repo_type: Optional[str] = None,
    user_agent: Optional[Union[str, Dict[str, str]]] = None,
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs,
) -> Optional[str]:
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("google-bert/bert-base-uncased", "pytorch_model.bin")
    ```
    """
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    # Private arguments
    #     _raise_exceptions_for_gated_repo: if False, do not raise an exception for gated repo error but return
    #         None.
    #     _raise_exceptions_for_missing_entries: if False, do not raise an exception for missing entries but return
    #         None.
    #     _raise_exceptions_for_connection_errors: if False, do not raise an exception for connection errors but return
    #         None.
    #     _commit_hash: passed when we are chaining several calls to various files (e.g. when loading a tokenizer or
    #         a pipeline). If files are cached for this commit hash, avoid calls to head and get from the cache.
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries and filename not in ["config.json", f"{subfolder}/config.json"]:
                raise EnvironmentError(
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                    f"'https://huggingface.co/{path_or_repo_id}/tree/{revision}' for available files."
                )
            else:
                return None
        return resolved_file

    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if _commit_hash is not None and not force_download:
        # If the file is cached under that commit hash, we return it directly.
        resolved_file = try_to_load_from_cache(
            path_or_repo_id, full_filename, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
        )
        if resolved_file is not None:
            if resolved_file is not _CACHED_NO_EXIST:
                return resolved_file
            elif not _raise_exceptions_for_missing_entries:
                return None
            else:
                raise EnvironmentError(f"Could not locate {full_filename} inside {path_or_repo_id}.")

    user_agent = http_user_agent(user_agent)
    try:
        # Load from URL or cache if already cached
        resolved_file = hf_hub_download(
            path_or_repo_id,
            filename,
            subfolder=None if len(subfolder) == 0 else subfolder,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
    except GatedRepoError as e:
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        if resolved_file is not None or not _raise_exceptions_for_gated_repo:
            return resolved_file
        raise EnvironmentError(
            "You are trying to access a gated repo.\nMake sure to have access to it at "
            f"https://huggingface.co/{path_or_repo_id}.\n{str(e)}"
        ) from e
    except RepositoryNotFoundError as e:
        raise EnvironmentError(
            f"{path_or_repo_id} is not a local folder and is not a valid model identifier "
            "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token "
            "having permission to this repo either by logging in with `huggingface-cli login` or by passing "
            "`token=<your_token>`"
        ) from e
    except RevisionNotFoundError as e:
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists "
            "for this model name. Check the model page at "
            f"'https://huggingface.co/{path_or_repo_id}' for available revisions."
        ) from e
    except LocalEntryNotFoundError as e:
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        if (
            resolved_file is not None
            or not _raise_exceptions_for_missing_entries
            or not _raise_exceptions_for_connection_errors
        ):
            return resolved_file
        raise EnvironmentError(
            f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this file, couldn't find it in the"
            f" cached files and it looks like {path_or_repo_id} is not the path to a directory containing a file named"
            f" {full_filename}.\nCheckout your internet connection or see how to run the library in offline mode at"
            " 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        ) from e
    except EntryNotFoundError as e:
        if not _raise_exceptions_for_missing_entries:
            return None
        if revision is None:
            revision = "main"
        if filename in ["config.json", f"{subfolder}/config.json"]:
            return None
        raise EnvironmentError(
            f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
            f"'https://huggingface.co/{path_or_repo_id}/tree/{revision}' for available files."
        ) from e
    except HTTPError as err:
        resolved_file = _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
        if resolved_file is not None or not _raise_exceptions_for_connection_errors:
            return resolved_file
        raise EnvironmentError(f"There was a specific connection error when trying to load {path_or_repo_id}:\n{err}")
    except HFValidationError as e:
        raise EnvironmentError(
            f"Incorrect path_or_model_id: '{path_or_repo_id}'. Please provide either the path to a local folder or the repo_id of a model on the Hub."
        ) from e
    return resolved_file


# TODO: deprecate `get_file_from_repo` or document it differently?
#       Docstring is exactly the same as `cached_repo` but behavior is slightly different. If file is missing or if
#       there is a connection error, `cached_repo` will return None while `get_file_from_repo` will raise an error.
#       IMO we should keep only 1 method and have a single `raise_error` argument (to be discussed).
def get_file_from_repo(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    **deprecated_kwargs,
):
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo) or `None` if the
        file does not exist.

    Examples:

    ```python
    # Download a tokenizer configuration from huggingface.co and cache.
    tokenizer_config = get_file_from_repo("google-bert/bert-base-uncased", "tokenizer_config.json")
    # This model does not have a tokenizer config so the result will be None.
    tokenizer_config = get_file_from_repo("FacebookAI/xlm-roberta-base", "tokenizer_config.json")
    ```
    """
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    return cached_file(
        path_or_repo_id=path_or_repo,
        filename=filename,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )


def download_url(url, proxies=None):
    """
    Downloads a given url in a temporary file. This function is not safe to use in multiple processes. Its only use is
    for deprecated behavior allowing to download config/models with a single url instead of using the Hub.

    Args:
        url (`str`): The url of the file to download.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.

    Returns:
        `str`: The location of the temporary file where the url was downloaded.
    """
    warnings.warn(
        f"Using `from_pretrained` with the url of a file (here {url}) is deprecated and won't be possible anymore in"
        " v5 of Transformers. You should host your file on the Hub (hf.co) instead and use the repository ID. Note"
        " that this is not compatible with the caching system (your file will be downloaded at each execution) or"
        " multiple processes (each process will download the file in a different temporary file).",
        FutureWarning,
    )
    tmp_fd, tmp_file = tempfile.mkstemp()
    with os.fdopen(tmp_fd, "wb") as f:
        http_get(url, f, proxies=proxies)
    return tmp_file


def has_file(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    revision: Optional[str] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    *,
    local_files_only: bool = False,
    cache_dir: Union[str, Path, None] = None,
    repo_type: Optional[str] = None,
    **deprecated_kwargs,
):
    """
    Checks if a repo contains a given file without downloading it. Works for remote repos and local folders.

    If offline mode is enabled, checks if the file exists in the cache.

    <Tip warning={false}>

    This function will raise an error if the repository `path_or_repo` is not valid or if `revision` does not exist for
    this repo, but will return False for regular connection errors.

    </Tip>
    """
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    # If path to local directory, check if the file exists
    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, filename))

    # Else it's a repo => let's check if the file exists in local cache or on the Hub

    # Check if file exists in cache
    # This information might be outdated so it's best to also make a HEAD call (if allowed).
    cached_path = try_to_load_from_cache(
        repo_id=path_or_repo,
        filename=filename,
        revision=revision,
        repo_type=repo_type,
        cache_dir=cache_dir,
    )
    has_file_in_cache = isinstance(cached_path, str)

    # If local_files_only, don't try the HEAD call
    if local_files_only:
        return has_file_in_cache

    # Check if the file exists
    try:
        response = get_session().head(
            hf_hub_url(path_or_repo, filename=filename, revision=revision, repo_type=repo_type),
            headers=build_hf_headers(token=token, user_agent=http_user_agent()),
            allow_redirects=False,
            proxies=proxies,
            timeout=10,
        )
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        # Actually raise for those subclasses of ConnectionError
        raise
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        OfflineModeIsEnabled,
    ):
        return has_file_in_cache

    try:
        hf_raise_for_status(response)
        return True
    except GatedRepoError as e:
        logger.error(e)
        raise EnvironmentError(
            f"{path_or_repo} is a gated repository. Make sure to request access at "
            f"https://huggingface.co/{path_or_repo} and pass a token having permission to this repo either by "
            "logging in with `huggingface-cli login` or by passing `token=<your_token>`."
        ) from e
    except RepositoryNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(
            f"{path_or_repo} is not a local folder or a valid repository name on 'https://hf.co'."
        ) from e
    except RevisionNotFoundError as e:
        logger.error(e)
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
            f"model name. Check the model page at 'https://huggingface.co/{path_or_repo}' for available revisions."
        ) from e
    except EntryNotFoundError:
        return False  # File does not exist
    except requests.HTTPError:
        # Any authentication/authorization error will be caught here => default to cache
        return has_file_in_cache


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def _create_repo(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> str:
        """
        Create the repo if needed, cleans up repo_id with deprecated kwargs `repo_url` and `organization`, retrieves
        the token.
        """
        if repo_url is not None:
            warnings.warn(
                "The `repo_url` argument is deprecated and will be removed in v5 of Transformers. Use `repo_id` "
                "instead."
            )
            if repo_id is not None:
                raise ValueError(
                    "`repo_id` and `repo_url` are both specified. Please set only the argument `repo_id`."
                )
            repo_id = repo_url.replace(f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/", "")
        if organization is not None:
            warnings.warn(
                "The `organization` argument is deprecated and will be removed in v5 of Transformers. Set your "
                "organization directly in the `repo_id` passed instead (`repo_id={organization}/{model_id}`)."
            )
            if not repo_id.startswith(organization):
                if "/" in repo_id:
                    repo_id = repo_id.split("/")[-1]
                repo_id = f"{organization}/{repo_id}"

        url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        return url.repo_id

    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

    def _upload_modified_files(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        files_timestamps: Dict[str, float],
        commit_message: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        revision: str = None,
        commit_description: str = None,
    ):
        """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            elif "Tokenizer" in self.__class__.__name__:
                commit_message = "Upload tokenizer"
            elif "FeatureExtractor" in self.__class__.__name__:
                commit_message = "Upload feature extractor"
            elif "Processor" in self.__class__.__name__:
                commit_message = "Upload processor"
            else:
                commit_message = f"Upload {self.__class__.__name__}"
        modified_files = [
            f
            for f in os.listdir(working_dir)
            if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
        ]

        # filter for actual files + folders at the root level
        modified_files = [
            f
            for f in modified_files
            if os.path.isfile(os.path.join(working_dir, f)) or os.path.isdir(os.path.join(working_dir, f))
        ]

        operations = []
        # upload standalone files
        for file in modified_files:
            if os.path.isdir(os.path.join(working_dir, file)):
                # go over individual files of folder
                for f in os.listdir(os.path.join(working_dir, file)):
                    operations.append(
                        CommitOperationAdd(
                            path_or_fileobj=os.path.join(working_dir, file, f), path_in_repo=os.path.join(file, f)
                        )
                    )
            else:
                operations.append(
                    CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file)
                )

        if revision is not None and not revision.startswith("refs/pr"):
            try:
                create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)
            except HfHubHTTPError as e:
                if e.response.status_code == 403 and create_pr:
                    # If we are creating a PR on a repo we don't have access to, we can't create the branch.
                    # so let's assume the branch already exists. If it's not the case, an error will be raised when
                    # calling `create_commit` below.
                    pass
                else:
                    raise

        logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
        return create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            create_pr=create_pr,
            revision=revision,
        )

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "5GB",
        create_pr: bool = False,
        safe_serialization: bool = True,
        revision: str = None,
        commit_description: str = None,
        tags: Optional[List[str]] = None,
        **deprecated_kwargs,
    ) -> str:
        """
        Upload the {object_files} to the 🤗 Model Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
                Google Colab instances without any CPU OOM issues.
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether or not to convert the model weights in safetensors format for safer serialization.
            revision (`str`, *optional*):
                Branch to push the uploaded files to.
            commit_description (`str`, *optional*):
                The description of the commit that will be created
            tags (`List[str]`, *optional*):
                List of tags to push on the Hub.

        Examples:

        ```python
        from transformers import {object_class}

        {object} = {object_class}.from_pretrained("google-bert/bert-base-cased")

        # Push the {object} to your namespace with the name "my-finetuned-bert".
        {object}.push_to_hub("my-finetuned-bert")

        # Push the {object} to an organization with the name "my-finetuned-bert".
        {object}.push_to_hub("huggingface/my-finetuned-bert")
        ```
        """
        use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
        ignore_metadata_errors = deprecated_kwargs.pop("ignore_metadata_errors", False)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        repo_path_or_name = deprecated_kwargs.pop("repo_path_or_name", None)
        if repo_path_or_name is not None:
            # Should use `repo_id` instead of `repo_path_or_name`. When using `repo_path_or_name`, we try to infer
            # repo_id from the folder path, if it exists.
            warnings.warn(
                "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
                "`repo_id` instead.",
                FutureWarning,
            )
            if repo_id is not None:
                raise ValueError(
                    "`repo_id` and `repo_path_or_name` are both specified. Please set only the argument `repo_id`."
                )
            if os.path.isdir(repo_path_or_name):
                # repo_path: infer repo_id from the path
                repo_id = repo_id.split(os.path.sep)[-1]
                working_dir = repo_id
            else:
                # repo_name: use it as repo_id
                repo_id = repo_path_or_name
                working_dir = repo_id.split("/")[-1]
        else:
            # Repo_id is passed correctly: infer working_dir from it
            working_dir = repo_id.split("/")[-1]

        # Deprecation warning will be sent after for repo_url and organization
        repo_url = deprecated_kwargs.pop("repo_url", None)
        organization = deprecated_kwargs.pop("organization", None)

        repo_id = self._create_repo(
            repo_id, private=private, token=token, repo_url=repo_url, organization=organization
        )

        # Create a new empty model card and eventually tag it
        model_card = create_and_tag_model_card(
            repo_id, tags, token=token, ignore_metadata_errors=ignore_metadata_errors
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save_pretrained(work_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)

            # Update model card if needed:
            model_card.save(os.path.join(work_dir, "README.md"))

            return self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                revision=revision,
                commit_description=commit_description,
            )


def send_example_telemetry(example_name, *example_args, framework="pytorch"):
    """
    Sends telemetry that helps tracking the examples use.

    Args:
        example_name (`str`): The name of the example.
        *example_args (dataclasses or `argparse.ArgumentParser`): The arguments to the script. This function will only
            try to extract the model and dataset name from those. Nothing else is tracked.
        framework (`str`, *optional*, defaults to `"pytorch"`): The framework for the example.
    """
    if is_offline_mode():
        return

    data = {"example": example_name, "framework": framework}
    for args in example_args:
        args_as_dict = {k: v for k, v in args.__dict__.items() if not k.startswith("_") and v is not None}
        if "model_name_or_path" in args_as_dict:
            model_name = args_as_dict["model_name_or_path"]
            # Filter out local paths
            if not os.path.isdir(model_name):
                data["model_name"] = args_as_dict["model_name_or_path"]
        if "dataset_name" in args_as_dict:
            data["dataset_name"] = args_as_dict["dataset_name"]
        elif "task_name" in args_as_dict:
            # Extract script name from the example_name
            script_name = example_name.replace("tf_", "").replace("flax_", "").replace("run_", "")
            script_name = script_name.replace("_no_trainer", "")
            data["dataset_name"] = f"{script_name}-{args_as_dict['task_name']}"

    # Send telemetry in the background
    send_telemetry(
        topic="examples", library_name="transformers", library_version=__version__, user_agent=http_user_agent(data)
    )


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:
    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError("`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=None,
    local_files_only=False,
    token=None,
    user_agent=None,
    revision=None,
    subfolder="",
    _commit_hash=None,
    **deprecated_kwargs,
):
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.

    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """
    import json

    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    cached_filenames = []
    # Check if the model is already cached or not. We only try the last checkpoint, this should cover most cases of
    # downloaded (if interrupted).
    last_shard = try_to_load_from_cache(
        pretrained_model_name_or_path, shard_filenames[-1], cache_dir=cache_dir, revision=_commit_hash
    )
    show_progress_bar = last_shard is None or force_download
    for shard_filename in tqdm(shard_filenames, desc="Downloading shards", disable=not show_progress_bar):
        try:
            # Load from URL
            cached_filename = cached_file(
                pretrained_model_name_or_path,
                shard_filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=_commit_hash,
            )
        # We have already dealt with RepositoryNotFoundError and RevisionNotFoundError when getting the index, so
        # we don't have to catch them here.
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {shard_filename} which is "
                "required according to the checkpoint index."
            )
        except HTTPError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {shard_filename}. You should try"
                " again after checking your internet connection."
            )

        cached_filenames.append(cached_filename)

    return cached_filenames, sharded_metadata


# All what is below is for conversion between old cache format and new cache format.


def get_all_cached_files(cache_dir=None):
    """
    Returns a list for all files cached with appropriate metadata.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = str(cache_dir)
    if not os.path.isdir(cache_dir):
        return []

    cached_files = []
    for file in os.listdir(cache_dir):
        meta_path = os.path.join(cache_dir, f"{file}.json")
        if not os.path.isfile(meta_path):
            continue

        with open(meta_path, encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
            url = metadata["url"]
            etag = metadata["etag"].replace('"', "")
            cached_files.append({"file": file, "url": url, "etag": etag})

    return cached_files


def extract_info_from_url(url):
    """
    Extract repo_name, revision and filename from an url.
    """
    search = re.search(r"^https://huggingface\.co/(.*)/resolve/([^/]*)/(.*)$", url)
    if search is None:
        return None
    repo, revision, filename = search.groups()
    cache_repo = "--".join(["models"] + repo.split("/"))
    return {"repo": cache_repo, "revision": revision, "filename": filename}


def create_and_tag_model_card(
    repo_id: str,
    tags: Optional[List[str]] = None,
    token: Optional[str] = None,
    ignore_metadata_errors: bool = False,
):
    """
    Creates or loads an existing model card and tags it.

    Args:
        repo_id (`str`):
            The repo_id where to look for the model card.
        tags (`List[str]`, *optional*):
            The list of tags to add in the model card
        token (`str`, *optional*):
            Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to the stored token.
        ignore_metadata_errors (`str`):
            If True, errors while parsing the metadata section will be ignored. Some information might be lost during
            the process. Use it at your own risk.
    """
    try:
        # Check if the model card is present on the remote repo
        model_card = ModelCard.load(repo_id, token=token, ignore_metadata_errors=ignore_metadata_errors)
    except EntryNotFoundError:
        # Otherwise create a simple model card from template
        model_description = "This is the model card of a 🤗 transformers model that has been pushed on the Hub. This model card has been automatically generated."
        card_data = ModelCardData(tags=[] if tags is None else tags, library_name="transformers")
        model_card = ModelCard.from_template(card_data, model_description=model_description)

    if tags is not None:
        # Ensure model_card.data.tags is a list and not None
        if model_card.data.tags is None:
            model_card.data.tags = []
        for model_tag in tags:
            if model_tag not in model_card.data.tags:
                model_card.data.tags.append(model_tag)

    return model_card


def clean_files_for(file):
    """
    Remove, if they exist, file, file.json and file.lock
    """
    for f in [file, f"{file}.json", f"{file}.lock"]:
        if os.path.isfile(f):
            os.remove(f)


def move_to_new_cache(file, repo, filename, revision, etag, commit_hash):
    """
    Move file to repo following the new huggingface hub cache organization.
    """
    os.makedirs(repo, exist_ok=True)

    # refs
    os.makedirs(os.path.join(repo, "refs"), exist_ok=True)
    if revision != commit_hash:
        ref_path = os.path.join(repo, "refs", revision)
        with open(ref_path, "w") as f:
            f.write(commit_hash)

    # blobs
    os.makedirs(os.path.join(repo, "blobs"), exist_ok=True)
    blob_path = os.path.join(repo, "blobs", etag)
    shutil.move(file, blob_path)

    # snapshots
    os.makedirs(os.path.join(repo, "snapshots"), exist_ok=True)
    os.makedirs(os.path.join(repo, "snapshots", commit_hash), exist_ok=True)
    pointer_path = os.path.join(repo, "snapshots", commit_hash, filename)
    huggingface_hub.file_download._create_relative_symlink(blob_path, pointer_path)
    clean_files_for(file)


def move_cache(cache_dir=None, new_cache_dir=None, token=None):
    if new_cache_dir is None:
        new_cache_dir = TRANSFORMERS_CACHE
    if cache_dir is None:
        # Migrate from old cache in .cache/huggingface/transformers
        old_cache = Path(TRANSFORMERS_CACHE).parent / "transformers"
        if os.path.isdir(str(old_cache)):
            cache_dir = str(old_cache)
        else:
            cache_dir = new_cache_dir
    cached_files = get_all_cached_files(cache_dir=cache_dir)
    logger.info(f"Moving {len(cached_files)} files to the new cache system")

    hub_metadata = {}
    for file_info in tqdm(cached_files):
        url = file_info.pop("url")
        if url not in hub_metadata:
            try:
                hub_metadata[url] = get_hf_file_metadata(url, token=token)
            except requests.HTTPError:
                continue

        etag, commit_hash = hub_metadata[url].etag, hub_metadata[url].commit_hash
        if etag is None or commit_hash is None:
            continue

        if file_info["etag"] != etag:
            # Cached file is not up to date, we just throw it as a new version will be downloaded anyway.
            clean_files_for(os.path.join(cache_dir, file_info["file"]))
            continue

        url_info = extract_info_from_url(url)
        if url_info is None:
            # Not a file from huggingface.co
            continue

        repo = os.path.join(new_cache_dir, url_info["repo"])
        move_to_new_cache(
            file=os.path.join(cache_dir, file_info["file"]),
            repo=repo,
            filename=url_info["filename"],
            revision=url_info["revision"],
            etag=etag,
            commit_hash=commit_hash,
        )


class PushInProgress:
    """
    Internal class to keep track of a push in progress (which might contain multiple `Future` jobs).
    """

    def __init__(self, jobs: Optional[futures.Future] = None) -> None:
        self.jobs = [] if jobs is None else jobs

    def is_done(self):
        return all(job.done() for job in self.jobs)

    def wait_until_done(self):
        futures.wait(self.jobs)

    def cancel(self) -> None:
        self.jobs = [
            job
            for job in self.jobs
            # Cancel the job if it wasn't started yet and remove cancelled/done jobs from the list
            if not (job.cancel() or job.done())
        ]


cache_version_file = os.path.join(TRANSFORMERS_CACHE, "version.txt")
if not os.path.isfile(cache_version_file):
    cache_version = 0
else:
    with open(cache_version_file) as f:
        try:
            cache_version = int(f.read())
        except ValueError:
            cache_version = 0

cache_is_not_empty = os.path.isdir(TRANSFORMERS_CACHE) and len(os.listdir(TRANSFORMERS_CACHE)) > 0

if cache_version < 1 and cache_is_not_empty:
    if is_offline_mode():
        logger.warning(
            "You are offline and the cache for model files in Transformers v4.22.0 has been updated while your local "
            "cache seems to be the one of a previous version. It is very likely that all your calls to any "
            "`from_pretrained()` method will fail. Remove the offline mode and enable internet connection to have "
            "your cache be updated automatically, then you can go back to offline mode."
        )
    else:
        logger.warning(
            "The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a "
            "one-time only operation. You can interrupt this and resume the migration later on by calling "
            "`transformers.utils.move_cache()`."
        )
    try:
        if TRANSFORMERS_CACHE != constants.HF_HUB_CACHE:
            # Users set some env variable to customize cache storage
            move_cache(TRANSFORMERS_CACHE, TRANSFORMERS_CACHE)
        else:
            move_cache()
    except Exception as e:
        trace = "\n".join(traceback.format_tb(e.__traceback__))
        logger.error(
            f"There was a problem when trying to move your cache:\n\n{trace}\n{e.__class__.__name__}: {e}\n\nPlease "
            "file an issue at https://github.com/huggingface/transformers/issues/new/choose and copy paste this whole "
            "message and we will do our best to help."
        )

if cache_version < 1:
    try:
        os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)
        with open(cache_version_file, "w") as f:
            f.write("1")
    except Exception:
        logger.warning(
            f"There was a problem when trying to write in your cache folder ({TRANSFORMERS_CACHE}). You should set "
            "the environment variable TRANSFORMERS_CACHE to a writable directory."
        )

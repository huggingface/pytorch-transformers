from typing import Optional

import requests
from huggingface_hub import Discussion, HfApi, get_repo_discussions

from .utils import cached_file, http_user_agent, logging


logger = logging.get_logger(__name__)


def previous_pr(api: HfApi, model_id: str, pr_title: str, token: str) -> Optional["Discussion"]:
    main_commit = api.list_repo_commits(model_id, token=token)[0].commit_id
    for discussion in get_repo_discussions(repo_id=model_id, token=token):
        if discussion.title == pr_title and discussion.status == "open" and discussion.is_pull_request:
            commits = api.list_repo_commits(model_id, revision=discussion.git_reference, token=token)

            if main_commit == commits[1].commit_id:
                return discussion
    return None


def spawn_conversion(token: str, private: bool, model_id: str):
    logger.info("Attempting to convert .bin model on the fly to safetensors.")

    safetensors_convert_space_url = "https://safetensors-convert.hf.space"
    sse_url = f"{safetensors_convert_space_url}/call/run"

    def start(_sse_connection):
        for line in _sse_connection.iter_lines():
            line = line.decode()
            if line.startswith("event:"):
                status = line[7:]
                logger.debug(f"Safetensors conversion status: {status}")

                if status == "complete":
                    return
                elif status == "heartbeat":
                    logger.debug("Heartbeat")
                else:
                    logger.debug(f"Unknown status {status}")
            else:
                logger.debug(line)

    data = {"data": [model_id, private, token]}

    result = requests.post(sse_url, stream=True, json=data).json()
    event_id = result["event_id"]

    with requests.get(f"{sse_url}/{event_id}", stream=True) as sse_connection:
        try:
            logger.debug("Spawning safetensors automatic conversion.")
            start(sse_connection)
        except Exception as e:
            logger.warning(f"Error during conversion: {repr(e)}")


def get_conversion_pr_reference(api: HfApi, model_id: str, **kwargs):
    private = api.model_info(model_id).private

    logger.info("Attempting to create safetensors variant")
    pr_title = "Adding `safetensors` variant of this model"
    token = kwargs.get("token")

    # This looks into the current repo's open PRs to see if a PR for safetensors was already open. If so, it
    # returns it. It checks that the PR was opened by the bot and not by another user so as to prevent
    # security breaches.
    pr = previous_pr(api, model_id, pr_title, token=token)

    if pr is None or (not private and pr.author != "SFconvertbot"):
        spawn_conversion(token, private, model_id)
        pr = previous_pr(api, model_id, pr_title, token=token)
    else:
        logger.info("Safetensors PR exists")

    sha = f"refs/pr/{pr.num}"

    return sha


def auto_conversion(pretrained_model_name_or_path: str, ignore_errors_during_conversion=False, **cached_file_kwargs):
    try:
        api = HfApi(token=cached_file_kwargs.get("token"), headers={"user-agent": http_user_agent()})
        sha = get_conversion_pr_reference(api, pretrained_model_name_or_path, **cached_file_kwargs)

        if sha is None:
            return None, None
        cached_file_kwargs["revision"] = sha
        del cached_file_kwargs["_commit_hash"]

        # This is an additional HEAD call that could be removed if we could infer sharded/non-sharded from the PR
        # description.
        sharded = api.file_exists(
            pretrained_model_name_or_path,
            "model.safetensors.index.json",
            revision=sha,
            token=cached_file_kwargs.get("token"),
        )
        filename = "model.safetensors.index.json" if sharded else "model.safetensors"

        resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
        return resolved_archive_file, sha, sharded
    except Exception as e:
        if not ignore_errors_during_conversion:
            raise e

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import time
from collections.abc import Callable
from tempfile import TemporaryDirectory
from typing import TypeVar

import git  # noqa: TID251  We allow direct import of Git in scripts, since most users won't interact with them.
from git.exc import (  # noqa: TID251  We allow direct import of Git in scripts, since most users won't interact with them.
    GitCommandError,
)
from huggingface_hub import create_repo, create_tag, repo_exists, upload_folder

CallableRetT = TypeVar("CallableRetT")
HUGGINGFACE_ORG_NAME = "qualcomm"


def is_hf_api_timeout_error(e: Exception) -> bool:
    return isinstance(e, TimeoutError)


def is_git_timeout_error(e: Exception) -> bool:
    return is_hf_api_timeout_error(e) or (
        isinstance(e, GitCommandError)
        and "The requested URL returned error: 429" in e.stderr
    )


def _timeout_retry(
    do: Callable[[], CallableRetT],
    max_retries: int,
    is_timeout_error: Callable[[Exception], bool] = is_hf_api_timeout_error,
) -> CallableRetT:
    """
    Execute the given callable and return the result.

    If the callable returns a TimeoutError, it will be retried with increasingly large sleep times inbetween calls,
    up to a max number of retries.

    This is used as a workaround for Hugging Face 429 (too many requests!) errors when uploading many models in 1 session.

    Parameters
    ----------
    do
        Callable to do and retry as necessary. Typically you just use (lambda: exp) for this parameter.
    max_retries
        Maximum number of times to retry if we hit timeouts. do() would be executed a maximum of max_retries + 1 times, if it times out on each attempt.
    is_timeout_error
        Returns true if an error (thrown by "do()") is a timeout that should be retried.

    Returns
    -------
    result : CallableRetT
        Result from successful execution of do().

    Raises
    ------
    Exception
        If the allowed number of retries has been exhaused and do() has not succeeded.
    """
    for attempt_idx in range(max_retries + 1):
        try:
            return do()
        except Exception as e:
            if attempt_idx >= max_retries or not is_timeout_error(e):
                raise  # No more retries available, so raise the timeout

        if attempt_idx <= 1:
            time.sleep(10**attempt_idx)  # 1, 10
        else:
            time.sleep(30 * (2**attempt_idx - 2))  # 30, 60, 120, ...
    raise AssertionError()  # line is not reachable


def commit_and_push_to_hf(
    release_root_path: str | os.PathLike,
    hf_model_name: str,
    version: str,
    commit_description: str,
    hf_token: str,
    max_retries: int = 5,
    org_name: str = HUGGINGFACE_ORG_NAME,
) -> None:
    """
    Upload a folder to a HuggingFace repository and create a version tag.

    If the version tag already exists, the previous tag and associated commit
    are deleted before uploading the new content.

    Parameters
    ----------
    release_root_path
        Path to the folder containing files to upload.
    hf_model_name
        Name of the model repository on HuggingFace (under the qualcomm org).
    version
        Version string (e.g. '1.2.3r1') used for the git tag.
    commit_description
        Commit description for the upload. If '$QAIHM_TAG' is in this string,
        it will be replaced by the version tag.
    hf_token
        HuggingFace token with write access.
    max_retries
        Maximum number of retries for timeout errors.
    org_name
        HuggingFace organization name.
    """
    # Upload to hugging face. Retry if we hit timeouts.
    repo_id = f"{org_name}/{hf_model_name}"
    version_tag = f"v{version}"

    # If this tag exists already, delete previous tag and associated commit.
    if _timeout_retry(lambda: repo_exists(repo_id, token=hf_token), max_retries):
        with TemporaryDirectory() as tmpdir:
            # Bare clone the repo (include only the history and no actual files)
            # so we can manipulate the repo git history cheaply.
            repo = _timeout_retry(
                lambda: git.Repo.clone_from(
                    f"https://oauth2:{hf_token}@huggingface.co/{repo_id}",
                    tmpdir,
                    depth=2,
                    bare=True,
                ),
                max_retries,
                is_git_timeout_error,
            )

            # Only modify the old repo if the given tag exists already.
            if version_tag in repo.tags:
                tag = repo.tags[version_tag]
                main_branch = repo.heads.main
                remote = repo.remote("origin")
                if (
                    tag.commit == main_branch.commit
                    and len(main_branch.commit.parents) > 0
                ):
                    # If the tag maps to the last commit in the main branch,
                    # remove that commit from history so we can replace it.
                    previous_commit = main_branch.commit.parents[0]
                    repo.git.update_ref(main_branch.path, previous_commit.hexsha)
                    _timeout_retry(
                        lambda: remote.push(main_branch, force=True),
                        max_retries,
                        is_git_timeout_error,
                    )
                # Delete the old tag.
                repo.delete_tag(tag)
                _timeout_retry(
                    lambda: remote.push(
                        refspec=f"refs/tags/{version_tag}", delete=True
                    ),
                    max_retries,
                    is_git_timeout_error,
                )

    # Upload new commit and tag.
    _timeout_retry(
        lambda: create_repo(repo_id=repo_id, exist_ok=True, token=hf_token),
        max_retries,
    )
    _timeout_retry(
        lambda: upload_folder(
            folder_path=str(release_root_path),
            repo_id=repo_id,
            delete_patterns="*",  # Delete all previous files
            commit_message=version_tag,
            commit_description=commit_description.replace("$QAIHM_TAG", version_tag),
            token=hf_token,
        ),
        max_retries,
    )
    _timeout_retry(
        lambda: create_tag(repo_id=repo_id, tag=version_tag, token=hf_token),
        max_retries,
    )

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import functools
import logging
import os
import re
import sys
from collections.abc import Callable
from typing import TypeVar

import boto3
import botocore.exceptions
import tqdm
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, NoCredentialsError
from mypy_boto3_s3.service_resource import Bucket, ObjectSummary

from qai_hub_models.utils.envvars import IsOnCIEnvvar

QAIHM_PUBLIC_S3_BUCKET = "qaihub-public-assets"
QAIHM_PRIVATE_S3_BUCKET = "qai-hub-models-private-assets"
QAIHM_AWS_PROFILE = "qaihm"

CallableRetT = TypeVar("CallableRetT")


def has_qaihm_aws_profile() -> bool:
    """
    Check whether the local AWS config contains a 'qaihm' profile.

    This indicates the user has previously run the credential setup script
    (scripts/aws/validate_credentials.py), meaning they are an internal user
    whose credentials may just need refreshing.

    Returns True if the profile section exists (regardless of whether
    the credentials are currently valid), False otherwise.
    """
    try:
        boto3.Session(profile_name=QAIHM_AWS_PROFILE)
        return True
    except (botocore.exceptions.ProfileNotFound, botocore.exceptions.BotoCoreError):
        return False


@functools.lru_cache(maxsize=1)
def can_access_private_s3() -> bool:
    """Check if user can access private S3 (CI or has AWS profile with valid credentials)."""
    if not IsOnCIEnvvar.get() or not has_qaihm_aws_profile():
        return False
    try:
        session = boto3.Session(profile_name=QAIHM_AWS_PROFILE)
        session.client("sts").get_caller_identity()
        return True
    except (botocore.exceptions.BotoCoreError, ClientError, NoCredentialsError):
        return False


def attempt_with_s3_credentials_warning(
    s3_call: Callable[[], CallableRetT],
) -> CallableRetT:
    """
    Attempt to call the given function. Wrap the failure with a helpful warning about missing credentials.

    Typically you would call this like so:
        list_s3_files_in_folder_recursive(lambda: get_s3_url(args))
    """
    try:
        return s3_call()
    except (ClientError, NoCredentialsError) as e:
        if isinstance(e, NoCredentialsError) or e.response.get("Error", {}).get(
            "Code", None
        ) in ["400", "ExpiredToken"]:
            raise ValueError(
                "S3 credentials not found or expired. Run `python scripts/build_and_test.py validate_aws_credentials` and retry."
            ) from e
        raise


def get_s3_url(bucket_name: str, key: str) -> str:
    return f"s3://{bucket_name}/{key}"


def list_s3_files_in_folder_recursive(
    bucket: Bucket, prefix: str
) -> list[ObjectSummary]:
    """
    List s3 objects in the given bucket under the folder with the given prefix.

    Removes "folder" objects from the returned list, so all returned objects point to files.
    """
    objs = list(
        attempt_with_s3_credentials_warning(
            lambda: bucket.objects.filter(Prefix=prefix)
        )
    )
    return [obj for obj in objs if not obj.key.endswith("/")]


def s3_download(
    bucket: Bucket, key: str, local_file_path: str | os.PathLike, verbose: bool = True
) -> None:
    """Download file at s3://<bucket>/<key> to local_file_path."""
    obj = bucket.Object(key)
    with tqdm.tqdm(
        total=obj.content_length, unit="B", unit_scale=True, disable=not verbose
    ) as t:
        attempt_with_s3_credentials_warning(
            lambda: obj.download_file(
                str(local_file_path),
                Callback=t.update if not IsOnCIEnvvar.get() else None,
            )
        )
        if IsOnCIEnvvar.get():
            t.update(obj.content_length)


def s3_file_exists(bucket: Bucket, key: str) -> bool:
    """
    Checks if a file (object) exists in an S3 bucket.

    Parameters
    ----------
    bucket
        The name of the S3 bucket.
    key
        The key (path) of the file within the bucket.

    Returns
    -------
    exists : bool
        True if the file exists, False otherwise.
    """
    try:
        attempt_with_s3_credentials_warning(lambda: bucket.Object(key).load())
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", None) == "404":
            # The object does not exist.
            return False
        raise


def s3_multipart_upload(
    bucket: Bucket,
    key: str,
    local_file_path: str | os.PathLike,
    make_public: bool = False,
) -> None:
    """
    Uploads file at local_file_path to s3://<bucket>/<key>.

    If make_public is true, the object is made publicly accessible.
    """
    # Create a TransferConfig to manage multipart settings
    config = TransferConfig(
        multipart_threshold=1024 * 25,  # Set threshold to 25MB
        max_concurrency=10,  # Max number of threads
        multipart_chunksize=1024 * 25,  # Set chunk size to 25MB
        use_threads=True,
    )

    # Perform the multipart upload
    file_size = os.path.getsize(local_file_path)
    with (
        open(local_file_path, "rb") as file_data,
        tqdm.tqdm(total=file_size, unit="B", unit_scale=True, desc=key) as progress_bar,
    ):

        def upload_progress(bytes_transferred: int) -> None:
            progress_bar.update(bytes_transferred)

        attempt_with_s3_credentials_warning(
            lambda: bucket.upload_fileobj(
                file_data,
                key,
                Config=config,
                Callback=upload_progress,
                ExtraArgs={"ACL": "public-read"} if make_public else {},
            )
        )

    print(
        f"Uploaded {local_file_path} to {get_s3_url(bucket.name, key)}{' and made it public.' if make_public else ''}"
    )


def s3_copy(
    src_bucket: Bucket,
    src_key: str,
    dst_bucket: Bucket,
    dst_key: str,
    make_dst_public: bool = False,
) -> None:
    attempt_with_s3_credentials_warning(
        lambda: dst_bucket.copy(
            {"Bucket": src_bucket.name, "Key": src_key},
            dst_key,
            SourceClient=src_bucket.meta.client,
        )
    )

    if make_dst_public:
        attempt_with_s3_credentials_warning(
            lambda: dst_bucket.Object(dst_key).Acl().put(ACL="public-read")
        )


def get_qaihm_s3(bucket_name: str, requires_admin: bool = False) -> tuple[Bucket, bool]:
    """
    Get boto3 objects for interacting with the given bucket using QAIHM credentials.
    Throws if credentials do not exist.

    Parameters
    ----------
    bucket_name
        Name of the s3 bucket to get objects for.
    requires_admin
        Whether admin permissions are required.

    Returns
    -------
    bucket : Bucket
        Bucket object for the specified S3 bucket.
    is_admin : bool
        Whether the current credentials have admin permissions.
    """
    try:
        session = boto3.Session(profile_name=QAIHM_AWS_PROFILE)
        bucket = session.resource("s3").Bucket(bucket_name)
    except Exception as e:
        raise ValueError(
            "Could not find valid AWS profile. Run <repo_root>/scripts/build_and_test.py --task validate_aws_credentials"
        ) from e

    # Only check admin role when explicitly required, to avoid an STS round-trip on every call.
    is_admin = False
    if requires_admin:
        admin_role = os.environ.get("QAIHM_AWS_ADMIN_ROLE", "")
        if admin_role:
            try:
                sts_client = session.client("sts")
                arn = sts_client.get_caller_identity().get("Arn", "")
                is_admin = f"role/{admin_role}" in arn
            except Exception:
                logging.warning("Could not determine admin role from STS")

    if requires_admin and not is_admin:
        raise ValueError(
            "This action requires administrator permissions. Current role is not an admin role."
        )
    return bucket, is_admin


def get_qaihm_s3_or_exit(
    bucket_name: str, requires_admin: bool = False
) -> tuple[Bucket, bool]:
    """
    Get boto3 objects for interacting with the given bucket using QAIHM credentials.
    Prints an error message and exits if credentials do not exist.

    Parameters
    ----------
    bucket_name
        Name of the s3 bucket to get objects for.
    requires_admin
        Whether admin permissions are required.

    Returns
    -------
    bucket : Bucket
        Bucket object for the specified S3 bucket.
    is_admin : bool
        Whether the current credentials have admin permissions.
    """
    try:
        return get_qaihm_s3(bucket_name, requires_admin)
    except ValueError as e:
        print(e)
        sys.exit(1)


def get_files_to_upload_remove(
    s3_bucket: Bucket,
    s3_root: str,
    local_files_to_upload: list[str] | None = None,
    remote_files_to_remove: list[str] | None = None,
    allow_overwrite: bool = False,
) -> tuple[
    list[ObjectSummary],
    list[tuple[str, str]],
    list[ObjectSummary],
    dict[str, ObjectSummary],
]:
    """
    Calculates what files to upload or remove on S3 based on the given parameters.

    Parameters
    ----------
    s3_bucket
        s3 bucket to modify
    s3_root
        directory inside the s3 bucket that should be modified
    local_files_to_upload
        paths (absolute, or relative to cwd) to local files & folders to be uploaded.
        Files are uploaded directly into the s3_root folder.

        Folders are copied into s3_root recursively.
        For example, local path /path/to/a_folder will be uploaded to <s3_root>/a_folder.
        The folder's containing files/folders will be uploaded (recursively) to S3.
        If the folder already exists on s3, the existing folder on s3 will be merged with the uploaded folder.
    remote_files_to_remove
        paths to remote files that should be deleted.
        These must be relative to s3_root. "absolute" S3 keys are not valid.
    allow_overwrite
        If true, uploaded files will replace existing files on s3.
        If false, files that exist already on s3 will not be replaced (upload skipped).

    Returns
    -------
    unmodified_objects : list[ObjectSummary]
        Existing objects in s3_root that will not be modified (added / removed).
    files_to_upload : list[tuple[str, str]]
        Files to upload to S3. Each tuple contains [local file path, s3 asset key].
    objects_to_remove : list[ObjectSummary]
        S3 Objects to be removed.
    files_skipped : dict[str, ObjectSummary]
        Files to upload that will be skipped because they exist already on S3.
        Mapping is { Local File Path: Associated Existing S3 Object }.
        Note: These S3 objects will also appear in the "unmodified objects" return value.

    """
    s3_root = s3_root.removesuffix("/")

    # map<s3 path (relative to s3_root), Object Summary>
    s3_objects_by_relative_path = {
        asset.key[len(s3_root) + 1 :]: asset
        for asset in list_s3_files_in_folder_recursive(s3_bucket, s3_root)
    }

    s3_objs_to_remove: list[ObjectSummary] = []
    if remote_files_to_remove:
        for file in remote_files_to_remove:
            if file in s3_objects_by_relative_path:
                # exact match
                matching_assets = [(file, s3_objects_by_relative_path[file])]
            else:
                # regex match
                matching_assets = [
                    (k, v)
                    for k, v in s3_objects_by_relative_path.items()
                    if re.search(file, k)
                ]

            if not matching_assets:
                print(f"Could not find matching assets to remove for {file}")

            for asset_rel_path, asset in matching_assets:
                del s3_objects_by_relative_path[asset_rel_path]
                s3_objs_to_remove.append(asset)

    files_to_upload: list[tuple[str, str]] = []
    files_skipped_due_to_overwrite: dict[str, ObjectSummary] = {}
    if local_files_to_upload:
        # local path : s3 Path
        for file in local_files_to_upload:
            if not os.path.exists(file):
                raise ValueError(f"File {file} does not exist")

            def _process_file(local_file_path: str, s3_relative_path: str) -> None:
                asset_key = os.path.join(s3_root, s3_relative_path)
                if s3_relative_path not in s3_objects_by_relative_path:
                    files_to_upload.append((local_file_path, asset_key))
                elif not allow_overwrite:
                    files_skipped_due_to_overwrite[local_file_path] = (
                        s3_objects_by_relative_path[s3_relative_path]
                    )

                else:
                    s3_objs_to_remove.append(
                        s3_objects_by_relative_path[s3_relative_path]
                    )
                    del s3_objects_by_relative_path[s3_relative_path]
                    files_to_upload.append((local_file_path, asset_key))

            # If this is a file, add it to the list of files to upload.
            # If it's a folder, grab all regular files inside and add them to files_to_upload.
            if not os.path.isdir(file):
                _process_file(file, os.path.basename(file))
            else:
                file = file.removesuffix("/")
                s3_root_dirname = os.path.basename(file)
                for root, _, files in os.walk(file):
                    subroot = root[len(file) + 1 :]
                    for subfile in files:
                        _process_file(
                            os.path.join(root, subfile),
                            os.path.join(s3_root_dirname, subroot, subfile),
                        )

    return (
        list(s3_objects_by_relative_path.values()),
        files_to_upload,
        s3_objs_to_remove,
        files_skipped_due_to_overwrite,
    )

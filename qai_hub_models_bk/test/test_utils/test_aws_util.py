# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

from qai_hub_models.utils.asset_loaders import EXECUTING_IN_CI_ENVIRONMENT
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    get_files_to_upload_remove,
    get_qaihm_s3,
    list_s3_files_in_folder_recursive,
)


def test_list_s3_files_in_folder_recursive() -> None:
    file = MagicMock(key="s3://myfolder/myfile")

    # Mock around use of the S3 bucket list API
    # Testing the S3 API is of scope for this unit testing
    aws_preuploaded_files = [
        MagicMock(key="s3://myfolder/"),
        file,
        MagicMock(key="s3://myfolder/subfolder/"),
    ]
    mock_objs = MagicMock()
    mock_objs.filter.return_value = aws_preuploaded_files

    # list s3 files should strip out folders
    assert list_s3_files_in_folder_recursive(MagicMock(objects=mock_objs), "") == [file]


def test_get_files_to_upload_remove_single_files() -> None:
    # Mock around use of the S3 bucket list API
    # Testing the S3 API is of scope for this unit testing

    aws_preuploaded_files = [
        MagicMock(key="myfolder/myfile"),
        MagicMock(key="myfolder/subfolder/myfile"),
    ]
    patch_list_s3 = mock.patch(
        "qai_hub_models.utils.aws.list_s3_files_in_folder_recursive",
        return_value=aws_preuploaded_files,
    )
    patch_os = mock.patch("os.path.exists", return_value=True)

    # No changes
    with patch_list_s3:
        (
            unmodified_aws_files,
            files_to_upload,
            s3_files_to_remove,
            local_files_skipped,
        ) = get_files_to_upload_remove(MagicMock(), "myfolder")
        assert unmodified_aws_files == aws_preuploaded_files
        assert len(files_to_upload) == 0
        assert len(s3_files_to_remove) == 0
        assert len(local_files_skipped) == 0

    # Files to be added and removed
    with patch_list_s3, patch_os:
        (
            unmodified_aws_files,
            files_to_upload,
            s3_files_to_remove,
            local_files_skipped,
        ) = get_files_to_upload_remove(
            MagicMock(),
            "myfolder",
            local_files_to_upload=["apple.txt"],
            remote_files_to_remove=["myfile", "subfolder/myfile"],
        )
        assert unmodified_aws_files == []
        assert files_to_upload == [("apple.txt", "myfolder/apple.txt")]
        assert s3_files_to_remove == aws_preuploaded_files
        assert len(local_files_skipped) == 0

    # Files to be replaced
    with patch_list_s3, patch_os:
        (
            unmodified_aws_files,
            files_to_upload,
            s3_files_to_remove,
            local_files_skipped,
        ) = get_files_to_upload_remove(
            MagicMock(),
            "myfolder",
            local_files_to_upload=["myfile"],
            allow_overwrite=True,
        )
        assert unmodified_aws_files == [aws_preuploaded_files[1]]
        assert files_to_upload == [("myfile", "myfolder/myfile")]
        assert s3_files_to_remove == [aws_preuploaded_files[0]]
        assert len(local_files_skipped) == 0

    # Files to be replaced (upload skipped because file exists already)
    with patch_list_s3, patch_os:
        (
            unmodified_aws_files,
            files_to_upload,
            s3_files_to_remove,
            local_files_skipped,
        ) = get_files_to_upload_remove(
            MagicMock(),
            "myfolder",
            local_files_to_upload=["myfile"],
            allow_overwrite=False,
        )
        assert unmodified_aws_files == aws_preuploaded_files
        assert len(files_to_upload) == 0
        assert len(s3_files_to_remove) == 0
        assert local_files_skipped == {"myfile": aws_preuploaded_files[0]}


def test_get_files_to_upload_remove_folders() -> None:
    # Same test as above, but with folder instead of file

    # Mock around use of the S3 bucket list API
    # Testing the S3 API is of scope for this unit testing
    aws_preuploaded_files = [
        MagicMock(key="myfolder/myfile"),
        MagicMock(key="myfolder/subfolder/myfile"),
    ]
    patch_list_s3 = mock.patch(
        "qai_hub_models.utils.aws.list_s3_files_in_folder_recursive",
        return_value=aws_preuploaded_files,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        subfolder = tmpdir + "/subfolder"
        subfolder_depth_2 = subfolder + "/subfolder_depth_2"
        os.makedirs(subfolder_depth_2)
        (Path(subfolder) / "myfile").touch()
        (Path(subfolder_depth_2) / "myfile2").touch()

        # Files to be added and removed
        with patch_list_s3:
            (
                unmodified_aws_files,
                files_to_upload,
                s3_files_to_remove,
                local_files_skipped,
            ) = get_files_to_upload_remove(
                MagicMock(),
                "myfolder",
                local_files_to_upload=[subfolder],
                remote_files_to_remove=["subfolder/myfile"],
            )
            assert unmodified_aws_files == [aws_preuploaded_files[0]]
            assert files_to_upload == [
                (subfolder + "/myfile", "myfolder/subfolder/myfile"),
                (
                    subfolder_depth_2 + "/myfile2",
                    "myfolder/subfolder/subfolder_depth_2/myfile2",
                ),
            ]
            assert s3_files_to_remove == [aws_preuploaded_files[1]]
            assert len(local_files_skipped) == 0

        # Files to be replaced
        with patch_list_s3:
            (
                unmodified_aws_files,
                files_to_upload,
                s3_files_to_remove,
                local_files_skipped,
            ) = get_files_to_upload_remove(
                MagicMock(),
                "myfolder",
                local_files_to_upload=[subfolder],
                allow_overwrite=True,
            )
            assert unmodified_aws_files == [aws_preuploaded_files[0]]
            assert files_to_upload == [
                (subfolder + "/myfile", "myfolder/subfolder/myfile"),
                (
                    subfolder_depth_2 + "/myfile2",
                    "myfolder/subfolder/subfolder_depth_2/myfile2",
                ),
            ]
            assert s3_files_to_remove == [aws_preuploaded_files[1]]
            assert len(local_files_skipped) == 0

        # Files to be replaced (upload skipped because file exists already)
        with patch_list_s3:
            (
                unmodified_aws_files,
                files_to_upload,
                s3_files_to_remove,
                local_files_skipped,
            ) = get_files_to_upload_remove(
                MagicMock(),
                "myfolder",
                local_files_to_upload=[subfolder],
                allow_overwrite=False,
            )
            assert unmodified_aws_files == aws_preuploaded_files
            assert files_to_upload == [
                (
                    subfolder_depth_2 + "/myfile2",
                    "myfolder/subfolder/subfolder_depth_2/myfile2",
                )
            ]
            assert len(s3_files_to_remove) == 0
            assert local_files_skipped == {
                subfolder + "/myfile": aws_preuploaded_files[1]
            }


def test_ci_private_aws_access() -> None:
    if EXECUTING_IN_CI_ENVIRONMENT:
        get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)

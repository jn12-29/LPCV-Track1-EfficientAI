# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from pathlib import Path

from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    get_qaihm_s3,
    s3_download,
)

# S3 key / local filename for QAIRT auto SDK
QAIRT_AUTO_SDK_S3_KEY = "qai-hub-models/qairt/2.42.0-auto-20260106/artifact.zip"
QAIRT_AUTO_SDK_FILENAME = "qairt_auto_sdk.zip"


def download_qairt_auto_sdk(local_path: str) -> None:
    """Download the QAIRT SDK for automotive devices from S3."""
    bucket, _ = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)
    s3_download(bucket, QAIRT_AUTO_SDK_S3_KEY, local_path)


def main() -> None:
    download_qairt_auto_sdk(str(Path.home() / QAIRT_AUTO_SDK_FILENAME))


if __name__ == "__main__":
    main()

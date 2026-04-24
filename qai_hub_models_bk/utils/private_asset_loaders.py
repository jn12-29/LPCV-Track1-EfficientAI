# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    CachedWebAsset,
    ModelZooAssetConfig,
    download_from_private_s3,
)
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    can_access_private_s3,
)
from qai_hub_models.utils.path_helpers import is_internal_repo


class CachedPrivateAsset(CachedWebAsset):
    """
    Cached asset that is only available via the private S3 bucket.

    The asset is downloaded when the user has a ``qaihm`` AWS profile
    configured (via ``scripts/aws/validate_credentials.py``).

    Internal users (detected via git remote URL) without the profile
    are prompted to set up credentials. External users get the
    ``access_denied_error`` (e.g. manual download instructions).
    """

    def __init__(
        self,
        private_s3_key: str,
        local_cache_path: Path,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
        access_denied_error: Exception | None = None,
        local_cache_extracted_path: str | Path | None = None,
    ) -> None:
        """
        Parameters
        ----------
        private_s3_key
            Key inside the private S3 bucket
            (e.g. `"qai-hub-models/datasets/foo/bar.zip"`).
        local_cache_path
            Path to store the downloaded asset on disk.
        asset_config
            Asset config to use to save this file.
        access_denied_error
            Custom error to raise when the user cannot access private S3.
            If `None`, a default `ValueError` is used.
        local_cache_extracted_path
            Path where extracted archive contents will live.
            Defaults to ``local_cache_path`` without its extension.
        """
        self.access_denied_error = access_denied_error
        url = f"s3://{QAIHM_PRIVATE_S3_BUCKET}/{private_s3_key}"
        super().__init__(
            url,
            local_cache_path,
            asset_config,
            download_from_private_s3,
            local_cache_extracted_path=local_cache_extracted_path,
        )

    def fetch(
        self,
        extract: bool = False,
        local_path: str | Path | None = None,
    ) -> Path:
        if not local_path and not self.is_fetched and not can_access_private_s3():
            # No CI env and no AWS profile. Show a helpful error:
            # - Internal repo users get a credential setup prompt
            # - External users get the access_denied_error (e.g., manual download steps)
            if is_internal_repo():
                raise ValueError(
                    "You appear to be using the internal repository but have not "
                    "set up AWS credentials for private asset downloads.\n"
                    "Run `python scripts/aws/validate_credentials.py` to configure them."
                )
            raise self.access_denied_error or ValueError(
                f"This is a private asset ({self.url}) and cannot be accessed "
                "without valid Qualcomm AWS credentials."
            )
        return Path(super().fetch(extract=extract, local_path=local_path))


class UnfetchableDatasetError(Exception):
    def __init__(self, dataset_name: str, installation_steps: list[str] | None) -> None:
        """
        Create an error for datasets that cannot be automatically fetched in code.
        These datasets often require a login, license agreement acceptance, etc., to download.

        Parameters
        ----------
        dataset_name
            The name of the dataset being fetched.

        installation_steps
            Steps required for a 3rd party user to install this dataset manually.
            If None, the dataset is assumed to be not publicly available.
        """
        self.dataset_name = dataset_name
        self.installation_steps = installation_steps
        if installation_steps is None:
            super().__init__(
                f"Dataset {dataset_name} is for Qualcomm-internal usage only. If you have reached this error message when running an export or evaluate script, please file an issue at https://github.com/qualcomm/ai-hub-models/issues."
            )
        else:
            super().__init__(
                f"To use dataset {dataset_name}, you must download it manually. Follow these steps:\n"
                + "\n".join(
                    [f"{i + 1}. {step}" for i, step in enumerate(installation_steps)]
                )
            )


class CachedPrivateDatasetAsset(CachedPrivateAsset):
    """Private S3 cached asset scoped to a dataset."""

    def __init__(
        self,
        private_s3_key: str,
        dataset_id: str,
        dataset_version: int | str,
        filename: str,
        asset_config: ModelZooAssetConfig = ASSET_CONFIG,
        installation_steps: list[str] | None = None,
        local_cache_extracted_path: str | Path | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        extracted: Path | None = None
        if local_cache_extracted_path is not None:
            extracted = asset_config.get_local_store_dataset_path(
                dataset_id, dataset_version, local_cache_extracted_path
            )
        super().__init__(
            private_s3_key,
            asset_config.get_local_store_dataset_path(
                dataset_id, dataset_version, filename
            ),
            asset_config,
            UnfetchableDatasetError(
                dataset_name=dataset_id,
                installation_steps=installation_steps,
            ),
            local_cache_extracted_path=extracted,
        )

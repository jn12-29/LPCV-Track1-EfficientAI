# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Tests that validate release-assets.yaml files.

This includes validation that S3 keys referenced in release-assets.yaml
files actually exist, which helps catch issues before release.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from qai_hub_models.configs.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    can_access_private_s3,
    get_qaihm_s3,
    s3_file_exists,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_MODELS_ROOT

# S3 keys with this prefix are created by automated CI and assumed to exist
AUTOMATED_S3_KEY_PREFIX = "pre_release_assets/gh_actions/"


def get_all_s3_keys_from_release_assets(
    model_id: str,
) -> list[tuple[str, str]]:
    """
    Extract all S3 keys from a model's release-assets.yaml.

    Returns a list of (description, s3_key) tuples for error reporting.
    """
    assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    if assets.empty:
        return []

    s3_keys: list[tuple[str, str]] = []
    for precision, precision_details in assets.precisions.items():
        # Universal assets
        for path, asset_details in precision_details.universal_assets.items():
            if asset_details.s3_key:
                desc = f"{model_id}/{precision}/{path}"
                s3_keys.append((desc, asset_details.s3_key))

        # Chipset-specific assets
        for chipset, chipset_paths in precision_details.chipset_assets.items():
            for path, asset_details in chipset_paths.items():
                if asset_details.s3_key:
                    desc = f"{model_id}/{precision}/{chipset}/{path}"
                    s3_keys.append((desc, asset_details.s3_key))

    return s3_keys


def get_models_with_release_assets() -> list[str]:
    """Get list of model IDs that have release-assets.yaml files."""
    return [
        model_id
        for model_id in MODEL_IDS
        if (QAIHM_MODELS_ROOT / model_id / "release-assets.yaml").exists()
    ]


def test_release_assets_yaml_schema() -> None:
    """Validate that all release-assets.yaml files conform to the schema."""
    for model_id in get_models_with_release_assets():
        try:
            QAIHMModelReleaseAssets.from_model(model_id)
        except Exception as err:  # noqa: PERF203
            raise AssertionError(
                f"{model_id} release-assets.yaml validation failed: {err!s}"
            ) from err


@pytest.mark.skipif(
    not can_access_private_s3(),
    reason="Requires AWS credentials to access private S3 bucket",
)
def test_release_assets_s3_keys_exist() -> None:
    """
    Validate that all S3 keys in release-assets.yaml files exist.

    This test iterates through all models with release-assets.yaml
    and verifies that every referenced S3 key exists in the bucket.
    Collects all errors to report them at once rather than failing
    on the first missing key.

    Optimizations:
    - Skips automated keys (gh_actions/) which are assumed to exist
    - Uses threading to parallelize S3 HEAD requests
    """
    bucket, _ = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)

    # Collect all keys to check (excluding automated ones)
    keys_to_check: list[tuple[str, str]] = []
    for model_id in get_models_with_release_assets():
        for desc, s3_key in get_all_s3_keys_from_release_assets(model_id):
            # Skip automated keys - they are created by CI and assumed to exist
            if not s3_key.startswith(AUTOMATED_S3_KEY_PREFIX):
                keys_to_check.append((desc, s3_key))

    if not keys_to_check:
        # No manual keys to check
        return

    missing_keys_by_model: dict[str, list[str]] = {}

    def check_key(desc: str, s3_key: str) -> tuple[str, str] | None:
        if not s3_file_exists(bucket, s3_key):
            # Extract model_id from desc (format: model_id/precision/...)
            model_id, rest = desc.split("/", maxsplit=1)
            return (model_id, f"  - {rest}: {s3_key}")
        return None

    # Run checks in parallel with limited thread count
    with ThreadPoolExecutor(max_workers=min(64, len(keys_to_check))) as pool:
        futures = [
            pool.submit(check_key, desc, s3_key) for desc, s3_key in keys_to_check
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                model_id, msg = result
                if model_id not in missing_keys_by_model:
                    missing_keys_by_model[model_id] = []
                missing_keys_by_model[model_id].append(msg)

    if missing_keys_by_model:
        total_missing = sum(len(v) for v in missing_keys_by_model.values())
        lines = [
            f"Found {total_missing} missing S3 keys in release-assets.yaml files:\n"
        ]
        for model_id, missing_keys in sorted(missing_keys_by_model.items()):
            lines.append(f"{model_id}:")
            lines.extend(missing_keys)
            lines.append("")
        pytest.fail("\n".join(lines))

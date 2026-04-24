# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models_cli import common as cli_common
from qai_hub_models_cli import fetch as cli_fetch

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG


def test_cli_runtimes_match_models() -> None:
    """CLI TargetRuntime enum must stay in sync with models TargetRuntime."""
    models_values = {rt.value for rt in TargetRuntime}
    cli_values = {rt.value for rt in cli_common.TargetRuntime}
    assert models_values == cli_values, (
        f"Mismatch between models and CLI TargetRuntime enums.\n"
        f"  In models but not CLI: {models_values - cli_values}\n"
        f"  In CLI but not models: {cli_values - models_values}"
    )


def test_cli_precisions_match_models() -> None:
    """CLI Precision enum must stay in sync with models Precision named instances."""
    models_values = {
        str(v) for v in vars(Precision).values() if isinstance(v, Precision)
    }
    cli_values = {p.value for p in cli_common.Precision}
    assert models_values == cli_values, (
        f"Mismatch between models and CLI Precision.\n"
        f"  In models but not CLI: {models_values - cli_values}\n"
        f"  In CLI but not models: {cli_values - models_values}"
    )


def test_cli_asset_urls_match_models() -> None:
    """CLI asset URL constants must stay in sync with models asset config."""
    assert cli_common.STORE_URL.rstrip("/") == ASSET_CONFIG.asset_url.rstrip("/")
    assert cli_common.ASSET_FOLDER.strip(
        "/"
    ) == ASSET_CONFIG.released_asset_folder.strip("/")
    assert ASSET_CONFIG.released_asset_filename == cli_fetch.ASSET_FILENAME
    assert (
        ASSET_CONFIG.released_asset_with_chipset_filename
        == cli_fetch.ASSET_CHIPSET_FILENAME
    )

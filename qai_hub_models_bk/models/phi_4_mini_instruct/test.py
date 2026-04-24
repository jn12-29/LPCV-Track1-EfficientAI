# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import pytest
import qai_hub as hub

from qai_hub_models.models._shared.llm import test
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.models.phi_4_mini_instruct import (
    MODEL_ID,
    Model,
)
from qai_hub_models.models.phi_4_mini_instruct.export import (
    DEFAULT_EXPORT_DEVICE,
    NUM_SPLITS,
)
from qai_hub_models.models.phi_4_mini_instruct.export import (
    main as export_main,
)
from qai_hub_models.models.phi_4_mini_instruct.model import (
    DEFAULT_PRECISION,
)
from qai_hub_models.utils.model_cache import CacheMode


@pytest.mark.unmarked
@pytest.mark.parametrize(
    ("skip_inferencing", "skip_profiling", "target_runtime"),
    [
        (True, True, TargetRuntime.GENIE),
        (True, False, TargetRuntime.GENIE),
        (False, True, TargetRuntime.GENIE),
        (False, False, TargetRuntime.GENIE),
    ],
)
def test_cli_device_with_skips(
    tmp_path: Path,
    skip_inferencing: bool,
    skip_profiling: bool,
    target_runtime: TargetRuntime,
) -> None:
    test.test_cli_device_with_skips(
        export_main,
        Model,
        tmp_path,
        MODEL_ID,
        NUM_SPLITS,
        hub.Device(DEFAULT_EXPORT_DEVICE),
        skip_inferencing,
        skip_profiling,
        target_runtime,
        precision=DEFAULT_PRECISION,
    )


@pytest.mark.unmarked
@pytest.mark.parametrize(
    ("chipset", "context_length", "sequence_length", "target_runtime"),
    [
        ("qualcomm-snapdragon-8gen2", 2048, 256, TargetRuntime.GENIE),
        ("qualcomm-snapdragon-x-elite", 4096, 128, TargetRuntime.GENIE),
    ],
)
def test_cli_chipset_with_options(
    tmp_path: Path,
    context_length: int,
    sequence_length: int,
    chipset: str,
    target_runtime: TargetRuntime,
) -> None:
    test.test_cli_chipset_with_options(
        export_main,
        Model,
        tmp_path,
        MODEL_ID,
        NUM_SPLITS,
        chipset,
        context_length,
        sequence_length,
        target_runtime,
    )


@pytest.mark.unmarked
@pytest.mark.parametrize(
    ("cache_mode", "skip_download", "skip_summary", "target_runtime"),
    [
        (CacheMode.ENABLE, True, True, TargetRuntime.GENIE),
        (CacheMode.DISABLE, True, False, TargetRuntime.GENIE),
        (CacheMode.OVERWRITE, False, False, TargetRuntime.GENIE),
    ],
)
def test_cli_default_device_select_component(
    tmp_path: Path,
    cache_mode: CacheMode,
    skip_download: bool,
    skip_summary: bool,
    target_runtime: TargetRuntime,
) -> None:
    test.test_cli_default_device_select_component(
        export_main,
        Model,
        tmp_path,
        MODEL_ID,
        NUM_SPLITS,
        hub.Device(DEFAULT_EXPORT_DEVICE),
        cache_mode,
        skip_download,
        skip_summary,
        target_runtime,
        decode_sequence_length=1,
        precision=DEFAULT_PRECISION,
    )


def test_cli_device_with_skips_unsupported_context_length(tmp_path: Path) -> None:
    test.test_cli_device_with_skips_unsupported_context_length(
        export_main, Model, tmp_path, MODEL_ID
    )

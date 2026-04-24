# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.envvars import (
    EnabledPathsEnvvar,
    EnabledPrecisionsEnvvar,
    IgnoreKnownFailuresEnvvar,
    SpecialPathSetting,
    SpecialPrecisionSetting,
)
from qai_hub_models.scorecard.execution_helpers import (
    get_compile_parameterized_pytest_config,
    get_profile_parameterized_pytest_config,
    get_quantize_parameterized_pytest_config,
)


@pytest.fixture(autouse=True)
def set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    IgnoreKnownFailuresEnvvar.patchenv(monkeypatch, True)


def test_get_quantize_precisions(monkeypatch: pytest.MonkeyPatch) -> None:
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "",
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
    )
    assert set(quantize_precisions) == {Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT, "w8a8"}
    )
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "",
        {k: [] for k in [Precision.float]},
        {k: [] for k in [Precision.float]},
    )
    assert set(quantize_precisions) == {Precision.w8a8}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_MINUS_FLOAT}
    )
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "",
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
    )
    assert set(quantize_precisions) == {Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_QUANTIZED}
    )
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "",
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
    )
    assert set(quantize_precisions) == {Precision.w8a16}

    quantize_precisions = get_quantize_parameterized_pytest_config(
        "",
        {k: [] for k in [Precision.float]},
        {k: [] for k in [Precision.float]},
    )
    assert set(quantize_precisions) == {Precision.w8a8}


def test_get_compile_precisions(monkeypatch: pytest.MonkeyPatch) -> None:
    EnabledPathsEnvvar.patchenv(monkeypatch, {"qnn_context_binary"})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})
    precision_mapping = {
        k: [TargetRuntime.QNN_CONTEXT_BINARY]
        for k in [Precision.float, Precision.w8a8, Precision.w8a16]
    }
    compile_paths = get_compile_parameterized_pytest_config(
        "", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float, Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_MINUS_FLOAT}
    )
    compile_paths = get_compile_parameterized_pytest_config(
        "", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_QUANTIZED}
    )
    precision_mapping = {
        k: [TargetRuntime.QNN_CONTEXT_BINARY] for k in [Precision.float]
    }
    compile_paths = get_compile_parameterized_pytest_config(
        "", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.w8a8}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT, "w8a8"}
    )
    compile_paths = get_compile_parameterized_pytest_config(
        "", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float, Precision.w8a8}

    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.BENCH})
    precision_mapping = {
        k: [TargetRuntime.QNN_CONTEXT_BINARY] for k in [Precision.float, Precision.w8a8]
    }
    compile_paths = get_compile_parameterized_pytest_config(
        "", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float}

    compile_paths = get_compile_parameterized_pytest_config(
        "resnet18", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float, Precision.w8a8}


# ---- Tests for should_run_path_for_model ----

JIT_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.float: [TargetRuntime.QNN_DLC, TargetRuntime.ONNX, TargetRuntime.TFLITE],
}
AOT_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.float: [
        TargetRuntime.QNN_CONTEXT_BINARY,
        TargetRuntime.PRECOMPILED_QNN_ONNX,
    ],
}


def test_default_paths_jit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default paths for JIT models include JIT paths, not AOT."""
    EnabledPathsEnvvar.patchenv(monkeypatch, {SpecialPathSetting.DEFAULT})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    profile_paths = get_profile_parameterized_pytest_config(
        "", JIT_RUNTIMES, JIT_RUNTIMES
    )
    path_values = {p[1].value for p in profile_paths}
    assert "qnn_dlc" in path_values
    assert "onnx" in path_values
    assert "tflite" in path_values
    assert "qnn_context_binary" not in path_values
    assert "precompiled_qnn_onnx" not in path_values


def test_default_paths_aot_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default paths for AOT models include AOT paths, not JIT."""
    EnabledPathsEnvvar.patchenv(monkeypatch, {SpecialPathSetting.DEFAULT})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    profile_paths = get_profile_parameterized_pytest_config(
        "", AOT_RUNTIMES, AOT_RUNTIMES
    )
    path_values = {p[1].value for p in profile_paths}
    assert "qnn_context_binary" in path_values
    assert "precompiled_qnn_onnx" in path_values
    assert "qnn_dlc" not in path_values
    assert "onnx" not in path_values
    assert "tflite" not in path_values


def test_explicit_ep_path_aot_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicitly requested qnn_dlc_via_qnn_ep works for AOT models."""
    EnabledPathsEnvvar.patchenv(
        monkeypatch, {SpecialPathSetting.DEFAULT, "qnn_dlc_via_qnn_ep"}
    )
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    profile_paths = get_profile_parameterized_pytest_config(
        "", AOT_RUNTIMES, AOT_RUNTIMES
    )
    path_values = {p[1].value for p in profile_paths}
    assert "qnn_dlc_via_qnn_ep" in path_values
    assert "qnn_context_binary" in path_values
    assert "precompiled_qnn_onnx" in path_values
    # Regular qnn_dlc NOT present (only via default, runtime not in supported)
    assert "qnn_dlc" not in path_values


def test_explicit_ep_path_jit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicitly requested qnn_dlc_via_qnn_ep also works for JIT models."""
    EnabledPathsEnvvar.patchenv(
        monkeypatch, {SpecialPathSetting.DEFAULT, "qnn_dlc_via_qnn_ep"}
    )
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    profile_paths = get_profile_parameterized_pytest_config(
        "", JIT_RUNTIMES, JIT_RUNTIMES
    )
    path_values = {p[1].value for p in profile_paths}
    assert "qnn_dlc_via_qnn_ep" in path_values
    assert "qnn_dlc" in path_values
    assert "onnx" in path_values
    assert "tflite" in path_values


def test_engine_prefix_aot_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine prefix 'qnn' only enables paths whose runtime is in supported."""
    EnabledPathsEnvvar.patchenv(monkeypatch, {"qnn"})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    profile_paths = get_profile_parameterized_pytest_config(
        "", AOT_RUNTIMES, AOT_RUNTIMES
    )
    path_values = {p[1].value for p in profile_paths}
    # qnn_context_binary runtime IS in supported runtimes
    assert "qnn_context_binary" in path_values
    # qnn_dlc runtime is NOT in supported runtimes (strict match)
    assert "qnn_dlc" not in path_values


def test_engine_prefix_jit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine prefix 'qnn' only enables paths whose runtime is in supported."""
    EnabledPathsEnvvar.patchenv(monkeypatch, {"qnn"})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    profile_paths = get_profile_parameterized_pytest_config(
        "", JIT_RUNTIMES, JIT_RUNTIMES
    )
    path_values = {p[1].value for p in profile_paths}
    assert "qnn_dlc" in path_values
    assert "qnn_context_binary" not in path_values

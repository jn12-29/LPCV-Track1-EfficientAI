# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import importlib.util
import sys
import types
from pathlib import Path

# changes.py lives in scripts/tasks/ outside the installed package.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TASKS_DIR = _REPO_ROOT / "scripts" / "tasks"

# Register 'tasks' as a namespace package so submodule imports work.
_tasks_pkg = types.ModuleType("tasks")
_tasks_pkg.__path__ = [str(_TASKS_DIR)]
sys.modules["tasks"] = _tasks_pkg


def _load(name: str) -> types.ModuleType:
    path = _TASKS_DIR / (name.split(".")[-1] + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("tasks.constants")
_load("tasks.github")
_load("tasks.plan")
_load("tasks.util")
_changes = _load("tasks.changes")

get_python_import_expression = _changes.get_python_import_expression
resolve_affected_models = _changes.resolve_affected_models


def test_import_expression_matches_real_imports() -> None:
    """
    Verify get_python_import_expression produces import paths that
    actually appear in the codebase (no stale 'src.' prefix, etc.).
    """
    filepath = "src/qai_hub_models/models/mobilenet_v2/model.py"
    expr = get_python_import_expression(filepath)

    assert not expr.startswith("src."), (
        f"Import expression '{expr}' has 'src.' prefix — "
        "imports in code use 'qai_hub_models.*', not 'src.qai_hub_models.*'"
    )
    assert expr == "qai_hub_models.models.mobilenet_v2.model"


def test_import_expression_for_shared_module() -> None:
    """Shared modules under _shared/ should also resolve correctly."""
    filepath = "src/qai_hub_models/models/_shared/stable_diffusion/model.py"
    expr = get_python_import_expression(filepath)
    assert expr == "qai_hub_models.models._shared.stable_diffusion.model"


def test_import_expression_for_init() -> None:
    """__init__.py should strip the filename."""
    filepath = "src/qai_hub_models/models/mobilenet_v2/__init__.py"
    expr = get_python_import_expression(filepath)
    assert expr == "qai_hub_models.models.mobilenet_v2"


def test_shared_model_change_detects_dependent_models() -> None:
    """
    Changing a _shared module should detect the concrete models
    that import from it.
    """
    changed = ["src/qai_hub_models/models/_shared/stable_diffusion/model.py"]
    models = resolve_affected_models(changed)
    assert "stable_diffusion_v1_5" in models
    assert "controlnet_canny" in models


# ── End-to-end: simulated git diffs → affected models ──────────────


def test_direct_model_file_change() -> None:
    """Changing a model's own model.py should detect that model."""
    models = resolve_affected_models(
        ["src/qai_hub_models/models/mobilenet_v2/model.py"]
    )
    assert "mobilenet_v2" in models


def test_direct_export_change() -> None:
    """Changing a model's export.py should detect that model."""
    models = resolve_affected_models(
        ["src/qai_hub_models/models/mobilenet_v2/export.py"]
    )
    assert "mobilenet_v2" in models


def test_model_change_propagates_to_export() -> None:
    """
    Changing model.py should also flag export.py (via the manual edge
    in _get_file_edges), so the model is detected even when
    include_model=False.
    """
    models = resolve_affected_models(
        ["src/qai_hub_models/models/mobilenet_v2/model.py"],
        include_model=False,
    )
    assert "mobilenet_v2" in models


def test_shared_whisper_change_detects_whisper_models() -> None:
    """Changing _shared/hf_whisper should detect whisper variants."""
    models = resolve_affected_models(
        ["src/qai_hub_models/models/_shared/hf_whisper/model.py"]
    )
    assert "whisper_tiny" in models


def test_utility_with_manual_edges_detects_representative_models() -> None:
    """
    Changing a utility file listed in MANUAL_EDGES should detect
    the representative models (DFS is short-circuited).
    """
    models = resolve_affected_models(["src/qai_hub_models/utils/input_spec.py"])
    assert "sinet" in models
    assert "whisper_tiny" in models


def test_base_model_change_detects_sd() -> None:
    """
    Changing base_model.py should detect stable_diffusion_v1_5
    via its MANUAL_EDGES entry.
    """
    models = resolve_affected_models(["src/qai_hub_models/utils/base_model.py"])
    assert "stable_diffusion_v1_5" in models
    assert "sinet" in models
    assert "whisper_tiny" in models


def test_unrelated_file_detects_nothing() -> None:
    """Changing a file outside qai_hub_models should detect no models."""
    models = resolve_affected_models(["scripts/tasks/changes.py"])
    assert len(models) == 0


def test_shared_change_does_not_detect_shared_as_model() -> None:
    """
    _shared/ directories are not models — they should never appear
    in the output even though they're under models/.
    """
    models = resolve_affected_models(
        ["src/qai_hub_models/models/_shared/stable_diffusion/model.py"]
    )
    for m in models:
        assert not m.startswith("_shared")


def test_model_inherited_by_others_detects_dependents() -> None:
    """
    SAM's model.py is imported by mobilesam, edgetam, etc.
    Changing it should detect those downstream models.
    """
    models = resolve_affected_models(["src/qai_hub_models/models/sam/model.py"])
    assert "sam" in models
    assert "mobilesam" in models


def test_utils_not_in_manual_edges_fans_out() -> None:
    """
    Utils files NOT in MANUAL_EDGES should follow real imports
    and find all affected models (potentially many).
    """
    models = resolve_affected_models(["src/qai_hub_models/utils/checkpoint.py"])
    # checkpoint.py is imported by SD models, LLM models, etc.
    assert "stable_diffusion_v1_5" in models
    assert len(models) > 5


def test_aimet_onnx_utils_change_detects_sd() -> None:
    """
    Changing quantization_aimet_onnx.py should detect stable_diffusion_v1_5
    via its MANUAL_EDGES entry for AIMET coverage.
    """
    models = resolve_affected_models(
        ["src/qai_hub_models/utils/quantization_aimet_onnx.py"]
    )
    assert "stable_diffusion_v1_5" in models

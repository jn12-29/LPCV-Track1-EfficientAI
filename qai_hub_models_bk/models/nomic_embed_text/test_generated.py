# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
import qai_hub as hub
import torch

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.nomic_embed_text import MODEL_ID, Model
from qai_hub_models.models.nomic_embed_text import export as model_export_module
from qai_hub_models.models.nomic_embed_text.export import (
    compile_model,
    export_model,
    inference_model,
    link_model,
    profile_model,
    quantize_model,
)
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.execution_helpers import (
    get_compile_parameterized_pytest_config,
    get_evaluation_parameterized_pytest_config,
    get_export_parameterized_pytest_config,
    get_link_parameterized_pytest_config,
    get_profile_parameterized_pytest_config,
    get_quantize_parameterized_pytest_config,
    needs_pre_quantize_compile,
    pytest_device_idfn,
)
from qai_hub_models.utils.args import get_model_kwargs
from qai_hub_models.utils.testing import skip_invalid_runtime_device, verify_io_names
from qai_hub_models.utils.testing_async_utils import CachedScorecardJobError
from qai_hub_models.utils.testing_export_eval import (
    accuracy_on_dataset_via_evaluate_and_export,
    accuracy_on_sample_inputs_via_export,
    compile_via_export,
    export_test_e2e,
    inference_via_export,
    link_via_export,
    on_device_inference_for_accuracy_validation,
    pre_quantize_compile_via_export,
    profile_via_export,
    quantize_via_export,
    sim_accuracy_on_dataset,
    split_and_group_accuracy_validation_output_batches,
    torch_accuracy_on_dataset,
    torch_inference_for_accuracy_validation,
    torch_inference_for_accuracy_validation_outputs,
)

# All runtime + precision pairs that are enabled for testing and are compatibile with this model.
# NOTE:
#   Certain supported pairs may be excluded from this list if they are not enabled for testing.
#   For example, models that allow JIT (on-device) compile will not test AOT runtimes; we assume that if it works on JIT it will work on AOT.
ENABLED_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.float: [
        TargetRuntime.TFLITE,
        TargetRuntime.QNN_DLC,
        TargetRuntime.ONNX,
    ],
}


# All runtime + precision pairs that are enabled for testing and have no known failure reasons.
# NOTE:
#   Certain supported pairs may be excluded from this list if they are not enabled for testing.
#   For example, models that allow JIT (on-device) compile will not test AOT runtimes; we assume that if it works on JIT it will work on AOT.
PASSING_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.float: [
        TargetRuntime.TFLITE,
        TargetRuntime.QNN_DLC,
        TargetRuntime.ONNX,
    ],
}


EVAL_DEVICE = ScorecardDevice.get("Samsung Galaxy S25 (Family)")
HAS_EVAL_DATASET = len(Model.eval_datasets()) > 0


@pytest.mark.compile
def test_verify_io_names() -> None:
    verify_io_names(Model)


@pytest.mark.pre_quantize_compile
@pytest.mark.skipif(
    not needs_pre_quantize_compile(
        MODEL_ID, ENABLED_PRECISION_RUNTIMES, PASSING_PRECISION_RUNTIMES
    ),
    reason="Model does not require pre-quantize compile step",
)
def test_pre_quantize_compile() -> None:
    pre_quantize_compile_via_export(
        compile_model,
        MODEL_ID,
        Model.from_pretrained(),
        EVAL_DEVICE,
    )


@pytest.mark.parametrize(
    "precision",
    get_quantize_parameterized_pytest_config(
        MODEL_ID, ENABLED_PRECISION_RUNTIMES, PASSING_PRECISION_RUNTIMES
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.quantize
def test_quantize(precision: Precision) -> None:
    try:
        quantize_via_export(
            quantize_model,
            MODEL_ID,
            Model.from_pretrained(),
            precision,
            EVAL_DEVICE,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_compile_parameterized_pytest_config(
        MODEL_ID, ENABLED_PRECISION_RUNTIMES, PASSING_PRECISION_RUNTIMES
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.compile
def test_compile(
    precision: Precision, scorecard_path: ScorecardCompilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        compile_via_export(
            compile_model,
            MODEL_ID,
            Model.from_pretrained(),
            precision,
            scorecard_path,
            device,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_link_parameterized_pytest_config(
        MODEL_ID, ENABLED_PRECISION_RUNTIMES, PASSING_PRECISION_RUNTIMES
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.link
def test_link(
    precision: Precision, scorecard_path: ScorecardCompilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        link_via_export(
            link_model,
            MODEL_ID,
            Model.from_pretrained(),
            precision,
            scorecard_path,
            device,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_profile_parameterized_pytest_config(
        MODEL_ID, ENABLED_PRECISION_RUNTIMES, PASSING_PRECISION_RUNTIMES
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.profile
def test_profile(
    precision: Precision, scorecard_path: ScorecardProfilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        profile_via_export(
            profile_model,
            MODEL_ID,
            Model.from_pretrained(),
            precision,
            scorecard_path,
            device,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_evaluation_parameterized_pytest_config(
        MODEL_ID,
        EVAL_DEVICE,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.inference
def test_inference(
    precision: Precision, scorecard_path: ScorecardProfilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        if HAS_EVAL_DATASET:
            on_device_inference_for_accuracy_validation(
                Model,
                Model.eval_datasets()[0],
                MODEL_ID,
                precision,
                scorecard_path,
                device,
            )
        else:
            inference_via_export(
                inference_model,
                MODEL_ID,
                Model.from_pretrained(),
                precision,
                scorecard_path,
                device,
            )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.inference
def test_val_data_torch() -> None:
    if not HAS_EVAL_DATASET:
        return
    torch_inference_for_accuracy_validation(
        Model.from_pretrained(), Model.eval_datasets()[0], MODEL_ID
    )


@pytest.fixture(scope="module")
def torch_val_outputs() -> list[np.ndarray]:
    """
    Because the below method downloads a dataset over the internet,
    it is called in a fixture so it can be reused.
    """
    if not HAS_EVAL_DATASET:
        return []
    return torch_inference_for_accuracy_validation_outputs(MODEL_ID)


@pytest.fixture(scope="module")
def torch_evaluate_mock_outputs(
    torch_val_outputs: list[np.ndarray],
) -> list[torch.Tensor | tuple[torch.Tensor, ...]]:
    """
    Because the below method does some memory movement,
    it is called in a fixture so its output can be reused.
    """
    if not HAS_EVAL_DATASET:
        return []
    return split_and_group_accuracy_validation_output_batches(torch_val_outputs)


@pytest.mark.inference
def test_torch_accuracy(
    torch_evaluate_mock_outputs: list[torch.Tensor | tuple[torch.Tensor, ...]],
) -> None:
    if not HAS_EVAL_DATASET:
        return
    torch_accuracy_on_dataset(
        Model.from_pretrained(),
        Model.eval_datasets()[0],
        torch_evaluate_mock_outputs,
        MODEL_ID,
    )


@pytest.mark.parametrize(
    "precision",
    get_quantize_parameterized_pytest_config(
        MODEL_ID, ENABLED_PRECISION_RUNTIMES, PASSING_PRECISION_RUNTIMES
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.inference
def test_sim_accuracy(
    precision: Precision,
    torch_evaluate_mock_outputs: list[torch.Tensor | tuple[torch.Tensor, ...]],
) -> None:
    if not HAS_EVAL_DATASET:
        return
    try:
        sim_accuracy_on_dataset(
            Model.from_pretrained(**get_model_kwargs(Model, dict(precision=precision))),
            Model.eval_datasets()[0],
            MODEL_ID,
            precision,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_evaluation_parameterized_pytest_config(
        MODEL_ID,
        EVAL_DEVICE,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.compute_device_accuracy
def test_val_accuracy(
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    torch_val_outputs: list[np.ndarray],
    torch_evaluate_mock_outputs: list[torch.Tensor | tuple[torch.Tensor, ...]],
) -> None:
    try:
        if HAS_EVAL_DATASET:
            accuracy_on_dataset_via_evaluate_and_export(
                export_model,
                Model.from_pretrained(
                    **get_model_kwargs(Model, dict(precision=precision))
                ),
                Model.eval_datasets()[0],
                torch_val_outputs,
                torch_evaluate_mock_outputs,
                MODEL_ID,
                precision,
                scorecard_path,
                device,
            )
        else:
            accuracy_on_sample_inputs_via_export(
                export_model,
                MODEL_ID,
                Model.from_pretrained(),
                precision,
                scorecard_path,
                device,
            )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_export_parameterized_pytest_config(
        MODEL_ID,
        EVAL_DEVICE,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
    ),
    ids=pytest_device_idfn,
)
@pytest.mark.export
def test_export(
    precision: Precision, scorecard_path: ScorecardProfilePath, device: ScorecardDevice
) -> None:
    skip_invalid_runtime_device(Model, scorecard_path.runtime, device)
    try:
        export_test_e2e(
            export_model, Model, MODEL_ID, precision, scorecard_path, device
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


# Trace the model & upload to hub only once for all module + input pairs.
# This speeds up tests and limits memory leaks for torch jit trace.
@pytest.fixture(scope="module", autouse=True)
def cached_torch_trace_for_export() -> Generator[pytest.MonkeyPatch, None, None]:
    with pytest.MonkeyPatch.context() as mp:
        model_cache: dict[str, hub.Model] = {}
        torch_trace = torch.jit.trace

        def _cached_torch_trace(
            module: torch.nn.Module,
            inputs: torch.Tensor | tuple[torch.Tensor, ...],
            check_trace: bool = True,
        ) -> hub.Model:
            model_key = str(module) + str(
                sorted([f"{x.shape}{x.type()}" for x in inputs])
            )
            model = model_cache.get(model_key)
            if not model:
                trace = torch_trace(module, inputs, check_trace=check_trace)
                model = hub.upload_model(trace)
                model_cache[model_key] = model
            assert isinstance(model, hub.Model)
            return model

        mp.setattr(model_export_module.torch.jit, "trace", _cached_torch_trace)
        yield mp

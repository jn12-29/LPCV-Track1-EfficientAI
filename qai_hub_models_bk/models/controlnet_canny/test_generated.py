# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import qai_hub as hub
import torch
from qai_hub.client import SourceModel

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.controlnet_canny import MODEL_ID, Model
from qai_hub_models.models.controlnet_canny.export import (
    compile_model,
    export_model,
    inference_model,
    link_model,
    profile_model,
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
    pytest_device_idfn,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.testing import skip_invalid_runtime_device
from qai_hub_models.utils.testing_async_utils import CachedScorecardJobError
from qai_hub_models.utils.testing_export_eval import (
    accuracy_on_sample_inputs_via_export,
    compile_via_export,
    export_test_e2e,
    inference_via_export,
    link_via_export,
    on_device_inference_for_accuracy_validation,
    profile_via_export,
    split_and_group_accuracy_validation_output_batches,
    torch_inference_for_accuracy_validation,
    torch_inference_for_accuracy_validation_outputs,
)

# All runtime + precision pairs that are enabled for testing and are compatibile with this model.
# NOTE:
#   Certain supported pairs may be excluded from this list if they are not enabled for testing.
#   For example, models that allow JIT (on-device) compile will not test AOT runtimes; we assume that if it works on JIT it will work on AOT.
ENABLED_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.w8a16: [
        TargetRuntime.QNN_CONTEXT_BINARY,
        TargetRuntime.PRECOMPILED_QNN_ONNX,
    ],
}


# All runtime + precision pairs that are enabled for testing and have no known failure reasons.
# NOTE:
#   Certain supported pairs may be excluded from this list if they are not enabled for testing.
#   For example, models that allow JIT (on-device) compile will not test AOT runtimes; we assume that if it works on JIT it will work on AOT.
PASSING_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.w8a16: [
        TargetRuntime.QNN_CONTEXT_BINARY,
        TargetRuntime.PRECOMPILED_QNN_ONNX,
    ],
}


EVAL_DEVICE = ScorecardDevice.get("Samsung Galaxy S25 (Family)")
HAS_EVAL_DATASET = len(Model.eval_datasets()) > 0


ALL_COMPONENTS = Model.component_class_names


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_compile_parameterized_pytest_config(
        MODEL_ID,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
        can_use_quantize_job=False,
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
            is_aimet=True,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_link_parameterized_pytest_config(
        MODEL_ID,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
        can_use_quantize_job=False,
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
        MODEL_ID,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
        can_use_quantize_job=False,
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
        can_use_quantize_job=False,
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


@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    get_evaluation_parameterized_pytest_config(
        MODEL_ID,
        EVAL_DEVICE,
        ENABLED_PRECISION_RUNTIMES,
        PASSING_PRECISION_RUNTIMES,
        can_use_quantize_job=False,
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
        can_use_quantize_job=False,
        requires_aot_prepare=True,
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
            export_model,
            Model,
            MODEL_ID,
            precision,
            scorecard_path,
            device,
            ALL_COMPONENTS,
        )
    except CachedScorecardJobError as e:
        pytest.skip(str(e))


# Trace the model & upload to hub only once for all module + input pairs.
# This speeds up tests and limits memory leaks for torch jit trace.
@pytest.fixture(scope="module", autouse=True)
def cached_torch_trace_for_export() -> Generator[pytest.MonkeyPatch, None, None]:
    with pytest.MonkeyPatch.context() as mp:
        model_cache: dict[str, hub.Model] = {}
        convert_to_hub_source_model = BaseModel.convert_to_hub_source_model

        def _cached_convert_to_hub_source_model(
            self: BaseModel,
            target_runtime: TargetRuntime,
            output_path: str | Path,
            input_spec: InputSpec | None = None,
            check_trace: bool = True,
        ) -> SourceModel | hub.Model:
            source_model_format = self.preferred_hub_source_model_format(target_runtime)
            model_key = str(self) + str(input_spec) + str(source_model_format)
            model = model_cache.get(model_key)
            if not model:
                source_model = convert_to_hub_source_model(
                    self, target_runtime, output_path, input_spec, check_trace
                )
                assert source_model is not None
                model = hub.upload_model(source_model)
                model_cache[model_key] = model
            return model

        mp.setattr(
            BaseModel,
            "convert_to_hub_source_model",
            _cached_convert_to_hub_source_model,
        )
        yield mp

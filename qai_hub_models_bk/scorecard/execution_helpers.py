# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeVar, cast

import qai_hub as hub

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard import ScorecardCompilePath, ScorecardProfilePath
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.envvars import (
    DisableWorkbenchJobTimeoutEnvvar,
    EnabledPrecisionsEnvvar,
    IgnoreKnownFailuresEnvvar,
    SpecialPrecisionSetting,
)
from qai_hub_models.scorecard.static.list_models import (
    get_bench_pytorch_w8a8_models,
    get_bench_pytorch_w8a16_models,
)
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


def wait_for_prerequisite_job(
    job: hub.Job, max_wait_seconds: int | None = None
) -> hub.JobStatus:
    """
    Wait for a job up to max_wait_seconds after the job was submitted and returns the status after the wait period.

    Parameters
    ----------
    job
        The job to wait for.
    max_wait_seconds
        This is the max number of seconds AFTER THE JOB'S SUBMISSION TIME to allow the job to run.
        If None, waits DisableWorkbenchJobTimeoutEnvvar.max_workbench_job_duration_seconds() seconds.
        If QAIHM_TEST_DISABLE_WORKBENCH_JOB_TIMEOUT is truthy, this method will wait indefinitely (until the job stops).

    Returns
    -------
    status: hub.JobStatus
        Status of the job at the end of the wait period.
        If the job is still running, the status will reflect that.

    """
    max_wait_seconds = (
        max_wait_seconds
        if max_wait_seconds is not None
        else DisableWorkbenchJobTimeoutEnvvar.max_workbench_job_duration_seconds()
    )
    wait_period = (
        max(0, max_wait_seconds - int(time.time() - job.date.timestamp()))
        if max_wait_seconds is not None
        else None
    )
    try:
        return job.wait(wait_period)
    except TimeoutError:
        # If we hit a timeout, return the current job status.
        return job.get_status()


def get_enabled_test_precisions() -> tuple[
    SpecialPrecisionSetting | None, list[Precision]
]:
    """
    Determine what precisions are enabled based on the test environment.

    Returns
    -------
    special_precision_setting : SpecialPrecisionSetting | None
        Any special precision setting with which the run was configured.
    extra_enabled_precisions : list[Precision]
        Precisions that should be enabled beyond the defaults, if a model supports quantize job.
    """
    precisions_set = EnabledPrecisionsEnvvar.get()
    precisions_special_settings = [
        p for p in precisions_set if isinstance(p, SpecialPrecisionSetting)
    ]
    if len(precisions_special_settings) > 1:
        raise ValueError(
            "Multiple special settings found in precision list."
            f"Cannot set both {precisions_special_settings[0].value} and {precisions_special_settings[1].value}."
        )

    return (
        precisions_special_settings[0] if precisions_special_settings else None,
        [Precision.parse(p.strip()) for p in precisions_set if isinstance(p, str)],
    )


def get_quantized_bench_models_path() -> Path:
    return QAIHM_MODELS_ROOT / "scorecard" / "static" / "pytorch_bench_models_w8a8.txt"


@lru_cache(maxsize=1)
def get_quantized_bench_models() -> set[str]:
    with open(get_quantized_bench_models_path()) as f:
        return set(f.read().strip().split("\n"))


def get_model_test_precisions(
    model_id: str,
    enabled_model_test_precisions: set[Precision],
    passing_model_test_precisions: set[Precision] | None = None,
    can_use_quantize_job: bool = True,
    include_unsupported_paths: bool | None = None,
) -> list[Precision]:
    """
    Get the list of precisions that should be tested in this environment.

    Parameters
    ----------
    model_id
        The model ID.
    enabled_model_test_precisions
        All Precisions that are enabled for testing with this model.
    passing_model_test_precisions
        All Precisions that are enabled for testing with this model and have no known failure reasons in code-gen.yaml
        If None, assumes this is the same as enabled_model_test_precisions
    can_use_quantize_job
        Whether this model can use quantize job.
    include_unsupported_paths
        If true, all enabled paths will be included, instead of the ones compatible with
        parameter supported_paths.

    Returns
    -------
    model_test_precisions : list[Precision]
        The list of precisions to test for this model.
    """
    if include_unsupported_paths is None:
        include_unsupported_paths = IgnoreKnownFailuresEnvvar.get()
    model_supported_precisions = (
        enabled_model_test_precisions
        if include_unsupported_paths or passing_model_test_precisions is None
        else passing_model_test_precisions
    )

    enabled_test_precisions = get_enabled_test_precisions()
    special_precision_setting, extra_enabled_precisions = enabled_test_precisions
    enabled_precisions: set[Precision] = set()
    if special_precision_setting in [
        SpecialPrecisionSetting.DEFAULT,
        SpecialPrecisionSetting.DEFAULT_MINUS_FLOAT,
    ]:
        # If default precisions are enabled, always run tests with default precisions.
        enabled_precisions.update(model_supported_precisions)

    if (
        special_precision_setting == SpecialPrecisionSetting.DEFAULT_MINUS_FLOAT
        and Precision.float in enabled_precisions
    ):
        enabled_precisions.remove(Precision.float)
    if special_precision_setting == SpecialPrecisionSetting.DEFAULT_QUANTIZED:
        enabled_precisions.add(
            Precision.w8a16
            if (Precision.w8a16 in model_supported_precisions)
            else Precision.w8a8
        )
    if special_precision_setting == SpecialPrecisionSetting.BENCH:
        if Precision.float in model_supported_precisions:
            enabled_precisions.add(Precision.float)
        if (
            Precision.w8a8 in model_supported_precisions
            and model_id in get_bench_pytorch_w8a8_models()
        ):
            enabled_precisions.add(Precision.w8a8)
        if (
            Precision.w8a16 in model_supported_precisions
            and model_id in get_bench_pytorch_w8a16_models()
        ):
            enabled_precisions.add(Precision.w8a16)
    if can_use_quantize_job and include_unsupported_paths:
        # If quantize job is supported, this model can run tests on any desired precision.
        enabled_precisions.update(extra_enabled_precisions)
    else:
        # If quantize job is not supported, we can still run enabled precisions that happen to be in the model's supported precisions list.
        enabled_precisions.update(
            set(model_supported_precisions).intersection(extra_enabled_precisions)
        )

    return list(enabled_precisions)


ScorecardPathTypeVar = TypeVar(
    "ScorecardPathTypeVar", ScorecardCompilePath, ScorecardProfilePath
)


def get_enabled_paths_for_testing(
    model_id: str,
    model_supported_test_paths: dict[Precision, list[TargetRuntime]],
    model_passing_test_paths: dict[Precision, list[TargetRuntime]],
    path_type: type[ScorecardPathTypeVar],
    can_use_quantize_job: bool = True,
    include_unsupported_paths: bool | None = None,
) -> dict[Precision, list[ScorecardPathTypeVar]]:
    """
    Get a list of precision + runtime pairs for testing a model.

    Parameters
    ----------
    model_id
        model_id of the relevant model.
    model_supported_test_paths
        The list of (Precision, Runtime) pairs that are supported for testing for a the given model.
    model_passing_test_paths
        The list of (Precision, Runtime) pairs that have no known failure reasons for a the given model.
    path_type
        The type of scorecard path to return (Compile or Profile)
    can_use_quantize_job
        Whether this model can be quantized with QuantizeJob.
        If true, extra precisions set in parameter `enabled_test_precisions` will be included.
    include_unsupported_paths
        If true, all enabled paths will be included, instead of the ones compatible with
        parameter supported_paths.

    Returns
    -------
    enabled_test_paths : dict[Precision, list[ScorecardPathTypeVar]]
        A dict of (Precision, ScorecardPath) pairs to test.
        Each (Precision, ScorecardPath) pair will:
        * Only include items enabled in this environment via env variables
            (each arg is a comma separated list)
            - QAIHM_TEST_PRECISIONS (enabled precisions, default is DEFAULT (only include precisions supported by each model)
            - QAIHM_TEST_PATHS (enabled runtimes, default is ALL)
        * Be compatible with each other
        * Be compatible with the model
    """
    if include_unsupported_paths is None:
        include_unsupported_paths = IgnoreKnownFailuresEnvvar.get()

    # Get the precisions enabled for this model in this test environment.
    test_precisions = get_model_test_precisions(
        model_id,
        set(model_supported_test_paths.keys()),
        set(model_passing_test_paths.keys())
        if model_passing_test_paths is not None
        else None,
        can_use_quantize_job,
        include_unsupported_paths,
    )

    if include_unsupported_paths:
        model_test_precision_runtimes = model_supported_test_paths.copy()

        # Users can "force enable" a precision. In this case, models may run a precision that is not in their enabled list.
        # For these "forced" precisions, we test against all runtimes that are supported by this model.
        all_enabled_test_runtimes = {
            runtime
            for runtimes_by_precision in model_supported_test_paths.values()
            for runtime in runtimes_by_precision
        }
        for precision in set(test_precisions) - model_test_precision_runtimes.keys():
            if precision_runtimes := [
                x for x in all_enabled_test_runtimes if x.supports_precision(precision)
            ]:
                model_test_precision_runtimes[precision] = precision_runtimes
    else:
        model_test_precision_runtimes = model_passing_test_paths

    out: dict[Precision, list[ScorecardPathTypeVar]] = {}
    for precision in test_precisions:
        sc_paths: list[ScorecardPathTypeVar] = []
        for path in path_type:
            path = cast(ScorecardPathTypeVar, path)
            if path.should_run_path_for_model(precision, model_test_precision_runtimes):
                sc_paths.append(path)
        if sc_paths:
            out[precision] = sc_paths
    return out


def get_model_test_parameterizations(
    model_id: str,
    model_supported_test_paths: dict[Precision, list[TargetRuntime]],
    model_passing_test_paths: dict[Precision, list[TargetRuntime]],
    path_type: type[ScorecardPathTypeVar],
    can_use_quantize_job: bool = True,
    devices: list[ScorecardDevice] | None = None,
    include_unsupported_paths: bool | None = None,
    include_mirror_devices: bool = False,
) -> list[tuple[Precision, ScorecardPathTypeVar, ScorecardDevice]]:
    """
    Get a list of parameterizations for testing a model.

    Parameters
    ----------
    model_id
        model_id of the relevant model.
    model_supported_test_paths
        The list of (Precision, Runtime) pairs that are enabled for testing for a the given model.
    model_passing_test_paths
        The list of (Precision, Runtime) pairs that have no known failure reasons for a the given model.
    path_type
        The type of scorecard path to return (Compile or Profile)
    can_use_quantize_job
        Whether this model can be quantized with QuantizeJob.
        If true, extra precisions set in parameter `enabled_test_precisions` will be included.
    devices
        The list of devices to include. If None, all enabled devices are included.
    include_unsupported_paths
        If true, all enabled paths will be included, instead of the ones compatible with
        parameter supported_paths.
    include_mirror_devices
        If true, mirror devices will be included in the output.
        Jobs are never run on "mirror" devices. Instead, results are copied ("mirrored") from a different device.

    Returns
    -------
    enabled_test_paths : list[tuple[Precision, ScorecardPathTypeVar, ScorecardDevice]]
        A list of (Precision, ScorecardPath, Device) pairs to test.
        Each (Precision, ScorecardPath, Device) pair will:
        * Only include items enabled in this environment via env variables
            (each arg is a comma separated list)
            - QAIHM_TEST_PRECISIONS (enabled precisions, default is DEFAULT (only include precisions supported by each model)
            - QAIHM_TEST_PATHS (enabled runtimes, default is ALL)
            - QAIHM_TEST_DEVICES (enabled devices, default is ALL)
        * Be compatible with each other:
            - The ScorecardPath will be compatible with the Precision.
            - The ScorecardPath will be applicable to the Device.
            - The Precision can run on the Device's NPU.
        * Be compatible with the model:
            - See parameter documentation for details.
    """
    enabled_test_paths = get_enabled_paths_for_testing(
        model_id,
        model_supported_test_paths,
        model_passing_test_paths,
        path_type,
        can_use_quantize_job,
        include_unsupported_paths,
    )

    # Calculate the tests to run based on enabled paths, devices, and precisions
    ret: list[tuple[Precision, ScorecardPathTypeVar, ScorecardDevice]] = []
    for precision, sc_paths in enabled_test_paths.items():
        for sc_path in sc_paths:
            for device in (
                devices if devices is not None else ScorecardDevice.all_devices()
            ):
                if (
                    not device.enabled
                    or not device.npu_supports_precision(precision)
                    or (not include_mirror_devices and device.mirror_device is not None)
                ):
                    continue
                if (
                    isinstance(sc_path, ScorecardCompilePath)
                    and sc_path not in device.compile_paths
                ):
                    continue
                if (
                    isinstance(sc_path, ScorecardProfilePath)
                    and sc_path not in device.profile_paths
                ):
                    continue
                ret.append((precision, sc_path, device))
    return ret


def pytest_device_idfn(val: object) -> str | None:
    """
    Pytest generates test titles based on the parameterization of each test.
    This title can both be used as a filter during test selection and is
    printed to console to identify the test. An example title:
    qai_hub_models/models/whisper_base/test_generated.py::test_compile[qnn-cs_8_gen_3]

    Several unit tests parameterize based on device objects. Pytest is not capable by default
    of understanding what string identifier to use for a device object, so it will print
    `device##` in the title of those tests rather than the actual device name.

    Passing this function to the @pytest.mark.parametrize hook (ids=pytest_device_idfn) will
    instruct pytest to print the name of the device in the test title instead.

    See https://docs.pytest.org/en/stable/example/parametrize.html#different-options-for-test-ids
    """
    if isinstance(val, ScorecardDevice):
        return val.name
    if isinstance(val, Precision):
        return str(val)
    return None


def needs_pre_quantize_compile(
    model_id: str,
    enabled_test_paths: dict[Precision, list[TargetRuntime]],
    passing_test_paths: dict[Precision, list[TargetRuntime]],
) -> bool:
    return (
        len(
            get_quantize_parameterized_pytest_config(
                model_id, enabled_test_paths, passing_test_paths
            )
        )
        > 0
    )


def get_quantize_parameterized_pytest_config(
    model_id: str,
    enabled_test_paths: dict[Precision, list[TargetRuntime]],
    passing_test_paths: dict[Precision, list[TargetRuntime]],
) -> list[Precision]:
    precisions = get_model_test_precisions(
        model_id,
        set(enabled_test_paths.keys()),
        set(passing_test_paths.keys()),
        can_use_quantize_job=True,
    )
    return [x for x in precisions if x.has_quantized_activations]


def get_compile_parameterized_pytest_config(
    model_id: str,
    enabled_test_paths: dict[Precision, list[TargetRuntime]],
    passing_test_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
    include_mirror_devices: bool = False,
) -> list[tuple[Precision, ScorecardCompilePath, ScorecardDevice]]:
    """Get a pytest parameterization list of all enabled (device, compile path) pairs."""
    return get_model_test_parameterizations(
        model_id,
        enabled_test_paths,
        passing_test_paths,
        ScorecardCompilePath,
        can_use_quantize_job,
        include_mirror_devices=include_mirror_devices,
    )


def get_link_parameterized_pytest_config(
    model_id: str,
    enabled_test_paths: dict[Precision, list[TargetRuntime]],
    passing_test_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
    include_mirror_devices: bool = False,
) -> list[tuple[Precision, ScorecardCompilePath, ScorecardDevice]]:
    """
    Get a pytest parameterization list of all enabled (precision, compile path, device)
    tuples that require link jobs.

    Link jobs are needed for runtimes that use hub.link() to convert DLCs to
    device-specific context binaries.
    """
    compile_configs = get_model_test_parameterizations(
        model_id,
        enabled_test_paths,
        passing_test_paths,
        ScorecardCompilePath,
        can_use_quantize_job,
        include_mirror_devices=include_mirror_devices,
    )
    # Filter to only runtimes that use hub.link()
    return [
        (precision, path, device)
        for precision, path, device in compile_configs
        if path.runtime.uses_hub_link
    ]


def get_profile_parameterized_pytest_config(
    model_id: str,
    enabled_test_paths: dict[Precision, list[TargetRuntime]],
    passing_test_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
    include_mirror_devices: bool = False,
) -> list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """Get a pytest parameterization list of all enabled (device, profile path) pairs."""
    return get_model_test_parameterizations(
        model_id,
        enabled_test_paths,
        passing_test_paths,
        ScorecardProfilePath,
        can_use_quantize_job,
        include_mirror_devices=include_mirror_devices,
    )


def get_export_parameterized_pytest_config(
    model_id: str,
    device: ScorecardDevice,
    enabled_test_paths: dict[Precision, list[TargetRuntime]],
    passing_test_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
    requires_aot_prepare: bool = False,
) -> list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """Get a pytest parameterization list of all enabled (device, profile path) pairs."""
    return get_model_test_parameterizations(
        model_id,
        enabled_test_paths,
        passing_test_paths,
        ScorecardProfilePath,
        can_use_quantize_job,
        ScorecardDevice.all_devices(enabled=True, include_universal=False)
        if requires_aot_prepare
        else [device],
    )


def get_evaluation_parameterized_pytest_config(
    model_id: str,
    device: ScorecardDevice,
    enabled_test_paths: dict[Precision, list[TargetRuntime]],
    passing_test_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
) -> list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """Get a pytest parameterization list of all enabled (device, profile path) pairs."""
    enabled_devices = ScorecardDevice.all_devices(enabled=True, include_universal=False)
    if device not in enabled_devices:
        if len(enabled_devices) == 1:
            device = enabled_devices[0]
        else:
            # "When running numerical evaluation, must specify exactly one device or have {device} as part of the device list."
            return []

    return get_model_test_parameterizations(
        model_id,
        enabled_test_paths,
        passing_test_paths,
        ScorecardProfilePath,
        can_use_quantize_job,
        [device],
    )


def get_async_job_cache_name(
    path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None,
    model_id: str,
    device: ScorecardDevice,
    precision: Precision = Precision.float,
    component: str | None = None,
    graph_name: str | None = None,
) -> str:
    """
    Get the key for this job in the YAML that stores asyncronously-ran scorecard jobs.

    Parameters
    ----------
    path
        Applicable scorecard path
    model_id
        The ID of the QAIHM model being tested
    device
        The targeted device
    precision
        The precision in which this model is running
    component
        The name of the model component being tested, if applicable
    graph_name
        The name of the graph being executed (for multi-graph models)

    Returns
    -------
    cache_key : str
        The cache key for this job.
    """
    return (
        f"{model_id}"
        + ("_" + str(precision) if precision != Precision.float else "")
        + ("_" + path.name if path else "")
        + ("-" + device.name if device != cs_universal else "")
        + ("_" + component if component else "")
        + ("_" + graph_name if graph_name else "")
    )


def _on_staging() -> bool:
    """
    Returns whether the workbench client is pointing to staging.
    Can be sometimes useful to diverge logic between PR CI (prod) and nightly (staging).
    """
    client = hub.client.Client()
    client.get_devices()
    client_config = client._config
    assert client_config is not None
    return "staging" in client_config.api_url


@dataclass
class ClientState:
    on_staging: bool


class ClientStateSingleton:
    _instance: ClientState | None = None

    def __init__(self) -> None:
        if self._instance is None:
            self._instance = ClientState(on_staging=_on_staging())

    def on_staging(self) -> bool:
        assert self._instance is not None
        return self._instance.on_staging

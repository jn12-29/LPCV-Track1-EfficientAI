# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import copy

from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.model_disable_reasons import (
    ModelDisableReasons,
    ModelDisableReasonsMapping,
)
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.device import cs_x_elite
from qai_hub_models.scorecard.execution_helpers import (
    get_model_test_precisions,
)
from qai_hub_models.scorecard.results.chipset_helpers import (
    get_supported_devices,
    sorted_chipsets,
)
from qai_hub_models.scorecard.results.numerics_diff import (
    NumericsDiff,
)
from qai_hub_models.scorecard.results.performance_summary import (
    CompileScorecardJob,
    ModelCompileSummary,
    ModelPerfSummary,
    ProfileScorecardJob,
)
from qai_hub_models.utils.numerics_yaml import QAIHMModelNumerics
from qai_hub_models.utils.testing_export_eval import QAIHMModelReleaseAssets

# Maximum acceptable inference time (milliseconds).
# Above this inference time, a model will not be published.
MAX_ACCEPTABLE_INFERENCE_TIME_MS = 4000


def _clean_old_failure_reasons(
    precisions: list[Precision],
    code_gen_config: QAIHMModelCodeGen,
    clean_general: bool,
    clean_accuracy: bool,
) -> None:
    """In the code gen config, delete failure reasons for all enabled runtimes + given precision pairs."""
    passing_precisions: list[Precision] = []
    for precision, reasons_by_runtime in code_gen_config.disabled_paths.data.items():
        if precision in precisions:
            for path in ScorecardProfilePath:
                if (not path.is_published or path.enabled) and (
                    reasons := reasons_by_runtime.get(path.runtime)
                ):
                    if clean_general:
                        reasons.scorecard_failure = None
                    if clean_accuracy:
                        reasons.scorecard_accuracy_failure = None
                    if not reasons.has_failure:
                        reasons_by_runtime.pop(path.runtime)

        # Delete precisions that are no longer valid
        if precision not in code_gen_config.supported_precisions:
            for runtime, reasons in reasons_by_runtime.items():
                if clean_general:
                    reasons.scorecard_failure = None
                if clean_accuracy:
                    reasons.scorecard_accuracy_failure = None
                if not reasons.has_failure:
                    reasons_by_runtime.pop(runtime)

        if not reasons_by_runtime:
            passing_precisions.append(precision)

    for precision in passing_precisions:
        code_gen_config.disabled_paths.data.pop(precision)


def update_code_gen_failure_reasons(
    enabled_test_paths: dict[Precision, list[ScorecardProfilePath]],
    components: list[str] | None,
    compile_summary: ModelCompileSummary,
    link_summary: ModelCompileSummary,
    profile_summary: ModelPerfSummary,
    code_gen_config: QAIHMModelCodeGen,
) -> None:
    """
    Updates the provided model_info.code_gen_config to reflect job failures in the provided summaries.
    <path>_scorecard_failure will be set if certain jobs fail, and will be unset if no failing jobs are found.

    If relevant jobs can't be found in the given scorecard summaries, then no changes are made to the config.
    """
    model_id = compile_summary.model_id

    # Limit to only public paths. If a path is not public, then we don't track it in code-gen.yaml.
    enabled_test_paths = {
        p: [path for path in paths if path.is_published]
        for p, paths in enabled_test_paths.items()
    }

    compile_failures: dict[
        Precision, dict[ScorecardCompilePath, dict[str, CompileScorecardJob]]
    ] = {
        p: {path.compile_path: {} for path in paths}
        for p, paths in enabled_test_paths.items()
    }
    link_failures: dict[
        Precision, dict[ScorecardCompilePath, dict[str, CompileScorecardJob]]
    ] = {
        p: {path.compile_path: {} for path in paths}
        for p, paths in enabled_test_paths.items()
    }
    profile_failures: dict[
        Precision, dict[ScorecardProfilePath, dict[str, ProfileScorecardJob]]
    ] = {p: {path: {} for path in paths} for p, paths in enabled_test_paths.items()}
    too_slow_profile_jobs: dict[
        Precision, dict[ScorecardProfilePath, dict[str, ProfileScorecardJob]]
    ] = copy.deepcopy(profile_failures)

    has_profile_jobs: dict[Precision, dict[ScorecardProfilePath, bool]] = {
        p: {} for p in enabled_test_paths
    }

    default_device = ScorecardDevice.get(device_name=code_gen_config.default_device)
    canary_devices = ScorecardDevice.canary_devices()

    def process_model(
        precision: Precision, path: ScorecardProfilePath, device: ScorecardDevice
    ) -> None:
        for component_id in components or [model_id]:
            # Skip model if it can't compile for any canary device.
            compile_job = compile_summary.get_run(
                precision, device, path.compile_path, component_id
            )
            if compile_job.failed:
                compile_failures[precision][path.compile_path][component_id] = (
                    compile_job
                )

            # Skip model if it can't link for any canary device (AOT runtimes only).
            link_job = link_summary.get_run(
                precision, device, path.compile_path, component_id
            )
            if link_job.failed:
                link_failures[precision][path.compile_path][component_id] = link_job

            if device == default_device:
                # Skip model if it can't be profiled on its default device.
                profile_job = profile_summary.get_run(
                    precision, device, path, component_id
                )
                if (
                    profile_job.status_message is not None
                    and "memory usage exceeded" in profile_job.status_message
                ):
                    # X Elite has the most memory by far. If X Elite can run it, then we support it,
                    x_elite_profile_job = profile_summary.get_run(
                        precision, cs_x_elite, path, component_id
                    )
                    if x_elite_profile_job.success:
                        profile_job = x_elite_profile_job

                if profile_job.failed:
                    profile_failures[precision][path][component_id] = profile_job
                if not profile_job.skipped:
                    has_profile_jobs[precision][path] = True
                if profile_job.success and (
                    profile_job.inference_time_milliseconds
                    >= MAX_ACCEPTABLE_INFERENCE_TIME_MS
                ):
                    too_slow_profile_jobs[precision][path][component_id] = profile_job

    for precision, sc_paths in enabled_test_paths.items():
        for path in sc_paths:
            for device in {default_device, *canary_devices}:
                process_model(precision, path, device)

    supported_precisions = list(enabled_test_paths.keys())
    _clean_old_failure_reasons(
        precisions=supported_precisions,
        code_gen_config=code_gen_config,
        clean_general=True,
        clean_accuracy=False,
    )

    # Add new failure reasons
    for precision, sc_paths in enabled_test_paths.items():
        for path in sc_paths:
            if not path.supports_precision(precision):
                path_failure_reason = f"{path.runtime} does not support {precision!s}"
            elif failed_compile_jobs := compile_failures[precision][path.compile_path]:
                failures = [
                    f"{key}:{val.job_id}" if key != model_id else str(val.job_id)
                    for key, val in failed_compile_jobs.items()
                ]
                path_failure_reason = f"Compilation Failure(s): {' '.join(failures)}"
            elif failed_link_jobs := link_failures[precision][path.compile_path]:
                failures = [
                    f"{key}:{val.job_id}" if key != model_id else str(val.job_id)
                    for key, val in failed_link_jobs.items()
                ]
                path_failure_reason = f"Link Failure(s): {' '.join(failures)}"
            elif failed_profile_jobs := profile_failures[precision][path]:
                failures = [
                    f"{key}:{val.job_id}" if key != model_id else str(val.job_id)
                    for key, val in failed_profile_jobs.items()
                ]
                path_failure_reason = f"Profiling Failure(s): {' '.join(failures)}"
            elif slow_profile_jobs := too_slow_profile_jobs[precision][path]:
                failures = [
                    f"{key}:{val.job_id}" if key != model_id else str(val.job_id)
                    for key, val in slow_profile_jobs.items()
                ]
                path_failure_reason = f"Profiling jobs slower than {MAX_ACCEPTABLE_INFERENCE_TIME_MS}ms: {' '.join(failures)}"
            elif not has_profile_jobs[precision].get(path, False):
                path_failure_reason = (
                    f"No profile jobs found with default device {default_device}"
                )
            else:
                path_failure_reason = ""

            if path_failure_reason:
                code_gen_config.disabled_paths.get_disable_reasons(
                    precision, path.runtime
                ).scorecard_failure = path_failure_reason


def update_code_gen_accuracy_failure_reasons(
    model_id: str, code_gen_config: QAIHMModelCodeGen, model_diff: NumericsDiff
) -> None:
    supported_precisions = get_model_test_precisions(
        model_id,
        set(code_gen_config.supported_precisions),
        can_use_quantize_job=code_gen_config.can_use_quantize_job,
    )

    _clean_old_failure_reasons(
        precisions=supported_precisions,
        code_gen_config=code_gen_config,
        clean_general=False,
        clean_accuracy=True,
    )

    for disabled_path in model_diff.device_vs_float_greater_than_enablement_threshold:
        diff_model_id = disabled_path[0]
        precision = disabled_path[4]
        path = disabled_path[5]
        if (
            diff_model_id != model_id
            or not path.is_published
            or not path.enabled
            or precision not in supported_precisions
        ):
            continue

        if precision not in code_gen_config.disabled_paths.data:
            code_gen_config.disabled_paths.data[precision] = {}
        reasons_by_runtime = code_gen_config.disabled_paths.data[precision]
        if path.runtime not in reasons_by_runtime:
            reasons_by_runtime[path.runtime] = ModelDisableReasons()
        reasons = reasons_by_runtime[path.runtime]
        reasons.scorecard_accuracy_failure = f"Torch and On-device accuracy diff ({disabled_path[8]}) above threshold ({disabled_path[9]})"

    for benchmark_failure in model_diff.benchmark_failures:
        diff_model_id = benchmark_failure[0]
        if diff_model_id != model_id:
            continue

        accuracy_type = benchmark_failure[3]
        bf_precision = benchmark_failure[4]
        bf_path = benchmark_failure[5]

        if accuracy_type == "device":
            assert bf_precision is not None and bf_path is not None
            if (
                not bf_path.is_published
                or not bf_path.enabled
                or bf_precision not in supported_precisions
            ):
                continue
            if bf_precision not in code_gen_config.disabled_paths.data:
                code_gen_config.disabled_paths.data[bf_precision] = {}
            reasons_by_runtime = code_gen_config.disabled_paths.data[bf_precision]
            if bf_path.runtime not in reasons_by_runtime:
                reasons_by_runtime[bf_path.runtime] = ModelDisableReasons()
            reasons = reasons_by_runtime[bf_path.runtime]
            reasons.scorecard_accuracy_failure = f"Device accuracy ({benchmark_failure[6]}) deviates from benchmark ({benchmark_failure[7]}) by {benchmark_failure[8]}, threshold: {benchmark_failure[9]}"

        elif accuracy_type == "torch":
            # Torch accuracy failure affects all paths for all precisions
            for p in supported_precisions:
                for sc_path in ScorecardProfilePath:
                    if not sc_path.is_published or not sc_path.enabled:
                        continue
                    if p not in code_gen_config.disabled_paths.data:
                        code_gen_config.disabled_paths.data[p] = {}
                    reasons_by_runtime = code_gen_config.disabled_paths.data[p]
                    if sc_path.runtime not in reasons_by_runtime:
                        reasons_by_runtime[sc_path.runtime] = ModelDisableReasons()
                    reasons = reasons_by_runtime[sc_path.runtime]
                    reasons.scorecard_accuracy_failure = f"Torch accuracy ({benchmark_failure[6]}) deviates from benchmark ({benchmark_failure[7]}) by {benchmark_failure[8]}, threshold: {benchmark_failure[9]}"


def update_model_publish_status(model_info: QAIHMModelInfo) -> bool:
    """Update the model publishing status based on failure reasons. Returns true if the status was changed, false otherwise."""
    cg = model_info.code_gen_config

    # Update model status & reason, if applicable
    if model_info.status == MODEL_STATUS.PENDING:
        can_promote, reason = model_info.can_promote_to_published()
        if can_promote:
            model_info.status = MODEL_STATUS.PUBLISHED
            model_info.status_reason = None
            print(f"{model_info.id} | Set model to PUBLISHED")
            return True
        if cg.supports_at_least_1_runtime:
            # Model has passing runtimes but is missing other requirements
            print(f"{model_info.id} | Skipping promotion to PUBLISHED: {reason}")
    elif (
        not cg.supports_at_least_1_runtime
        and model_info.status == MODEL_STATUS.PUBLISHED
    ):
        # Demote PUBLISHED to PENDING if no longer eligible
        model_info.status = MODEL_STATUS.PENDING
        model_info.status_reason = "No successful runtimes in scorecard (this field was auto-populated by the scorecard run)"
        print(f"{model_info.id} | Set model to PENDING (no successful runtimes)")
        return True
    # Note: PENDING stays PENDING if no successful runtimes yet

    return False


def remove_numerics_failures(
    numerics: QAIHMModelNumerics, failure_reasons: ModelDisableReasonsMapping
) -> QAIHMModelNumerics:
    """
    Drop all device + runtime + precision pairs from the numerics YAML for which a failure reason exists.

    Parameters
    ----------
    numerics
        The numerics YAML to modify.

    failure_reasons
        The failure reasons to consider.

    Returns
    -------
    QAIHMModelNumerics
        New numerics.yaml with failing device + runtime + precisions pairs removed.
    """
    metrics: list[QAIHMModelNumerics.MetricDetails] = []
    for metric in numerics.metrics:
        new_metrics_by_device: dict[
            ScorecardDevice,
            dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ],
        ] = {}

        for device, metrics_by_precision in metric.device_metric.items():
            new_metrics_by_precision: dict[
                Precision, dict[ScorecardProfilePath, QAIHMModelNumerics.DeviceDetails]
            ] = {}

            for precision, metrics_by_path in metrics_by_precision.items():
                runtimes_with_failures = [
                    x
                    for x, y in (failure_reasons.data.get(precision) or {}).items()
                    if y.has_failure
                ]
                new_metrics_by_path = {
                    path: copy.deepcopy(metrics_by_path[path])
                    for path in metrics_by_path
                    if path.runtime not in runtimes_with_failures
                }
                if new_metrics_by_path:
                    new_metrics_by_precision[precision] = new_metrics_by_path

            if new_metrics_by_precision:
                new_metrics_by_device[device] = new_metrics_by_precision

        if new_metrics_by_device:
            metrics.append(
                QAIHMModelNumerics.MetricDetails(
                    dataset_name=metric.dataset_name,
                    dataset_link=metric.dataset_link,
                    dataset_split_description=metric.dataset_split_description,
                    metric_name=metric.metric_name,
                    metric_description=metric.metric_description,
                    metric_unit=metric.metric_unit,
                    metric_range=copy.copy(metric.metric_range),
                    metric_enablement_threshold=metric.metric_enablement_threshold,
                    benchmark_value=metric.benchmark_value,
                    num_partial_samples=metric.num_partial_samples,
                    partial_torch_metric=metric.partial_torch_metric,
                    device_metric=new_metrics_by_device,
                )
            )

    return QAIHMModelNumerics(metrics=metrics)


def remove_perf_failures(
    perf: QAIHMModelPerf, failure_reason: ModelDisableReasonsMapping
) -> QAIHMModelPerf:
    """
    Drop all device + runtime + precision pairs from the perf YAML for which a failure reason exists.

    Parameters
    ----------
    perf
        The perf YAML to modify.
    failure_reason
        The failure reasons to consider.

    Returns
    -------
    QAIHMModelPerf
        New perf.yaml with failing device + runtime + precisions pairs removed.
    """
    supported_chipsets: set[str] = set()
    precisions: dict[Precision, QAIHMModelPerf.PrecisionDetails] = {}

    for precision, perf_precision_components in perf.precisions.items():
        runtimes_with_failures = [
            x
            for x, y in (failure_reason.data.get(precision) or {}).items()
            if y.has_failure
        ]
        components: dict[str, QAIHMModelPerf.ComponentDetails] = {}
        for (
            component,
            component_details,
        ) in perf_precision_components.components.items():
            device_assets: dict[
                ScorecardDevice, dict[ScorecardProfilePath, QAIHMModelPerf.AssetDetails]
            ] = {}
            for device, asset_by_path in component_details.device_assets.items():
                path_assets = {
                    path: copy.deepcopy(asset_by_path[path])
                    for path in asset_by_path
                    if path.runtime not in runtimes_with_failures
                }
                if path_assets:
                    device_assets[device] = path_assets

            performance_metrics: dict[
                ScorecardDevice,
                dict[ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails],
            ] = {}
            for device, perf_by_path in component_details.performance_metrics.items():
                path_perf = {
                    path: copy.deepcopy(perf_by_path[path])
                    for path in perf_by_path
                    if path.runtime not in runtimes_with_failures
                }
                if path_perf:
                    performance_metrics[device] = path_perf
                    supported_chipsets.update(device.extended_supported_chipsets)

            universal_assets = {
                path: copy.deepcopy(component_details.universal_assets[path])
                for path in component_details.universal_assets
                if path.runtime not in runtimes_with_failures
            }
            if device_assets or performance_metrics or universal_assets:
                components[component] = QAIHMModelPerf.ComponentDetails(
                    universal_assets=universal_assets,
                    device_assets=device_assets,
                    performance_metrics=performance_metrics,
                )

        if components:
            precisions[precision] = QAIHMModelPerf.PrecisionDetails(
                components=components
            )

    return QAIHMModelPerf(
        supported_devices=get_supported_devices(supported_chipsets),
        supported_chipsets=sorted_chipsets(supported_chipsets),
        precisions=precisions,
    )


def remove_asset_failures(
    assets: QAIHMModelReleaseAssets, failure_reasons: ModelDisableReasonsMapping
) -> QAIHMModelReleaseAssets:
    """
    Drop all device + runtime + precision pairs from the assets YAML for which a failure reason exists.

    Parameters
    ----------
    assets
        The assets YAML to modify.

    failure_reasons
        The failure reasons to consider.

    Returns
    -------
    QAIHMModelReleaseAssets
        New pre_release_assets.yaml with failing device + runtime + precisions pairs removed.
    """
    precisions: dict[Precision, QAIHMModelReleaseAssets.PrecisionDetails] = {}

    for precision, precision_details in assets.precisions.items():
        runtimes_with_failures = [
            x
            for x, y in (failure_reasons.data.get(precision) or {}).items()
            if y.has_failure
        ]

        chipset_assets: dict[
            str,
            dict[ScorecardProfilePath, QAIHMModelReleaseAssets.AssetDetails],
        ] = {}
        for chipset, asset_by_path in precision_details.chipset_assets.items():
            path_assets = {
                path: copy.deepcopy(asset_by_path[path])
                for path in asset_by_path
                if path.runtime not in runtimes_with_failures
            }
            if path_assets:
                chipset_assets[chipset] = path_assets

        universal_assets = {
            path: copy.deepcopy(precision_details.universal_assets[path])
            for path in precision_details.universal_assets
            if path.runtime not in runtimes_with_failures
        }
        if chipset_assets or universal_assets:
            precisions[precision] = QAIHMModelReleaseAssets.PrecisionDetails(
                universal_assets=universal_assets,
                chipset_assets=chipset_assets,
            )

    return QAIHMModelReleaseAssets(precisions=precisions)

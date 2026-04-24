# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.device import ScorecardDevice, cs_8_gen_3
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.performance_diff import PerformanceDiff

MODEL_ID = "dummy"
PREV_JOB_ID = "jgzr270o5"
JOB_ID = "jp4kr0kvg"
COMPONENT_ID = "dummy_component"


def get_basic_speedup_report(
    job_id: str = JOB_ID,
    onnx_tf_inference_time: float | None = None,
    onnx_ort_qnn_inference_time: float | None = 100.0,
) -> QAIHMModelPerf:
    return QAIHMModelPerf(
        precisions={
            Precision.float: QAIHMModelPerf.PrecisionDetails(
                components={
                    COMPONENT_ID: QAIHMModelPerf.ComponentDetails(
                        performance_metrics={
                            cs_8_gen_3: {
                                ScorecardProfilePath.TFLITE: QAIHMModelPerf.PerformanceDetails(
                                    job_id=job_id,
                                    inference_time_milliseconds=onnx_tf_inference_time,
                                ),
                                ScorecardProfilePath.ONNX: QAIHMModelPerf.PerformanceDetails(
                                    job_id=job_id, inference_time_milliseconds=5.0
                                ),
                                ScorecardProfilePath.QNN_DLC: QAIHMModelPerf.PerformanceDetails(
                                    job_id=job_id,
                                    inference_time_milliseconds=onnx_ort_qnn_inference_time,
                                ),
                            }
                        }
                    )
                }
            )
        }
    )


def validate_perf_diff_is_empty(perf_diff: PerformanceDiff) -> None:
    # No difference captured
    for val in perf_diff.progressions.values():
        assert len(val) == 0
    for val in perf_diff.regressions.values():
        assert len(val) == 0
    # No new reports captured
    assert len(perf_diff.new_models) == 0
    # No missing devices found in updated report
    assert len(perf_diff.missing_devices) == 0


def test_model_inference_run_toggle() -> None:
    # Test model inference fail/pass toggle is captured
    prev_perf_metrics = get_basic_speedup_report(
        PREV_JOB_ID, onnx_tf_inference_time=None, onnx_ort_qnn_inference_time=10.0
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=10.0, onnx_ort_qnn_inference_time=None
    )

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert perf_diff.progressions[float("inf")] == [
        (
            MODEL_ID,
            Precision.float,
            COMPONENT_ID,
            cs_8_gen_3,
            ScorecardProfilePath.TFLITE,
            float("-inf"),
            10.0,
            float("inf"),
            JOB_ID,
            PREV_JOB_ID,
        )
    ]


def test_perf_progression_basic() -> None:
    prev_perf_metrics = get_basic_speedup_report(
        PREV_JOB_ID, onnx_tf_inference_time=10.0, onnx_ort_qnn_inference_time=5.123
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=0.5, onnx_ort_qnn_inference_time=5.123
    )

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert perf_diff.progressions[10] == [
        (
            MODEL_ID,
            Precision.float,
            COMPONENT_ID,
            cs_8_gen_3,
            ScorecardProfilePath.TFLITE,
            10.0,
            0.5,
            20.0,
            JOB_ID,
            PREV_JOB_ID,
        )
    ]


def test_perf_regression_basic() -> None:
    # Test regression in perf numbers
    prev_perf_metrics = get_basic_speedup_report(
        PREV_JOB_ID, onnx_tf_inference_time=10.0, onnx_ort_qnn_inference_time=5.123
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=20.0, onnx_ort_qnn_inference_time=5.123
    )

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert perf_diff.regressions[2] == [
        (
            MODEL_ID,
            Precision.float,
            COMPONENT_ID,
            cs_8_gen_3,
            ScorecardProfilePath.TFLITE,
            10.0,
            20.0,
            2.0,
            JOB_ID,
            PREV_JOB_ID,
        ),
    ]


def test_missing_devices() -> None:
    prev_perf_metrics = get_basic_speedup_report(
        PREV_JOB_ID, onnx_tf_inference_time=1.123, onnx_ort_qnn_inference_time=5.123
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=0.372, onnx_ort_qnn_inference_time=5.123
    )

    # Override chipset
    new_perf_metrics.precisions[Precision.float].components[
        COMPONENT_ID
    ].performance_metrics.pop(cs_8_gen_3)
    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert len(perf_diff.missing_devices) == 1
    assert perf_diff.missing_devices[0] == (
        MODEL_ID,
        Precision.float,
        COMPONENT_ID,
        cs_8_gen_3,
    )


def test_empty_report() -> None:
    prev_perf_metrics = get_basic_speedup_report()
    new_perf_metrics = get_basic_speedup_report()
    new_perf_metrics.precisions.pop(Precision.float)

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)
    assert perf_diff.empty_models == [MODEL_ID]


def test_auto_device_excluded() -> None:
    """Automotive device regressions are excluded from the diff."""
    # Use an auto device reference name from devices_and_chipsets.yaml
    auto_device = ScorecardDevice(
        name="test_auto",
        reference_device_name="SA8295P ADP",
        register=False,
    )

    # Build reports with a clear 2x regression on the auto device
    auto_report = QAIHMModelPerf(
        precisions={
            Precision.float: QAIHMModelPerf.PrecisionDetails(
                components={
                    COMPONENT_ID: QAIHMModelPerf.ComponentDetails(
                        performance_metrics={
                            auto_device: {
                                ScorecardProfilePath.TFLITE: QAIHMModelPerf.PerformanceDetails(
                                    job_id=JOB_ID,
                                    inference_time_milliseconds=10.0,
                                ),
                            }
                        }
                    )
                }
            )
        }
    )
    auto_report_regressed = QAIHMModelPerf(
        precisions={
            Precision.float: QAIHMModelPerf.PrecisionDetails(
                components={
                    COMPONENT_ID: QAIHMModelPerf.ComponentDetails(
                        performance_metrics={
                            auto_device: {
                                ScorecardProfilePath.TFLITE: QAIHMModelPerf.PerformanceDetails(
                                    job_id=JOB_ID,
                                    inference_time_milliseconds=20.0,
                                ),
                            }
                        }
                    )
                }
            )
        }
    )

    perf_diff = PerformanceDiff()
    perf_diff.update_summary(MODEL_ID, auto_report, auto_report_regressed)

    # 2x regression should be excluded because device is automotive
    for val in perf_diff.regressions.values():
        assert len(val) == 0


def test_small_regression_excluded() -> None:
    """Regressions with absolute diff <=1ms are excluded (within noise margin)."""
    # 5.0 -> 5.9ms = 0.9ms diff, 1.18x ratio (would normally be in 1.1 bucket)
    prev = get_basic_speedup_report(
        PREV_JOB_ID,
        onnx_tf_inference_time=5.0,
        onnx_ort_qnn_inference_time=5.0,
    )
    new = get_basic_speedup_report(
        onnx_tf_inference_time=5.9,
        onnx_ort_qnn_inference_time=5.0,
    )

    perf_diff = PerformanceDiff()
    perf_diff.update_summary(MODEL_ID, prev, new)

    # 0.9ms regression should be excluded even though ratio is 1.18x
    for val in perf_diff.regressions.values():
        assert len(val) == 0

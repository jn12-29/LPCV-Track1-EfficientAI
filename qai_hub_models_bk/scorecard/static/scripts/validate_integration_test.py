# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import functools
import os
import traceback
from copy import deepcopy

import pandas as pd
from junitparser import Error, Failure, JUnitXml, TestCase, TestSuite

from qai_hub_models.scorecard.device import DEFAULT_SCORECARD_DEVICE
from qai_hub_models.utils.testing_async_utils import (
    get_accuracy_csv_path,
    get_aggreggated_results_csv_path,
    get_scorecard_csv_path,
)

EXPECTED_MODEL_SETS = {
    "yolov8_det": {
        "float": ["onnx", "qnn", "tflite"],
        "w8a8": ["onnx", "qnn", "tflite"],
        "w8a16": ["onnx", "qnn"],
        "w8a8_mixed_int16": ["onnx", "qnn"],
    },
    "mask2former": {
        "float": ["onnx", "qnn"],
    },
    "mediapipe_face": {
        "float": ["onnx", "qnn", "tflite"],
        "w8a8": ["onnx", "qnn", "tflite"],
    },
    "amt_torchscript": {"float": ["onnx", "qnn", "tflite"]},
    "efficientformer_onnx": {"float": ["qnn", "tflite"]},
}
SELECTED_DEVICES = {
    ("mask2former", DEFAULT_SCORECARD_DEVICE.chipset),
    ("mediapipe_face", DEFAULT_SCORECARD_DEVICE.chipset),
    ("mediapipe_face::face_landmark_detector", DEFAULT_SCORECARD_DEVICE.chipset),
    ("mediapipe_face::face_detector", DEFAULT_SCORECARD_DEVICE.chipset),
    ("yolov8_det", DEFAULT_SCORECARD_DEVICE.chipset),
    ("amt_torchscript", DEFAULT_SCORECARD_DEVICE.chipset),
    ("efficientformer_onnx", "qualcomm-sa8295p"),
}


@functools.lru_cache(maxsize=1)
def num_configurations() -> int:
    """Number of (runtime, precision, model) combinations"""
    return sum(
        [
            len(runtime_list)
            for _, model_dict in EXPECTED_MODEL_SETS.items()
            for _, runtime_list in model_dict.items()
        ]
    )


def validate_results_df(results_df: pd.DataFrame) -> list[str]:
    """
    Checks the aggregated results csv and verifies that it has the expected outputs.
    Returns a list of error strings, if any.
    """
    results_df = results_df[
        results_df[["model_id", "chipset"]].apply(tuple, axis=1).isin(SELECTED_DEVICES)
    ]
    unfound_model_sets = deepcopy(EXPECTED_MODEL_SETS)
    unfound_model_sets["mediapipe_face::face_landmark_detector"] = deepcopy(
        unfound_model_sets["mediapipe_face"]
    )
    unfound_model_sets["mediapipe_face::face_detector"] = deepcopy(
        unfound_model_sets["mediapipe_face"]
    )

    errors = []
    # Dummy row added for tflite in aggregated results for tableau
    unfound_model_sets["yolov8_det"]["w8a16"].append("tflite")
    unfound_model_sets["yolov8_det"]["w8a8_mixed_int16"].append("tflite")
    unfound_model_sets["mask2former"]["float"].append("tflite")
    for _, row in results_df.iterrows():
        runtime_list = unfound_model_sets.get(row.model_id, {}).get(row.precision, [])
        if row.runtime in runtime_list:
            unfound_model_sets[row.model_id][row.precision].remove(row.runtime)
            if len(unfound_model_sets[row.model_id][row.precision]) == 0:
                unfound_model_sets[row.model_id].pop(row.precision)
            if len(unfound_model_sets[row.model_id]) == 0:
                unfound_model_sets.pop(row.model_id)
        else:
            errors.append(
                f"Unexpected (or duplicate) model configuration in aggregated csv ({row.model_id}, {row.precision}, {row.runtime})."
            )
    if len(unfound_model_sets) > 0:
        errors.append(
            f"Missing some rows in aggregated results csv: {unfound_model_sets}"
        )
    return errors


def validate_scorecard_df(scorecard_df: pd.DataFrame) -> list[str]:
    """
    Checks the performance results csv and verifies that it has the expected outputs.
    Returns a list of error strings, if any.
    """
    scorecard_df = scorecard_df[
        scorecard_df[["model_id", "chipset"]]
        .apply(tuple, axis=1)
        .isin(SELECTED_DEVICES)
    ]
    unfound_model_sets = deepcopy(EXPECTED_MODEL_SETS)
    unfound_model_sets["mediapipe_face::face_landmark_detector"] = deepcopy(
        unfound_model_sets["mediapipe_face"]
    )
    unfound_model_sets["mediapipe_face::face_detector"] = deepcopy(
        unfound_model_sets["mediapipe_face"]
    )
    unfound_model_sets.pop("mediapipe_face")

    errors = []
    for _, row in scorecard_df.iterrows():
        runtime_list = unfound_model_sets.get(row.model_id, {}).get(row.precision, [])
        if row.runtime in runtime_list:
            unfound_model_sets[row.model_id][row.precision].remove(row.runtime)
            if len(unfound_model_sets[row.model_id][row.precision]) == 0:
                unfound_model_sets[row.model_id].pop(row.precision)
            if len(unfound_model_sets[row.model_id]) == 0:
                unfound_model_sets.pop(row.model_id)
        else:
            errors.append(
                f"Unexpected (or duplicate) model configuration in scorecard csv ({row.model_id}, {row.precision}, {row.runtime})."
            )
    if len(unfound_model_sets) > 0:
        errors.append(
            f"Missing some rows in scorecard results csv: {unfound_model_sets}"
        )
    return errors


def validate_accuracy_df(accuracy_df: pd.DataFrame) -> list[str]:
    """
    Checks the accuracy results csv and verifies that it has the expected outputs.
    Returns a list of error strings, if any.
    """
    expected_models = set(EXPECTED_MODEL_SETS.keys())
    if (accuracy_models := set(accuracy_df.model_id.unique())) != expected_models:
        return [
            f"Mismatch between accuracy csv models ({accuracy_models}) and expected models ({expected_models})."
        ]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate scorecard integration test outputs"
    )
    parser.add_argument("--junit-xml", help="Path to write JUnit XML results")
    args = parser.parse_args()

    def write_junit_xml(
        result: Failure | Error | None,
    ) -> None:
        if not args.junit_xml:
            return
        suite = TestSuite(name="Scorecard Integration Test")
        tc = TestCase(
            name="validate_scorecard_outputs", classname="scorecard_integration_test"
        )
        if result is not None:
            tc.result = [result]
        suite.add_testcase(tc)

        xml = JUnitXml()
        xml.add_testsuite(suite)
        os.makedirs(os.path.dirname(args.junit_xml) or ".", exist_ok=True)
        xml.write(args.junit_xml)

    try:
        scorecard_csv = get_scorecard_csv_path()
        results_csv = get_aggreggated_results_csv_path()
        accuracy_csv = get_accuracy_csv_path()
        assert accuracy_csv is not None
        scorecard_df = pd.read_csv(scorecard_csv)
        results_df = pd.read_csv(results_csv)
        accuracy_df = pd.read_csv(accuracy_csv)

        errors = []
        errors.extend(validate_results_df(results_df))
        errors.extend(validate_scorecard_df(scorecard_df))
        errors.extend(validate_accuracy_df(accuracy_df))

        if errors:
            raise ValueError(  # noqa: TRY301
                "The following errors occurred during validation:\n\n"
                + "\n\n".join(errors)
            )
    except Exception as e:
        err = Error(message=str(e), type_=type(e).__name__)
        err.text = traceback.format_exc()
        write_junit_xml(err)
        raise

    write_junit_xml(None)


if __name__ == "__main__":
    main()

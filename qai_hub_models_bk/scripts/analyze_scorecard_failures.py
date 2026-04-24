#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""
Script to analyze scorecard CSV files and generate a clear failure summary.

This script ingests the existing scorecard CSV and creates a single comprehensive
summary focused on failure reasons, sorted by frequency and impact.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def extract_failure_reason(status: str) -> str:
    """Extract the failure reason from a status string."""
    if pd.isna(status) or status == "":
        return "Unknown"

    if status.startswith("Passed"):
        return "Passed"
    if status.startswith("Skipped"):
        return "Skipped"

    # Extract failure reason from "Failed (reason)" format
    match = re.search(r"Failed \(([^)]+)\)", status)
    if match:
        return match.group(1).strip()

    if status.startswith("Failed"):
        return "Job failed"

    return status


def determine_failure_reason(row: pd.Series) -> tuple[str, str, str]:
    """
    Determine the primary failure reason and stage for a model run.
    Check stages in order: quantize -> compile -> profile -> inference

    Parameters
    ----------
    row
        A row from the scorecard DataFrame containing status and URL columns.

    Returns
    -------
    failure_reason : str
        The reason for the failure.
    failure_stage : str
        The stage at which the failure occurred.
    job_url : str
        The URL of the failed job.
    """
    stages = ["quantize_status", "compile_status", "profile_status", "inference_status"]
    stage_names = ["Quantize", "Compile", "Profile", "Inference"]
    url_columns = ["quantize_url", "compile_url", "profile_url", "inference_url"]

    for status_col, stage_name, url_col in zip(
        stages, stage_names, url_columns, strict=False
    ):
        if status_col in row.index:
            status = row[status_col]
            if pd.notna(status) and status.startswith("Failed"):
                failure_reason = extract_failure_reason(status)
                job_url = row.get(url_col, "")

                # If we just got "Job failed", try to make it more specific
                if failure_reason == "Job failed":
                    runtime = row.get("runtime", "")

                    # Add context based on stage and runtime
                    if stage_name == "Compile":
                        failure_reason = f"{runtime.upper()} compilation failed"
                    elif stage_name == "Profile":
                        failure_reason = f"{runtime.upper()} profiling failed"
                    elif stage_name == "Inference":
                        failure_reason = f"{runtime.upper()} inference failed"
                    elif stage_name == "Quantize":
                        failure_reason = "Quantization failed"

                return failure_reason, stage_name, job_url

    return "Passed", "None", ""


def create_failure_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Create a failure analysis focused on failure reasons and their impact."""
    # Define all column names in one place for easy maintenance
    COL_FAILURE_STAGE = "Failure_Stage"
    COL_FAILURE_REASON = "Failure_Reason"
    COL_FAILURE_COUNT = "Failure_Count"
    COL_DEVICE_COUNT = "Device_Count"
    COL_RUNTIME_COUNT = "Runtime_Count"
    COL_MODEL_COUNT = "Model_Count"
    COL_JOB_URL = "Job_URL"
    COL_AFFECTED_DEVICES = "Affected_Devices"
    COL_AFFECTED_RUNTIMES = "Affected_Runtimes"
    COL_MODELS = "Models"
    COL_MODEL_LIST = "Model_List"
    COL_DEVICE_LIST = "Device_List"
    COL_RUNTIME_LIST = "Runtime_List"
    COL_STAGE_ORDER = "Stage_Order"

    # Add failure reason, stage, and job URL columns
    failure_data = df.apply(determine_failure_reason, axis=1)
    df["failure_reason"] = failure_data.apply(lambda x: x[0])
    df["failure_stage"] = failure_data.apply(lambda x: x[1])
    df["job_url"] = failure_data.apply(lambda x: x[2])

    # Filter to only failed runs
    failed_df = df[df["failure_reason"] != "Passed"].copy()

    if failed_df.empty:
        print("No failures found in the data!")
        return pd.DataFrame(
            columns=[
                COL_FAILURE_STAGE,
                COL_FAILURE_REASON,
                COL_FAILURE_COUNT,
                COL_DEVICE_COUNT,
                COL_RUNTIME_COUNT,
                COL_MODEL_COUNT,
                COL_JOB_URL,
                COL_AFFECTED_DEVICES,
                COL_AFFECTED_RUNTIMES,
                COL_MODELS,
            ]
        )

    # For compile failures, deduplicate by model_id + runtime to avoid overcounting
    # (compile jobs are device-agnostic but appear once per device in the CSV)
    compile_mask = failed_df["failure_stage"] == "Compile"
    compile_failures = failed_df[compile_mask].drop_duplicates(
        subset=["model_id", "runtime", "failure_reason"]
    )
    other_failures = failed_df[~compile_mask]

    # Combine deduplicated compile failures with other failures
    deduplicated_df = pd.concat([compile_failures, other_failures], ignore_index=True)

    # Group by failure stage and reason, aggregate information
    failure_analysis = (
        deduplicated_df.groupby(["failure_stage", "failure_reason"])
        .agg(
            {
                "model_id": ["count", lambda x: list(x.unique())],
                "chipset": lambda x: sorted(x.unique()),
                "runtime": lambda x: sorted(x.unique()),
                "job_url": lambda x: next(
                    (url for url in x if url), ""
                ),  # Get first non-empty URL
            }
        )
        .reset_index()
    )

    # Flatten column names
    failure_analysis.columns = [
        COL_FAILURE_STAGE,
        COL_FAILURE_REASON,
        COL_FAILURE_COUNT,
        COL_MODEL_LIST,
        COL_DEVICE_LIST,
        COL_RUNTIME_LIST,
        COL_JOB_URL,
    ]

    # Create count and detail columns
    failure_analysis[COL_DEVICE_COUNT] = failure_analysis[COL_DEVICE_LIST].apply(len)
    failure_analysis[COL_RUNTIME_COUNT] = failure_analysis[COL_RUNTIME_LIST].apply(len)
    failure_analysis[COL_MODEL_COUNT] = failure_analysis[COL_MODEL_LIST].apply(len)
    failure_analysis[COL_AFFECTED_DEVICES] = failure_analysis[COL_DEVICE_LIST].apply(
        lambda x: "; ".join(x)
    )
    failure_analysis[COL_AFFECTED_RUNTIMES] = failure_analysis[COL_RUNTIME_LIST].apply(
        lambda x: "; ".join(x)
    )
    failure_analysis[COL_MODELS] = failure_analysis[COL_MODEL_LIST].apply(
        lambda x: "; ".join(sorted(x)[:5])
        + (f" (+{len(x) - 5} more)" if len(x) > 5 else "")
    )

    # Sort by stage, then by failure count (descending)
    stage_order = {"Quantize": 1, "Compile": 2, "Profile": 3, "Inference": 4}
    failure_analysis[COL_STAGE_ORDER] = failure_analysis[COL_FAILURE_STAGE].map(
        stage_order
    )
    failure_analysis = failure_analysis.sort_values(
        [COL_STAGE_ORDER, COL_FAILURE_COUNT], ascending=[True, False]
    )

    # Select final columns
    return failure_analysis[
        [
            COL_FAILURE_STAGE,
            COL_FAILURE_REASON,
            COL_FAILURE_COUNT,
            COL_DEVICE_COUNT,
            COL_RUNTIME_COUNT,
            COL_MODEL_COUNT,
            COL_JOB_URL,
            COL_AFFECTED_DEVICES,
            COL_AFFECTED_RUNTIMES,
            COL_MODELS,
        ]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze scorecard CSV files and generate clear failure summary"
    )
    parser.add_argument("input_csv", help="Path to the input scorecard CSV file")
    parser.add_argument(
        "--output-file",
        default="scorecard_failure_analysis.csv",
        help="Output CSV file name (default: scorecard_failure_analysis.csv)",
    )

    args = parser.parse_args()

    # Read input CSV
    print(f"Reading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows from CSV")

    # Generate failure analysis
    print("Generating failure analysis...")
    analysis_df = create_failure_analysis(df)

    # Save to CSV
    output_path = Path(args.output_file)
    analysis_df.to_csv(output_path, index=False)
    print(f"Saved failure analysis to: {output_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS SUMMARY")
    print("=" * 80)

    if not analysis_df.empty:
        # Column names for accessing the analysis results
        COL_FAILURE_REASON = "Failure_Reason"
        COL_FAILURE_COUNT = "Failure_Count"
        COL_MODEL_COUNT = "Model_Count"
        COL_DEVICE_COUNT = "Device_Count"
        COL_RUNTIME_COUNT = "Runtime_Count"

        total_failures = analysis_df[COL_FAILURE_COUNT].sum()
        unique_failure_types = len(analysis_df)

        print(f"Total failures: {total_failures}")
        print(f"Unique failure types: {unique_failure_types}")
        print("\nTop 5 failure reasons:")

        for i, (_, row) in enumerate(analysis_df.head(5).iterrows(), 1):
            print(f"{i}. {row[COL_FAILURE_REASON]}: {row[COL_FAILURE_COUNT]} failures")
            print(
                f"   Affects {row[COL_MODEL_COUNT]} models across {row[COL_DEVICE_COUNT]} devices and {row[COL_RUNTIME_COUNT]} runtimes"
            )
    else:
        print("No failures found in the data!")

    print(f"\nDetailed analysis saved to: {output_path}")

    return 0


if __name__ == "__main__":
    main()

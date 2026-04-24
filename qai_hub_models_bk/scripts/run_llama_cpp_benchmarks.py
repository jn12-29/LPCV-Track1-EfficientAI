# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Run Llama.CPP benchmarks on QDC devices and generate perf.yaml files.

This script is called by the LlamaCppBenchmarkTask in scripts/tasks/test.py.
It requires QAIHM to be installed and handles:
- Running benchmarks via QDC
- Writing GitHub action summaries
- Generating perf.yaml files using QAIHMModelPerf
"""

from __future__ import annotations

import argparse
import os
import shutil

from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard import ScorecardDevice, ScorecardProfilePath
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT
from qai_hub_models.utils.qdc.llama_cpp_jobs import (
    ALL_COMPUTE_DEVICES,
    CONTEXT_LENGTHS,
    submit_llama_cpp_to_qdc_device,
)

# Llama.CPP model configurations with URLs and HTP device settings
# Each HTP PD is limited to ~3GB. Large models need multiple HTP devices.
# Keys are model_id (the model directory name in qai_hub_models/models/)
LLAMA_CPP_MODEL_CONFIGS: dict[str, dict[str, str | int]] = {
    "llama_v3_2_3b_instruct": {
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf",
        "htp_device_count": 1,
        "htp_devices": "HTP0",
        "name": "Llama-3.2-3B",
    },
    "gpt_oss_20b": {
        "url": "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf",
        "htp_device_count": 4,
        "htp_devices": "HTP0,HTP1,HTP2,HTP3",
        "name": "GPT-OSS-20B",
    },
    "gemma_3n_e4b": {
        "url": "https://huggingface.co/ggml-org/gemma-3n-E4B-it-GGUF/resolve/main/gemma-3n-E4B-it-Q8_0.gguf",
        "htp_device_count": 1,
        "htp_devices": "HTP0",
        "name": "Gemma-3n-E4B-it",
    },
    "olmoe_1b_7b_0125": {
        "url": "https://huggingface.co/allenai/OLMoE-1B-7B-0125-GGUF/resolve/main/OLMoE-1B-7B-0125-Q4_0.gguf",
        "htp_device_count": 2,
        "htp_devices": "HTP0,HTP1",
        "name": "OLMoE-1B-7B-0125",
    },
}

# Mapping from compute device to runtime key for perf.yaml
COMPUTE_TO_RUNTIME: dict[str, ScorecardProfilePath] = {
    "cpu": ScorecardProfilePath.LLAMA_CPP_CPU,
    "gpu": ScorecardProfilePath.LLAMA_CPP_GPU,
    "htp": ScorecardProfilePath.LLAMA_CPP_NPU,
}

# Device to chipset mapping
DEVICE_TO_CHIPSET = {
    "Snapdragon 8 Elite QRD": "qualcomm-snapdragon-8-elite",
    "Snapdragon 8 Elite Gen 5 QRD": "qualcomm-snapdragon-8-elite-gen5",
}


def append_to_github_summary(content: str) -> None:
    """Append content to GitHub Actions step summary if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(content)
            f.write("\n")
    else:
        print(content)


def validate_results(
    results: dict[str, dict[int, dict[str, float | str | None]]],
) -> list[tuple[str, int]]:
    """Validate benchmark results and return list of missing benchmarks."""
    missing = []
    for compute_unit in ALL_COMPUTE_DEVICES:
        for ctx_len in CONTEXT_LENGTHS:
            metrics = results.get(compute_unit, {}).get(ctx_len, {})
            tps_value = metrics.get("tps")
            if tps_value is None or tps_value == 0:
                missing.append((compute_unit, ctx_len))
    return missing


def write_github_summary(
    model_id: str,
    device: str,
    results: dict[str, dict[int, dict[str, float | str | None]]],
) -> None:
    """Write benchmark results to GitHub step summary."""
    # Count successful and failed benchmarks
    total = 0
    success = 0
    for compute_unit in ALL_COMPUTE_DEVICES:
        for ctx_len in CONTEXT_LENGTHS:
            total += 1
            metrics = results.get(compute_unit, {}).get(ctx_len, {})
            tps_val = metrics.get("tps")
            if isinstance(tps_val, (int, float)) and tps_val > 0:
                success += 1

    # Determine status emoji
    status = "pass" if success == total else "fail"
    status_emoji = ":white_check_mark:" if status == "pass" else ":x:"

    # Get model config for HTP info and display name
    model_config = LLAMA_CPP_MODEL_CONFIGS.get(model_id, {})
    model_name = str(model_config.get("name", model_id))
    htp_device_count = model_config.get("htp_device_count", 1)
    htp_devices = model_config.get("htp_devices", "HTP0")

    lines = [
        f"## Llama.CPP Benchmark Results {status_emoji}",
        "",
        f"**Model:** {model_name}",
        f"**Device:** {device}",
        f"**HTP Devices:** {htp_device_count} ({htp_devices})",
        f"**Status:** {success}/{total} benchmarks completed",
        "",
    ]

    # Add commands for all compute units
    lines.append("### Commands")
    lines.append("")
    for compute_unit in ALL_COMPUTE_DEVICES:
        # Get command from first context length that has one
        for ctx_len in CONTEXT_LENGTHS:
            metrics = results.get(compute_unit, {}).get(ctx_len, {})
            command = metrics.get("command")
            if command:
                lines.append(f"**{compute_unit.upper()}:** `{command}`")
                lines.append("")
                break

    lines.extend(
        [
            "| Compute | Context | Gen TPS | Prompt TPS | TTFT (ms) | Status |",
            "|---------|---------|---------|------------|-----------|--------|",
        ]
    )

    for compute_unit in ALL_COMPUTE_DEVICES:
        for ctx_len in CONTEXT_LENGTHS:
            metrics = results.get(compute_unit, {}).get(ctx_len, {})
            tps_value = metrics.get("tps")
            prompt_tps_value = metrics.get("prompt_tps")
            ttft_value = metrics.get("ttft_ms")
            tps = f"{tps_value:.2f}" if tps_value is not None else "-"
            prompt_tps = (
                f"{prompt_tps_value:.2f}" if prompt_tps_value is not None else "-"
            )
            ttft = f"{ttft_value:.2f}" if ttft_value is not None else "-"
            row_status = (
                ":white_check_mark:"
                if isinstance(tps_value, (int, float)) and tps_value > 0
                else ":x:"
            )
            lines.append(
                f"| {compute_unit.upper()} | {ctx_len} | {tps} | {prompt_tps} | {ttft} | {row_status} |"
            )

    lines.append("")
    append_to_github_summary("\n".join(lines))


def write_perf_yaml(
    model_id: str,
    results: dict[str, dict[str, dict[int, dict[str, float | str | None]]]],
    results_dir: str,
) -> None:
    """Generate perf.yaml for the model from benchmark results using QAIHMModelPerf.

    Parameters
    ----------
    model_id
        Model ID (e.g., "gpt_oss_20b") - matches the directory name in qai_hub_models/models/
    results
        Nested dict: {device: {compute_unit: {context_length: {metrics}}}}
    results_dir
        Directory to save the perf.yaml file
    """
    devices = list(results.keys())
    chipsets = [DEVICE_TO_CHIPSET.get(d, d) for d in devices]

    # Build performance_metrics structure using QAIHMModelPerf types
    performance_metrics: dict[
        ScorecardDevice,
        dict[ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails],
    ] = {}

    for device, device_results in results.items():
        scorecard_device = ScorecardDevice.get(device, return_unregistered=True)
        device_metrics: dict[
            ScorecardProfilePath, QAIHMModelPerf.PerformanceDetails
        ] = {}

        for compute_unit, ctx_results in device_results.items():
            runtime_key = COMPUTE_TO_RUNTIME.get(compute_unit)
            if runtime_key is None:
                continue

            llm_metrics_list = []

            for ctx_len, metrics in sorted(ctx_results.items()):
                tps_raw = metrics.get("tps")
                prompt_tps_raw = metrics.get("prompt_tps")

                # Skip if tps is not a valid number
                if not isinstance(tps_raw, (int, float)):
                    continue
                tps: float = float(tps_raw)

                # Calculate TTFT range if prompt_tps is valid
                if isinstance(prompt_tps_raw, (int, float)) and prompt_tps_raw > 0:
                    prompt_tps: float = float(prompt_tps_raw)
                    ttft_min = (128.0 / prompt_tps) * 1000
                    ttft_max = (float(ctx_len) / prompt_tps) * 1000
                    ttft_range = QAIHMModelPerf.PerformanceDetails.TimeToFirstTokenRangeMilliseconds(
                        min=round(ttft_min, 1),
                        max=round(ttft_max, 1),
                    )
                else:
                    ttft_range = QAIHMModelPerf.PerformanceDetails.TimeToFirstTokenRangeMilliseconds(
                        min=0.0,
                        max=0.0,
                    )

                llm_metrics_list.append(
                    QAIHMModelPerf.PerformanceDetails.LLMMetricsPerContextLength(
                        context_length=ctx_len,
                        tokens_per_second=round(tps, 2),
                        time_to_first_token_range_milliseconds=ttft_range,
                    )
                )

            if llm_metrics_list:
                device_metrics[runtime_key] = QAIHMModelPerf.PerformanceDetails(
                    llm_metrics=llm_metrics_list
                )

        if device_metrics:
            performance_metrics[scorecard_device] = device_metrics

    # Determine precision from model URL
    model_config = LLAMA_CPP_MODEL_CONFIGS.get(model_id, {})
    model_name = str(model_config.get("name", model_id))
    model_url = str(model_config.get("url", ""))
    if "mxfp4" in model_url.lower():
        precision = "mxfp4"
    elif "q4_0" in model_url.lower():
        precision = "q4_0"
    elif "q8_0" in model_url.lower():
        precision = "q8_0"
    else:
        precision = "default"

    # Build component details with performance metrics
    component_details = QAIHMModelPerf.ComponentDetails(
        performance_metrics=performance_metrics
    )

    # Build precision details (use display name for component key)
    precision_details = QAIHMModelPerf.PrecisionDetails(
        components={model_name: component_details}
    )

    # Build the full perf yaml structure
    perf_yaml = QAIHMModelPerf(
        supported_devices=[
            ScorecardDevice.get(d, return_unregistered=True) for d in devices
        ],
        supported_chipsets=chipsets,
        precisions={Precision.parse(precision): precision_details},
    )

    # Save to file: {results_dir}/{model_id}/perf.yaml
    model_dir = os.path.join(results_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)
    yaml_path = os.path.join(model_dir, "perf.yaml")

    perf_yaml.to_yaml(yaml_path)
    print(f"Generated perf.yaml: {yaml_path}")


def copy_results_to_model_dirs(results_dir: str) -> int:
    """Copy perf.yaml files from results directory to model directories.

    Only copies for models with genie_compatible=false (llama.cpp models,
    not QNN context binary models).

    Parameters
    ----------
    results_dir
        Directory containing benchmark results (e.g., llama_cpp_results/)

    Returns
    -------
    int
        Number of perf.yaml files copied
    """
    copied_count = 0

    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return copied_count

    for model_id in os.listdir(results_dir):
        model_results_dir = os.path.join(results_dir, model_id)
        if not os.path.isdir(model_results_dir):
            continue

        perf_yaml_path = os.path.join(model_results_dir, "perf.yaml")
        target_dir = QAIHM_MODELS_ROOT / model_id

        if not target_dir.exists():
            print(f"Warning: Target directory {target_dir} not found for {model_id}")
            continue

        if not os.path.exists(perf_yaml_path):
            print(f"Warning: No perf.yaml found for {model_id}")
            continue

        # Check if model has genie_compatible=true
        try:
            info = QAIHMModelInfo.from_model(model_id)
            is_genie_compatible = (
                info.llm_details.genie_compatible if info.llm_details else False
            )
        except Exception as e:
            print(f"Warning: Could not load info for {model_id}: {e}")
            is_genie_compatible = False

        if is_genie_compatible:
            print(f"Skipping {model_id} (genie_compatible=true)")
            continue

        # Copy perf.yaml to model directory
        target_path = target_dir / "perf.yaml"
        shutil.copy2(perf_yaml_path, target_path)
        print(f"Copied perf.yaml for {model_id}")
        copied_count += 1

    return copied_count


def run_benchmarks(
    model_ids: list[str],
    devices: list[str],
    api_token: str,
    llama_cpp_path: str,
    results_dir: str,
) -> bool:
    """Run Llama.CPP benchmarks for specified models and devices.

    Returns True if all benchmarks succeeded, False otherwise.
    """
    all_success = True

    # Accumulate results per model: {model_id: {device: {compute: {ctx: metrics}}}}
    all_results: dict[str, dict[str, dict]] = {}

    for model_id in model_ids:
        model_config = LLAMA_CPP_MODEL_CONFIGS[model_id]
        model_name = str(model_config.get("name", model_id))
        model_url = str(model_config["url"])
        htp_config = {
            "count": model_config.get("htp_device_count", 1),
            "devices": model_config.get("htp_devices", "HTP0"),
        }

        all_results[model_id] = {}

        for device in devices:
            print(f"Running benchmark: {model_name} on {device}")
            print(
                f"  HTP config: {htp_config['count']} device(s) ({htp_config['devices']})"
            )

            # Run the benchmark
            results = submit_llama_cpp_to_qdc_device(
                api_token=api_token,
                device=device,
                llama_cpp_path=llama_cpp_path,
                model_url=model_url,
                job_name=f"Llama.CPP CI - {model_name} on {device}",
                htp_config=htp_config,
            )

            # Store results for this device
            all_results[model_id][device] = results

            # Generate GitHub summary
            write_github_summary(model_id, device, results)

            # Validate that all benchmarks have TPS results
            missing_benchmarks = validate_results(results)
            if missing_benchmarks:
                print(f"ERROR: Missing TPS for {len(missing_benchmarks)} benchmarks:")
                for compute, ctx in missing_benchmarks:
                    print(f"  - {compute.upper()} @ CTX={ctx}")
                all_success = False

        # Generate perf.yaml for this model
        write_perf_yaml(model_id, all_results[model_id], results_dir)

    return all_success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Llama.CPP benchmarks on QDC devices and generate perf.yaml files."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated list of model IDs, or 'all' for all models. "
        f"Available: {', '.join(LLAMA_CPP_MODEL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=list(DEVICE_TO_CHIPSET.keys()),
        default=None,
        help="Single device to run benchmarks on. Takes precedence over --devices if provided.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="Snapdragon 8 Elite Gen 5 QRD",
        help="Comma-separated list of devices to run benchmarks on.",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help="QDC API token. If not provided, uses QDC_API_KEY env var.",
    )
    parser.add_argument(
        "--llama-cpp-path",
        type=str,
        default=None,
        help="Path to Llama.CPP build directory. If not provided, uses LLAMA_CPP_PATH env var.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save perf.yaml files. If not provided, uses LLAMA_CPP_RESULTS_DIR env var or 'llama_cpp_results'.",
    )

    args = parser.parse_args()

    # Parse model IDs
    if not args.models or args.models.lower() == "all":
        model_ids = list(LLAMA_CPP_MODEL_CONFIGS.keys())
    else:
        model_ids = [m.strip() for m in args.models.split(",")]

    # Validate model IDs
    for model_id in model_ids:
        if model_id not in LLAMA_CPP_MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_id}. "
                f"Available models: {list(LLAMA_CPP_MODEL_CONFIGS.keys())}"
            )

    # Parse devices (--device takes precedence over --devices)
    if args.device:
        devices = [args.device]
    else:
        devices = [d.strip() for d in args.devices.split(",")]

    # Get API token
    api_token = args.api_token or os.environ.get("QDC_API_KEY", "")
    if not api_token:
        raise ValueError(
            "QDC API token must be provided via --api-token or QDC_API_KEY env var"
        )

    # Get llama.cpp path
    llama_cpp_path = args.llama_cpp_path or os.environ.get(
        "LLAMA_CPP_PATH", "llama.cpp"
    )
    if not os.path.exists(llama_cpp_path):
        raise FileNotFoundError(f"Llama.CPP path not found: {llama_cpp_path}")

    # Get results directory
    results_dir = args.results_dir or os.environ.get(
        "LLAMA_CPP_RESULTS_DIR", "llama_cpp_results"
    )

    print("Running Llama.CPP benchmarks:")
    print(f"  Models: {', '.join(model_ids)}")
    print(f"  Devices: {', '.join(devices)}")
    print(f"  Llama.CPP path: {llama_cpp_path}")
    print(f"  Results directory: {results_dir}")
    print()

    success = run_benchmarks(
        model_ids=model_ids,
        devices=devices,
        api_token=api_token,
        llama_cpp_path=llama_cpp_path,
        results_dir=results_dir,
    )

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

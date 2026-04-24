# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import tempfile
import time
from abc import ABC, abstractmethod

from qualcomm_device_cloud_sdk.models import ArtifactType

from qai_hub_models.utils.qdc.qdc_jobs import (
    HUB_DEVICE_TO_QDC_DEVICE_MAP,
    POLL_INTERVAL,
    QDCDevice,
    QDCJobs,
)

# Llama.cpp jobs need longer timeout due to comprehensive benchmarking
LLAMA_CPP_JOB_TIMEOUT = 10800  # 3 hours

# Default HTP configuration for single device
DEFAULT_HTP_CONFIG: dict[str, str | int] = {"count": 1, "devices": "HTP0"}

# Available compute devices
ALL_COMPUTE_DEVICES = ["cpu", "gpu", "htp"]

# Context lengths to benchmark
CONTEXT_LENGTHS = [128, 1024, 4096]


class LlamaCppArtifactHandler(ABC):
    """Abstract base class for Llama.CPP artifact handlers."""

    @abstractmethod
    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        llama_cpp_path: os.PathLike | str,
        model_url: str,
        dest_dir: os.PathLike | str,
        htp_config: dict[str, str | int] | None = None,
    ) -> str:
        """Create artifact bundle and return path to the zip file."""
        raise NotImplementedError

    @property
    @abstractmethod
    def entry_script(self) -> str | None:
        raise NotImplementedError


class LlamaCppAndroidArtifactHandler(LlamaCppArtifactHandler):
    """Handler for Android Llama.CPP artifacts."""

    @property
    def entry_script(self) -> str | None:
        return None

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        llama_cpp_path: os.PathLike | str,
        model_url: str,
        dest_dir: os.PathLike | str,
        htp_config: dict[str, str | int] | None = None,
    ) -> str:
        # Use default HTP config if not provided
        if htp_config is None:
            htp_config = DEFAULT_HTP_CONFIG

        # Copy the test script
        test_folder = os.path.join(dest_dir, "tests")
        os.makedirs(test_folder, exist_ok=True)

        # Copy the run script and rename to test_appium.py
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "run_llama_cpp_posix.py"),
            os.path.join(test_folder, "test_appium.py"),
        )

        # Replace placeholders in the test script
        test_appium_path = os.path.join(test_folder, "test_appium.py")
        with open(test_appium_path, encoding="utf-8") as f:
            file_content = f.read()
        with open(test_appium_path, "w", encoding="utf-8") as f:
            file_content = file_content.replace("<<MODEL_URL>>", model_url)
            file_content = file_content.replace(
                "<<HTP_DEVICES>>", str(htp_config["count"])
            )
            file_content = file_content.replace(
                "<<HTP_DEVICE_LIST>>", str(htp_config["devices"])
            )
            f.write(file_content)

        # Requirements
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "requirements.txt"),
            dest_dir,
        )

        # Bundle the Llama.CPP content
        llama_cpp_folder = os.path.join(dest_dir, "llama_cpp_bundle")
        os.makedirs(llama_cpp_folder, exist_ok=True)
        shutil.copytree(llama_cpp_path, llama_cpp_folder, dirs_exist_ok=True)

        # Copy all sample prompt files
        device_scripts_dir = os.path.join(curr_dirname, "device_scripts")
        for prompt_file in glob.glob(
            os.path.join(device_scripts_dir, "sample_prompt_*.txt")
        ):
            shutil.copy(prompt_file, llama_cpp_folder)

        # Create zip directly in dest_dir (avoids polluting CWD)
        zip_path = os.path.join(dest_dir, "test")
        shutil.make_archive(zip_path, "zip", dest_dir)
        return f"{zip_path}.zip"


class LlamaCppQDCJobs(QDCJobs):
    """
    QDC job handler for Llama.CPP workloads.

    Handles uploading Llama.CPP binaries and prompt files
    to run comprehensive benchmarks on QDC devices (CPU, GPU, HTP)
    across multiple context lengths (128, 256, 512, 1024, 4096).
    """

    def _get_artifact_handler(self, qdc_device: QDCDevice) -> LlamaCppArtifactHandler:
        """Get the appropriate artifact handler based on device platform.

        Parameters
        ----------
        qdc_device
            QDCDevice instance (passed to avoid redundant instantiation).

        Returns
        -------
        llama_cpp_artifact_handler: LlamaCppArtifactHandler
            Instance of the appropriate LlamaCppArtifactHandler subclass.
        """
        # Check for Android platform (mobile format or os:android attribute)
        if qdc_device.mobile_platform or self._is_android_device(qdc_device):
            return LlamaCppAndroidArtifactHandler()
        raise ValueError(
            "Unsupported platform for Llama.CPP benchmarks. "
            "Device must be an Android device."
        )

    def _is_android_device(self, qdc_device: QDCDevice) -> bool:
        """Check if device is Android based on OS attribute."""
        for attr in qdc_device.device_attributes:
            if "os" in attr and "android" in attr:
                return True
        return False

    def add_job_artifacts(
        self,
        qdc_device: QDCDevice,
        llama_cpp_path: str,
        model_url: str,
        htp_config: dict[str, str | int] | None = None,
    ) -> tuple[list[str], str | None]:
        """
        Prepare and upload Llama.CPP artifacts for the job submission.

        Parameters
        ----------
        qdc_device
            QDCDevice instance for the target device.
        llama_cpp_path
            Path to the Llama.CPP build directory containing binaries and libraries.
        model_url
            URL to download the model on-device.
        htp_config
            HTP device configuration with 'count' and 'devices' keys.

        Returns
        -------
        job_artifacts: list[str]
            Artifact IDs returned by QDC upload.
        entry_script: str | None
            Optional entry script path used by the test framework.
        """
        curr_dirname = os.path.dirname(os.path.abspath(__file__))
        artifact_handler = self._get_artifact_handler(qdc_device)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create artifact returns the path to the zip file
            zip_path = artifact_handler.create_artifact(
                curr_dirname,
                llama_cpp_path,
                model_url,
                tmpdirname,
                htp_config,
            )
            upload_response = self.upload_file(zip_path, ArtifactType.TESTSCRIPT)

        return [upload_response], artifact_handler.entry_script

    def compute_metrics(
        self,
        job_log_files: list,
    ) -> dict[str, dict[int, dict[str, float | str | None]]]:
        """
        Compute metrics for all compute units and context lengths from the job log.

        Returns nested dict: {compute_unit: {context_length: {tps, prompt_tps, ttft_ms, command}}}
        """
        log_content = self._extract_llama_log(job_log_files)
        if not log_content:
            return {}
        return self._parse_benchmark_log(log_content)

    def _try_decode(self, raw_bytes: bytes, encoding: str) -> str | None:
        """Try to decode bytes with a specific encoding, returning None on failure."""
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            return None

    def _try_read_file_with_encodings(self, file_path: str) -> str | None:
        """Try to read a file with various encodings, returning content or None."""
        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        # Check for BOM to determine encoding
        if raw_bytes.startswith(b"\xff\xfe"):
            # UTF-16-LE BOM
            return self._try_decode(raw_bytes, "utf-16-le")
        if raw_bytes.startswith(b"\xfe\xff"):
            # UTF-16-BE BOM
            return self._try_decode(raw_bytes, "utf-16-be")

        # Check for null bytes pattern indicating UTF-16-LE without BOM
        # (e.g., ASCII chars followed by null bytes like "=\x00=\x00")
        if len(raw_bytes) >= 4 and raw_bytes[1] == 0 and raw_bytes[3] == 0:
            result = self._try_decode(raw_bytes, "utf-16-le")
            if result is not None:
                return result

        # UTF-8 is the most common encoding for log files - use error replacement
        # to handle any invalid byte sequences instead of failing entirely
        try:
            return raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            pass

        # Fallback to trying other encodings only if UTF-8 completely fails
        for encoding in ["utf-16", "utf-16-le"]:
            result = self._try_decode(raw_bytes, encoding)
            if result is not None:
                return result
        return None

    def _extract_llama_log(
        self,
        job_log_files: list,
    ) -> str | None:
        """Extract llama_cpp.log content from job log files."""
        print(f"Searching for llama_cpp.log in {len(job_log_files)} job log files...")
        with tempfile.TemporaryDirectory() as tmpdir:
            for job_log in job_log_files:
                print(f"  Processing log file: {job_log.filename}")
                zip_path = os.path.join(tmpdir, "log.zip")
                self.download_job_log_files(job_log.filename, zip_path)
                shutil.unpack_archive(zip_path, tmpdir, "zip")

                # Find and read llama_cpp.log
                for root, _, files in os.walk(tmpdir):
                    if "llama_cpp.log" in files:
                        log_path = os.path.join(root, "llama_cpp.log")
                        print(f"  Found llama_cpp.log at: {log_path}")
                        log_content = self._try_read_file_with_encodings(log_path)
                        if log_content is None:
                            continue

                        return log_content

        print("Warning: llama_cpp.log not found in any job log files")
        return None

    def _parse_benchmark_log(
        self, log_content: str
    ) -> dict[str, dict[int, dict[str, float | str | None]]]:
        """
        Parse benchmark log to extract TPS, prompt TPS, TTFT metrics, and command.

        Log format: === CPU | CTX=128 ===
        Metrics extracted from single run:
          - prompt eval time -> prompt_tps -> used to calculate TTFT
          - eval time -> gen_tps (generation TPS)
        Command format: COMMAND: llama-completion ...
        """
        # Initialize results with all metrics including prompt_tps and command
        results: dict[str, dict[int, dict[str, float | str | None]]] = {
            compute: {
                ctx: {
                    "tps": None,
                    "prompt_tps": None,
                    "ttft_ms": None,
                    "command": None,
                }
                for ctx in CONTEXT_LENGTHS
            }
            for compute in ALL_COMPUTE_DEVICES
        }

        # Parse sections: === COMPUTE | CTX=N ===
        section_pattern = r"=== (\w+) \| CTX=(\d+) ==="
        sections = re.split(section_pattern, log_content)

        # Process sections (groups of 3: compute, ctx_len, content)
        for i in range(1, len(sections) - 2, 3):
            compute = sections[i].lower()
            ctx_len = int(sections[i + 1])
            content = sections[i + 2]

            if compute not in results or ctx_len not in results[compute]:
                continue

            # Extract command from content
            command = self._extract_command(content)

            # Extract metrics from content
            prompt_tps, gen_tps = self._extract_tps_metrics(content)

            # Store all metrics from single run
            if gen_tps:
                results[compute][ctx_len]["tps"] = gen_tps
            if prompt_tps:
                results[compute][ctx_len]["prompt_tps"] = prompt_tps
            if command:
                results[compute][ctx_len]["command"] = command

            # Calculate TTFT from prompt_tps and gen_tps
            if prompt_tps and gen_tps:
                # TTFT = time to process ctx_len prompt tokens + time to output first token
                results[compute][ctx_len]["ttft_ms"] = (
                    float(ctx_len) / prompt_tps
                ) * 1000

        return results

    def _extract_command(self, content: str) -> str | None:
        """Extract command string from log content."""
        for line in content.split("\n"):
            if line.startswith("COMMAND:"):
                return line.replace("COMMAND:", "").strip()
        return None

    def _extract_tps_metrics(self, content: str) -> tuple[float | None, float | None]:
        """Extract prompt and generation TPS from log content.

        Supports two output formats:
        1. llama-cli: [ Prompt: X t/s | Generation: Y t/s ]
        2. llama-completion: common_perf_print: prompt eval time = ... tokens per second)
        """
        prompt_tps = None
        gen_tps = None

        for line in content.split("\n"):
            # Format 1 (llama-cli): [ Prompt: X t/s | Generation: Y t/s ]
            if "Prompt:" in line and "Generation:" in line:
                prompt_match = re.search(r"Prompt:\s*([\d.]+)\s*t/s", line)
                gen_match = re.search(r"Generation:\s*([\d.]+)\s*t/s", line)
                if prompt_match:
                    prompt_tps = float(prompt_match.group(1))
                if gen_match:
                    gen_tps = float(gen_match.group(1))

            # Format 2 (llama-completion): common_perf_print: prompt eval time = X ms / Y tokens (Z ms per token, W tokens per second)
            if "prompt eval time" in line and "tokens per second" in line:
                match = re.search(r"([\d.]+)\s*tokens per second", line)
                if match:
                    prompt_tps = float(match.group(1))

            # Format 2 (llama-completion): common_perf_print: eval time = X ms / Y runs (Z ms per token, W tokens per second)
            if (
                "eval time" in line
                and "tokens per second" in line
                and "prompt" not in line
            ):
                match = re.search(r"([\d.]+)\s*tokens per second", line)
                if match:
                    gen_tps = float(match.group(1))

        return prompt_tps, gen_tps


def submit_llama_cpp_to_qdc_device(
    api_token: str,
    device: str,
    llama_cpp_path: str,
    model_url: str,
    job_name: str = "LLM Llama.CPP",
    htp_config: dict[str, str | int] | None = None,
) -> dict[str, dict[int, dict[str, float | str | None]]]:
    """
    Submit a Llama.CPP bundle to QDC for comprehensive benchmarking.

    This runs CPU, GPU, and HTP benchmarks across all context lengths
    (128, 256, 512, 1024, 4096) in a single QDC job.

    Parameters
    ----------
    api_token
        API token for QDC authentication.
    device
        Hub device name to run the job on.
    llama_cpp_path
        Path to the Llama.CPP build directory containing binaries.
    model_url
        URL to download the model on-device.
    job_name
        Name for the QDC job.
    htp_config
        HTP device configuration with 'count' and 'devices' keys.

    Returns
    -------
    job_config: dict[str, dict[int, dict[str, float | str | None]]]
        Nested dictionary: {compute_unit: {context_length: {metric_name: value}}}
    """
    print(f"\n{'=' * 70}")
    print("Running comprehensive Llama.CPP benchmarks")
    print(f"  Compute units: {', '.join(c.upper() for c in ALL_COMPUTE_DEVICES)}")
    print(f"  Context lengths: {', '.join(str(c) for c in CONTEXT_LENGTHS)}")
    if htp_config:
        print(f"  HTP devices: {htp_config['count']} ({htp_config['devices']})")
    print(f"{'=' * 70}\n")

    # Create QDCDevice once and reuse it
    qdc_device = QDCDevice(device)

    llama_cpp_job = LlamaCppQDCJobs(
        api_key=api_token,
        app_name_header="LlamaCppQDCJobApp",
    )

    job_artifacts, entry_script = llama_cpp_job.add_job_artifacts(
        qdc_device,
        llama_cpp_path,
        model_url,
        htp_config,
    )

    job_id = llama_cpp_job.submit_automated_job(
        qdc_device, job_artifacts, entry_script, job_name=job_name
    )
    if job_id is None:
        raise RuntimeError("Job submission failed.")

    print(f"Job {job_id} submitted. Waiting for completion...")
    job_status = llama_cpp_job.status(job_id, timeout=LLAMA_CPP_JOB_TIMEOUT)
    print(f"Job {job_id}: {job_status}")
    llama_cpp_job.log_upload_status(job_id)
    job_log_files = llama_cpp_job.get_job_log_files(job_id)
    time.sleep(POLL_INTERVAL)

    # Parse results
    results = llama_cpp_job.compute_metrics(job_log_files)

    # Print summary table with prompt TPS
    print(f"\n{'=' * 70}")
    print(
        f"{'COMPUTE':<8} {'CTX':<6} {'Gen TPS':>10} {'Prompt TPS':>12} "
        f"{'TTFT (ms)':>12}"
    )
    print(f"{'=' * 70}")

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
            print(
                f"{compute_unit.upper():<8} {ctx_len:<6} {tps:>10} {prompt_tps:>12} "
                f"{ttft:>12}"
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit Llama.CPP benchmark job to QDC. "
        "Runs comprehensive benchmarks on all compute units (CPU, GPU, HTP) "
        "across context lengths (128, 256, 512, 1024, 4096)."
    )
    parser.add_argument(
        "--api-token",
        type=str,
        required=True,
        help="API token for authentication.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=HUB_DEVICE_TO_QDC_DEVICE_MAP.keys(),
        help="Device to use for the job.",
    )
    parser.add_argument(
        "--llama-cpp-path",
        type=str,
        required=True,
        help="Directory containing Llama.CPP binaries and libraries.",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        required=True,
        help="URL to download the model on-device.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=False,
        default="LLM Llama.CPP",
        help="QDC job name.",
    )

    args = parser.parse_args()
    if not os.path.exists(args.llama_cpp_path):
        raise FileNotFoundError(f"Llama.CPP path not found: {args.llama_cpp_path}")

    submit_llama_cpp_to_qdc_device(
        args.api_token,
        args.device,
        args.llama_cpp_path,
        args.model_url,
        args.job_name,
    )

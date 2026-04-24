# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import zipfile
from abc import ABC, abstractmethod

from qualcomm_device_cloud_sdk.models import ArtifactType

from qai_hub_models.utils.qdc.qdc_jobs import (
    HUB_DEVICE_TO_QDC_DEVICE_MAP,
    POLL_INTERVAL,
    QDCDevice,
    QDCJobs,
)


def create_zip(zip_path: str, source_dir: str | os.PathLike) -> None:
    """Create a zip archive from source_dir at zip_path."""
    if isinstance(source_dir, os.PathLike):
        source_dir = str(source_dir)

    files_to_zip = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, source_dir)
            files_to_zip.append((file_path, arcname))

    # Use ZIP_STORED (no compression) for speed - the files are already compressed binaries
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for file_path, arcname in files_to_zip:
            zf.write(file_path, arcname)


class GenieArtifactHandler(ABC):
    """Abstract base class for Genie artifact handlers."""

    @abstractmethod
    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
    ) -> str:
        """Create artifact bundle and return path to the zip file."""
        raise NotImplementedError

    @property
    @abstractmethod
    def entry_script(self) -> str | None:
        raise NotImplementedError


class GenieAndroidArtifactHandler(GenieArtifactHandler):
    def __init__(self, test_script: str) -> None:
        self.test_script: str = test_script

    @property
    def entry_script(self) -> str | None:
        return None

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
    ) -> str:
        # Copy the test script
        test_folder = os.path.join(dest_dir, "tests")
        os.makedirs(test_folder, exist_ok=True)

        # Copy 'run_posix.py' and rename it to 'test_appium.py' since pytest looks for files starting with 'test_'.
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", self.test_script),
            os.path.join(test_folder, "test_appium.py"),
        )

        # Replace the Hexagon and QAIRT version placeholders with actual values
        test_appium_path = os.path.join(test_folder, "test_appium.py")
        with open(test_appium_path, encoding="utf-8") as f:
            file_content = f.read()
        with open(test_appium_path, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("<<HEXAGON_VERSION>>", hexagon_version).replace(
                    "<<QAIRT_VERSION>>", qairt_version
                )
            )

        # Requirements
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "requirements.txt"),
            dest_dir,
        )

        # Bundle the Genie test content
        genie_folder = os.path.join(dest_dir, "genie_bundle")
        os.makedirs(genie_folder, exist_ok=True)
        shutil.copytree(genie_bundle_path, genie_folder, dirs_exist_ok=True)

        # Create zip in parent directory to avoid zipping the zip itself
        zip_path = os.path.join(os.path.dirname(dest_dir), "test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class GenieAutoArtifactHandler(GenieAndroidArtifactHandler):
    """Artifact handler for automotive (auto) devices.

    Extends the Android handler by bundling the QAIRT SDK into the artifact,
    since auto devices cannot download it at runtime.
    """

    def __init__(self, test_script: str, qairt_sdk_path: str) -> None:
        """
        Parameters
        ----------
        test_script
            Filename of the Appium/PyTest script to bundle (e.g., ``run_auto_posix.py``).
        qairt_sdk_path
            Path to the QAIRT SDK zip file to bundle with the artifact.
            Must be an accessible, valid zip file.
        """
        super().__init__(test_script)
        if not os.path.isfile(qairt_sdk_path):
            raise FileNotFoundError(
                f"QAIRT SDK path '{qairt_sdk_path}' does not exist or is not a file. "
                "Please verify the --qairt-sdk-path argument."
            )
        self.qairt_sdk_path: str = qairt_sdk_path

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
    ) -> str:
        # Build the standard Android artifact first
        zip_path = super().create_artifact(
            curr_dirname, genie_bundle_path, dest_dir, hexagon_version, qairt_version
        )

        # Append the QAIRT SDK into the artifact zip under genie_bundle/
        print(
            f"[QDC] Adding QAIRT SDK from {self.qairt_sdk_path} to {zip_path}...",
            flush=True,
        )
        with zipfile.ZipFile(zip_path, "a") as zf:
            zf.write(
                self.qairt_sdk_path,
                arcname=os.path.join("genie_bundle", "qairt_sdk.zip"),
            )
        print("[QDC] QAIRT SDK addition to zip complete", flush=True)
        return zip_path


class GenieLinuxArtifactHandler(GenieArtifactHandler):
    """Artifact handler for Linux IoT devices (e.g., IQ9).
    Uses Bash test framework — no Appium wrapper needed.
    """

    @property
    def entry_script(self) -> str:
        return "/bin/bash /data/local/tmp/TestContent/run_linux.sh"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
    ) -> str:
        # Copy the bash script directly into dest_dir
        script_dest = os.path.join(dest_dir, "run_linux.sh")
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "run_linux.sh"),
            script_dest,
        )

        # Replace the Hexagon and QAIRT version placeholders with actual values
        with open(script_dest, encoding="utf-8") as f:
            file_content = f.read()
        with open(script_dest, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("{HEXAGON_VERSION}", hexagon_version).replace(
                    "{QAIRT_VERSION}", qairt_version
                )
            )

        # Bundle the Genie test content
        genie_folder = os.path.join(dest_dir, "genie_bundle")
        os.makedirs(genie_folder, exist_ok=True)
        shutil.copytree(genie_bundle_path, genie_folder, dirs_exist_ok=True)

        # Create zip in parent directory to avoid zipping the zip itself
        zip_path = os.path.join(os.path.dirname(dest_dir), "test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class GenieWindowsArtifactHandler(GenieArtifactHandler):
    @property
    def entry_script(self) -> str:
        return "C:\\Temp\\TestContent\\run_windows.ps1"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
    ) -> str:
        # Copy the PowerShell script
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "run_windows.ps1"),
            dest_dir,
        )
        dest_script = os.path.join(dest_dir, "run_windows.ps1")
        shutil.copytree(genie_bundle_path, dest_dir, dirs_exist_ok=True)
        # Replace the Hexagon and QAIRT version placeholders with actual values
        with open(dest_script, encoding="utf-8") as f:
            file_content = f.read()
        with open(dest_script, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("{HEXAGON_VERSION}", hexagon_version).replace(
                    "{QAIRT_VERSION}", qairt_version
                )
            )

        # Create zip in parent directory to avoid zipping the zip itself
        zip_path = os.path.join(os.path.dirname(dest_dir), "test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class GenieQDCJobs(QDCJobs):
    """
    QDC job handler for Genie workloads.

    Handles uploading Genie bundles and parsing performance metrics
    from Genie benchmark logs.
    """

    def _get_artifact_handler(
        self, qdc_device: QDCDevice, qairt_sdk_path: str | None = None
    ) -> GenieArtifactHandler:
        """Get the appropriate artifact handler based on device platform.

        Parameters
        ----------
        qdc_device
            QDCDevice instance (passed to avoid redundant instantiation).
        qairt_sdk_path
            Path to the QAIRT SDK zip file. Required for auto devices.

        Returns
        -------
        genie_artifact_handler: GenieArtifactHandler
            Instance of the appropriate GenieArtifactHandler subclass.
        """
        if qdc_device.windows_platform:
            return GenieWindowsArtifactHandler()
        if qdc_device.iot_platform:
            return GenieLinuxArtifactHandler()
        if qdc_device.auto_platform:
            if qairt_sdk_path is None:
                raise ValueError(
                    "qairt_sdk_path is required for auto devices. "
                    "Please provide the path to the automotive QAIRT SDK zip file."
                )
            return GenieAutoArtifactHandler(
                test_script="run_auto_posix.py", qairt_sdk_path=qairt_sdk_path
            )
        if qdc_device.mobile_platform:
            return GenieAndroidArtifactHandler(test_script="run_posix.py")
        raise ValueError("Unsupported platform type for Genie artifact handler.")

    def add_job_artifacts(
        self,
        qdc_device: QDCDevice,
        genie_bundle_path: str,
        qairt_sdk_path: str | None = None,
        qairt_version: str = "2.45.40.260406",
    ) -> tuple[list[str], str | None]:
        """Prepare and upload Genie artifacts for the job submission.

        Parameters
        ----------
        qdc_device
            QDCDevice instance for the target device.
        genie_bundle_path
            Directory path containing the genie bundle.
        qairt_sdk_path
            Path to the QAIRT SDK zip file. Required for auto devices.
        qairt_version
            QAIRT SDK version to download on-device (e.g. ``"2.45.40.260406"``).

        Returns
        -------
        job_artifacts: list[str]
            List of artifact IDs returned by QDC upload.
        entry_script: str | None
            Optional entry script path used by the test framework.
        """
        curr_dirname = os.path.dirname(os.path.abspath(__file__))
        artifact_handler = self._get_artifact_handler(qdc_device, qairt_sdk_path)

        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = artifact_handler.create_artifact(
                curr_dirname,
                genie_bundle_path,
                tmpdirname,
                qdc_device.hexagon_version,
                qairt_version,
            )
            upload_response = self.upload_file(zip_path, ArtifactType.TESTSCRIPT)
            # Clean up zip file created outside temp directory
            if os.path.exists(zip_path):
                os.unlink(zip_path)

        return [upload_response], artifact_handler.entry_script

    def compute_metrics(
        self,
        job_log_files: list,
    ) -> tuple[float | None, float | None]:
        """Compute and print performance metrics from job logs.

        Parameters
        ----------
        job_log_files
            List of job log files retrieved from QDC.

        Returns
        -------
        avg_tokens_per_second : float | None
            Average tokens per second.
        min_time_to_first_token: float | None
            Minimum time to first token in ms.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            tps: list[float] = []
            ttft: list[float] = []

            if job_log_files:
                for job_log in job_log_files:
                    target_path = os.path.join(
                        tmpdirname, "logs", f"{job_log.filename}.zip"
                    )
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    self.download_job_log_files(job_log.filename, target_path)

                    if "genie" in job_log.filename:
                        print("On device output (genie.log):")
                        shutil.unpack_archive(target_path, tmpdirname, "zip")
                        genie_log_path = os.path.join(tmpdirname, "genie.log")
                        displayed = False
                        for encoding in ("utf-8", "utf-16", "utf-16-le"):
                            try:
                                with open(genie_log_path, encoding=encoding) as file:
                                    genie_content = file.read()
                                    print(genie_content)
                                    displayed = True
                                    break
                            except Exception:
                                pass
                        if not displayed:
                            print(f"Warning: Could not read {genie_log_path}")

                    if "profile" in job_log.filename:
                        shutil.unpack_archive(target_path, tmpdirname, "zip")
                        profile_path = os.path.join(
                            tmpdirname, job_log.filename.split("/")[-1]
                        )
                        with open(profile_path, encoding="utf-8") as file:
                            file_content = json.loads(file.read())

                        components = file_content.get("components", [])
                        if (
                            isinstance(components, list)
                            and len(components) > 0
                            and isinstance(components[0], dict)
                            and "events" in components[0]
                            and isinstance(components[0]["events"], list)
                            and len(components[0]["events"]) > 1
                        ):
                            component = components[0]["events"][1]
                            tps.append(
                                float(component["token-generation-rate"]["value"])
                            )
                            ttft.append(
                                float(component["time-to-first-token"]["value"])
                            )
                        else:
                            print(
                                "Warning: Unexpected profile log structure, "
                                "skipping metrics for this file."
                            )

        if len(tps) > 0:
            avg_tps = sum(tps) / len(tps)
            # TTFT in profile logs is in microseconds, convert to milliseconds
            min_ttft_ms = (sum(ttft) / len(ttft)) / 1000.0

            print("Perf metrics:")
            print(f"  Average Tokens Per Second: {avg_tps:.2f}")
            print(f"  Min Time to First Token (ms): {min_ttft_ms:.2f}")
            return avg_tps, min_ttft_ms

        print("No performance metrics found.")
        return None, None


def submit_genie_bundle_to_qdc_device(
    api_token: str,
    device: str,
    genie_bundle_path: str,
    job_name: str = "LLM Genie",
    qairt_sdk_path: str | None = None,
    qairt_version: str = "2.45.40.260406",
) -> tuple[float | None, float | None]:
    """
    Submit a Genie bundle to QDC for execution on the specified device.

    Parameters
    ----------
    api_token
        API token for QDC authentication.
    device
        Hub device name to run the job on.
    genie_bundle_path
        Directory where genie files are stored. Must contain 'prompt.txt'.
    job_name
        Name of QDC job.
    qairt_sdk_path
        Path to the QAIRT SDK zip file. Required for auto devices.
    qairt_version
        QAIRT SDK version to download on-device (e.g. ``"2.45.40.260406"``).

    Returns
    -------
    avg_tokens_per_second: float | None
        Average tokens per second.
    min_time_to_first_token: float | None
        Minimum time to first token in ms (used as lower bound for TTFT range).
    """
    qdc_device = QDCDevice(device)
    genie_job = GenieQDCJobs(
        api_key=api_token,
        app_name_header="GenieQDCJobApp",
    )

    job_artifacts, entry_script = genie_job.add_job_artifacts(
        qdc_device, genie_bundle_path, qairt_sdk_path, qairt_version
    )

    job_id = genie_job.submit_automated_job(
        qdc_device, job_artifacts, entry_script, job_name=job_name
    )
    if job_id is None:
        raise RuntimeError("Job submission failed.")

    print(f"Submitted QDC job with ID: {job_id}")
    job_status = genie_job.status(job_id)
    print(f"QDC job {job_id} completed with status: {job_status}")
    genie_job.log_upload_status(job_id)
    job_log_files = genie_job.get_job_log_files(job_id)
    time.sleep(POLL_INTERVAL)
    return genie_job.compute_metrics(job_log_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--genie-bundle-path",
        type=str,
        required=True,
        help="Directory where genie files are stored.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=False,
        default="LLM Genie",
        help="QDC job name.",
    )
    parser.add_argument(
        "--qairt-sdk-path",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to QAIRT SDK zip file. Required when targeting automotive devices "
            "(e.g., SA8295P ADP, SA7255P ADP, SA8775P ADP). "
            "Omitting this for an auto device will raise a ValueError at job submission time."
        ),
    )
    parser.add_argument(
        "--qairt-version",
        type=str,
        required=False,
        default="2.45.40.260406",
        help="QAIRT SDK version to download on-device (e.g. 2.45.40.260406).",
    )

    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.genie_bundle_path, "sample_prompt.txt")):
        raise FileNotFoundError(
            f"sample_prompt.txt not found in {args.genie_bundle_path}. Please add a file with prompt to run on-device."
        )
    avg_tps, avg_ttft_ms = submit_genie_bundle_to_qdc_device(
        args.api_token,
        args.device,
        args.genie_bundle_path,
        args.job_name,
        args.qairt_sdk_path,
        args.qairt_version,
    )

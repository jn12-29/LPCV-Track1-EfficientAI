# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Appium/PyTest test script for running Genie on automotive (auto) devices.

This is a template file — the ``<<HEXAGON_VERSION>>`` placeholder is
substituted at artifact-build time by ``GenieAutoArtifactHandler.create_artifact``.

Requires:
  - An Android device reachable over ADB.
  - An Appium server running on localhost:4723.
  - ANDROID_DEVICE_VERSION env var set to the target device name.
  - The genie_bundle (including qairt_sdk.zip) already pushed to the device.
"""

import os
import subprocess
import sys

import pytest
from appium import webdriver
from appium.options.common import AppiumOptions

options = AppiumOptions()
options.set_capability("automationName", "UiAutomator2")
options.set_capability("platformName", "Android")
options.set_capability("deviceName", os.getenv("ANDROID_DEVICE_VERSION"))


class TestGenie:
    @pytest.fixture
    def driver(self) -> webdriver.Remote:
        return webdriver.Remote(
            command_executor="http://127.0.0.1:4723/wd/hub", options=options
        )

    def test_genie(self) -> None:
        # Use pre-uploaded QAIRT SDK for auto devices
        # script to set environment variables
        # run genie-t2t-run on the device
        genie_command = [
            f"genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt --profile /data/local/tmp/QDC_logs/profile{i}.txt"
            for i in range(10)
        ]
        full_genie_command = " && ".join(genie_command)
        qairt_path = "/data/local/tmp/genie_bundle/qairt"
        genie_script = f"""set -e
cd /data/local/tmp/genie_bundle
unzip qairt_sdk.zip -d /data/local/tmp/genie_bundle
mv /data/local/tmp/genie_bundle/artifact /data/local/tmp/genie_bundle/qairt
export QAIRT_HOME={qairt_path}
export PATH={qairt_path}/bin/aarch64-android:${{PATH}}
export LD_LIBRARY_PATH={qairt_path}/lib/aarch64-android
export ADSP_LIBRARY_PATH={qairt_path}/lib/hexagon-<<HEXAGON_VERSION>>/unsigned
cp /data/local/tmp/qxa.qa_adsplib/libc++.so.1 ${{ADSP_LIBRARY_PATH}}/
cp /data/local/tmp/qxa.qa_adsplib/libc++abi.so.1 ${{ADSP_LIBRARY_PATH}}/
genie-t2t-run -c genie_config.json --prompt_file sample_prompt.txt > /data/local/tmp/QDC_logs/genie.log
{full_genie_command}
"""
        # Push the genie_bundle directory to the device
        subprocess.run(
            ["adb", "push", "/qdc/appium/genie_bundle/", "/data/local/tmp"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Run the shell script on the device
        subprocess.run(
            [
                "adb",
                "shell",
                "sh",
                "-c",
                genie_script,
            ],
            capture_output=True,
            text=True,
            check=True,
        )


if __name__ == "__main__":
    # Invoke Pytest on this file
    sys.exit(pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)]))

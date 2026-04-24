# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Llama.CPP benchmark script for Android devices via Appium.

This script runs llama-completion benchmarks on ALL compute units (CPU, GPU, HTP)
for multiple context lengths (128, 1024, 4096).

For each compute unit and context length, it runs one benchmark that measures:
  - Prompt TPS (tokens per second for prompt processing) - used to calculate TTFT
  - Generation TPS (tokens per second for generation)

Placeholders are replaced at artifact creation time:
  - <<MODEL_URL>>: URL to download the model
  - <<HTP_DEVICES>>: Number of HTP devices to use (1, 2, 4, or 5)
  - <<HTP_DEVICE_LIST>>: Comma-separated list of HTP devices (e.g., "HTP0,HTP1")
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

# Context lengths to benchmark
CONTEXT_LENGTHS = [128, 1024, 4096]

# Common llama-cli arguments
SYSTEM_PROMPT = "You are a helpful assistant. Be helpful but brief."
SEED = 1


class TestLlamaCpp:
    @pytest.fixture
    def driver(self) -> webdriver.Remote:
        return webdriver.Remote(
            command_executor="http://127.0.0.1:4723/wd/hub", options=options
        )

    def test_llama_cpp_all_devices(self, driver: webdriver.Remote) -> None:
        """Run Llama.CPP benchmark on all compute units (CPU, GPU, HTP) for all context lengths."""
        # Configuration from placeholders (replaced at artifact creation)
        model_url = "<<MODEL_URL>>"
        htp_devices = "<<HTP_DEVICES>>"
        htp_device_list = "<<HTP_DEVICE_LIST>>"

        # Commands to run on device
        llama_cpp_script = f"""
cd /data/local/tmp/llama_cpp_bundle

# Set library path for shared libraries
export LD_LIBRARY_PATH=/data/local/tmp/llama_cpp_bundle/lib:$LD_LIBRARY_PATH

# Set ADSP library path for Hexagon DSP to find HTP libraries
export ADSP_LIBRARY_PATH="/data/local/tmp/llama_cpp_bundle/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"

# Make binaries executable (permissions lost during adb push)
chmod +x /data/local/tmp/llama_cpp_bundle/bin/*

# Download model from URL (only once)
echo "Downloading model..."
curl -L -J --output /data/local/tmp/model.gguf "{model_url}"

# Initialize log file
LOG_FILE=/data/local/tmp/QDC_logs/llama_cpp.log
echo "============================================================" > $LOG_FILE
echo "LLAMA.CPP COMPREHENSIVE BENCHMARK" >> $LOG_FILE
echo "Context lengths: 128, 1024, 4096" >> $LOG_FILE
echo "Compute units: CPU, GPU, HTP" >> $LOG_FILE
echo "HTP devices: {htp_devices} ({htp_device_list})" >> $LOG_FILE
echo "============================================================" >> $LOG_FILE

LLAMA_COMPLETION=/data/local/tmp/llama_cpp_bundle/bin/llama-completion
MODEL=/data/local/tmp/model.gguf
SYSTEM_PROMPT="{SYSTEM_PROMPT}"
HTP_DEVICES="{htp_devices}"
HTP_DEVICE_LIST="{htp_device_list}"

# Function to run a single benchmark and log the command
run_benchmark() {{
    local COMPUTE=$1
    local CTX_LEN=$2
    local PROMPT_FILE=$3

    echo "" >> $LOG_FILE
    echo "=== $COMPUTE | CTX=$CTX_LEN ===" >> $LOG_FILE

    if [ "$COMPUTE" = "CPU" ]; then
        local CMD="GGML_HEXAGON_NDEV=0 $LLAMA_COMPLETION --model $MODEL --n-predict -1 --ctx-size $CTX_LEN --system-prompt \\"$SYSTEM_PROMPT\\" --file /data/local/tmp/llama_cpp_bundle/$PROMPT_FILE --seed {SEED} --single-turn --no-display-prompt --n-gpu-layers 0"
        echo "COMMAND: $CMD" >> $LOG_FILE
        GGML_HEXAGON_NDEV=0 $LLAMA_COMPLETION \\
            --model $MODEL \\
            --n-predict -1 \\
            --ctx-size $CTX_LEN \\
            --system-prompt "$SYSTEM_PROMPT" \\
            --file "/data/local/tmp/llama_cpp_bundle/$PROMPT_FILE" \\
            --seed {SEED} \\
            --single-turn \\
            --no-display-prompt \\
            --n-gpu-layers 0 \\
            2>&1 | tee -a $LOG_FILE
    elif [ "$COMPUTE" = "GPU" ]; then
        local CMD="GGML_HEXAGON_NDEV=0 $LLAMA_COMPLETION --model $MODEL --n-predict -1 --ctx-size $CTX_LEN --system-prompt \\"$SYSTEM_PROMPT\\" --file /data/local/tmp/llama_cpp_bundle/$PROMPT_FILE --seed {SEED} --single-turn --no-display-prompt -fa off"
        echo "COMMAND: $CMD" >> $LOG_FILE
        GGML_HEXAGON_NDEV=0 $LLAMA_COMPLETION \\
            --model $MODEL \\
            --n-predict -1 \\
            --ctx-size $CTX_LEN \\
            --system-prompt "$SYSTEM_PROMPT" \\
            --file "/data/local/tmp/llama_cpp_bundle/$PROMPT_FILE" \\
            --seed {SEED} \\
            --single-turn \\
            --no-display-prompt \\
            -fa off \\
            2>&1 | tee -a $LOG_FILE
    elif [ "$COMPUTE" = "HTP" ]; then
        # HTP with multi-device support
        local CMD="GGML_HEXAGON_NDEV=$HTP_DEVICES $LLAMA_COMPLETION --model $MODEL --n-predict -1 --ctx-size $CTX_LEN --system-prompt \\"$SYSTEM_PROMPT\\" --file /data/local/tmp/llama_cpp_bundle/$PROMPT_FILE --seed {SEED} --single-turn --no-display-prompt --no-mmap -t 6 --cpu-mask 0xfc --cpu-strict 1 -ctk f16 -ctv f16 -fa on --batch-size 128 --device \\"$HTP_DEVICE_LIST\\""
        echo "COMMAND: $CMD" >> $LOG_FILE
        GGML_HEXAGON_NDEV=$HTP_DEVICES $LLAMA_COMPLETION \\
            --model $MODEL \\
            --n-predict -1 \\
            --ctx-size $CTX_LEN \\
            --system-prompt "$SYSTEM_PROMPT" \\
            --file "/data/local/tmp/llama_cpp_bundle/$PROMPT_FILE" \\
            --seed {SEED} \\
            --single-turn \\
            --no-display-prompt \\
            --no-mmap \\
            -t 6 \\
            --cpu-mask 0xfc \\
            --cpu-strict 1 \\
            -ctk f16 \\
            -ctv f16 \\
            -fa on \\
            --batch-size 128 \\
            --device "$HTP_DEVICE_LIST" \\
            2>&1 | tee -a $LOG_FILE
    fi
}}

# Run benchmarks for each compute unit and context length
for COMPUTE in CPU GPU HTP; do
    echo "" >> $LOG_FILE
    echo "########################################################" >> $LOG_FILE
    echo "# BENCHMARKING: $COMPUTE" >> $LOG_FILE
    echo "########################################################" >> $LOG_FILE

    # Run one benchmark per context length (extracts both TTFT and TPS from single run)
    for CTX_LEN in 128 1024 4096; do
        echo "" >> $LOG_FILE
        echo "--- Context Length: $CTX_LEN ---" >> $LOG_FILE

        run_benchmark "$COMPUTE" "$CTX_LEN" "sample_prompt_${{CTX_LEN}}.txt"
    done
done

echo "" >> $LOG_FILE
echo "============================================================" >> $LOG_FILE
echo "=== BENCHMARK COMPLETE ===" >> $LOG_FILE
echo "============================================================" >> $LOG_FILE
"""

        # Push the llama_cpp_bundle directory to the device
        subprocess.run(
            ["adb", "push", "/qdc/appium/llama_cpp_bundle/", "/data/local/tmp"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )

        # Run the shell script on the device
        result = subprocess.run(
            [
                "adb",
                "shell",
                "sh",
                "-c",
                llama_cpp_script,
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        print(result.stdout)
        print(result.stderr)


if __name__ == "__main__":
    # Invoke Pytest on this file
    sys.exit(pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)]))

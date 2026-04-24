# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import os

import adbutils

GENIE_CONFIG_FILE_REL_PATH = "./genie_config.json"

"""
Generating TinyMMLU

1. Generate model bundles for model of interest
    a. Option A: Use LLM on Genie tutorial https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie
    b. Option B: You already have QNN context binary, tokenizer and QNN SDK in place
2. Use this script to run model on-device using genie-t2t-run and collect data
    a. Example usage:
        python compute_mmlu_on_device.py --bundle-path <bundle_path_from_point_1> --questions /path/to/directory/of/questions/
    b. You can provide `--adb_port <port>` if want to collect data on QDC device with different port than default i.e. 5037
    c. Provide fix prompt to run this multiple iterations
    d. Set `--keep-on-device-bundle` to keep data on-device. This is useful if you plan to use genie-bundle on device for further use.
3. This outputs MMLU value.
"""


def _push_bundle_to_device(
    device: "adbutils._device.AdbDevice", bundle_path: str, destination_bundle_path: str
) -> None:
    for _, dir_names, file_names in os.walk(bundle_path):
        if len(dir_names) > 0:
            raise RuntimeError(
                "Invalid source bundle directory. Expecting no sub-directory in bundle."
            )

        for file_name in file_names:
            full_file_path = str(os.path.join(bundle_path, file_name))
            print(f"Copying {file_name} to device.")
            device.sync.push(full_file_path, destination_bundle_path)


def _run_model_on_android_device(
    device: "adbutils._device.AdbDevice",
    working_dir: str,
    question_file: str,
    answers: list[int],
) -> bool:
    """Runs models on device from working directory"""
    idx = int(question_file.split("_")[1].split(".")[0])
    with open(question_file) as file:
        question = file.read()
    question = question.replace('"', r"\"")
    question = question.replace("'", r"\'")

    device_commands = [
        f"cd {working_dir}",
        f"export LD_LIBRARY_PATH={working_dir}",
        f"export ADSP_LIBRARY_PATH={working_dir}",
        f'./genie-t2t-run -c {GENIE_CONFIG_FILE_REL_PATH} -p "{question}"',
    ]
    command = ";".join(device_commands)

    model_output = device.shell(command)

    answer = model_output.split("[BEGIN]:")[-1].strip()[0]
    expected_answer = chr(answers[idx] + ord("A"))
    return expected_answer == answer


def main() -> None:
    parser = argparse.ArgumentParser(
        "Generate LLM metrics by running model on Android devices using Genie."
    )
    parser.add_argument(
        "--bundle-path",
        type=str,
        default="",
        help="Local path to genie-bundle generated as per https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie",
    )
    parser.add_argument(
        "--device-bundle-path",
        type=str,
        default="",
        help="If provided, uses bundle path from device and skips copying data from --bundle-path",
    )
    parser.add_argument(
        "--keep-on-device-bundle",
        action="store_true",
        help="If set, keeps copied on-device bundled as is. Otherwise, cleans up directory after profiling.",
    )
    parser.add_argument(
        "--adb-port",
        type=int,
        default=5037,
        help="Port to connect adb devices on. Usually 5037, but if using QDC devices, match this with device port you set ssh-tunneling to.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, dumps genie model inference output for each inference. Otherwise, minimal output of stdout and final metrics.",
    )
    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Directory of questions to compute MMLU on.",
    )
    parser.add_argument(
        "--answers-file", type=str, required=True, help="File path with answers."
    )
    parser.add_argument("--host", type=str, help="Host that is connected to device.")
    parser.add_argument(
        "--serial", type=str, help="Serial number of device to connect to."
    )

    args = parser.parse_args()

    bundle_path = args.bundle_path
    device_bundle_path = args.device_bundle_path
    adb_port = args.adb_port
    host = args.host
    serial = args.serial
    questions_directory = args.questions
    keep_on_device_bundle = args.keep_on_device_bundle
    answers_filepath = args.answers_file
    if bundle_path and device_bundle_path:
        raise RuntimeError(
            "Only one of --bundle-path and --device-bundle-path should be set."
        )

    if not os.path.exists(questions_directory):
        raise ValueError("The questions directory is not a valid path.")

    if not os.path.exists(answers_filepath):
        raise ValueError("The answers file is not a valid path.")

    # Connect to device using ADB client
    adb = adbutils.AdbClient(port=adb_port, host=host)
    device = adb.device(serial=serial)
    android_os_version = device.prop.get("ro.system.build.version.release")
    device_os = f"Android OS {android_os_version}"
    device_soc = device.prop.get("ro.soc.model")

    with open(answers_filepath) as file:
        answers = [int(ans) for ans in file.read().strip().split("\n")]

    if bundle_path:
        if not os.path.exists(bundle_path):
            raise RuntimeError(f"Provided --bundle-path {bundle_path} does not exists.")
    elif device_bundle_path:
        assert device is not None
        if device.shell(f"[ -e {device_bundle_path} ]; echo $?") != "0":
            raise RuntimeError(
                f"Provided --device-bundle-path {device_bundle_path} does not exists on device."
            )
    else:
        raise RuntimeError("One of --bundle-path and --device-bundle-path must be set.")

    if device_bundle_path:
        print("\nUsing bundle from device for profiling.")
        working_dir = device_bundle_path
    else:
        assert device is not None
        working_dir = device.shell("mktemp -d")
        _push_bundle_to_device(device, bundle_path, working_dir)
        print(f"\nBundled directory copied onto device at {working_dir}")

    print(
        f"\nCollecting performance numbers using Genie CLI on {device_soc} with {device_os}..."
    )

    try:
        assert device is not None
        correct = 0
        for question_file in os.listdir(questions_directory):
            corr = _run_model_on_android_device(
                device,
                working_dir,
                os.path.join(questions_directory, question_file),
                answers,
            )
            correct += int(corr)
        print(f"TinyMMLU: {correct / 100}")

    finally:
        if keep_on_device_bundle or device_bundle_path:
            print(f"Bundle directory from device {working_dir} not removed.")
        else:
            print(f"Removing bundle directory from device {working_dir}.")
            assert device is not None
            device.shell(f"rm -rf {working_dir}")


if __name__ == "__main__":
    main()

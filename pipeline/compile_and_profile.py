import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qai_hub
import onnx
import os
import argparse
import json


def run_profile(model, name, device) -> str:
    """Submit a profile job for the model."""
    profile_job = qai_hub.submit_profile_job(
        model=model, name=name, device=device, options="--max_profiler_iterations 100"
    )
    return profile_job.job_id


def compile_model(
    model,
    name,
    device,
    input_specs,
    *,
    force_channel_last_input_name: str | None = None,
) -> str:
    """Submits a compile job for the model and returns the job instance."""
    options = "--target_runtime qnn_dlc --truncate_64bit_io"
    if force_channel_last_input_name:
        options += f" --force_channel_last_input {force_channel_last_input_name}"

    compile_job = qai_hub.submit_compile_job(
        model=model,
        name=name,
        device=device,
        input_specs=input_specs,
        options=options,
    )
    compile_job.modify_sharing(add_emails=["lowpowervision@gmail.com"])
    print(f"Job {compile_job.job_id} shared with lowpowervision@gmail.com")
    return compile_job.job_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--ids-file", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    postfix = args.postfix

    # --- Configuration ---
    ONNX_DIR = f"exported_{model_name}_onnx"
    # ---------------------

    # Construct the full paths
    IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, f"image_encoder{postfix}.onnx")
    TEXT_ONNX_PATH = os.path.join(ONNX_DIR, f"text_encoder{postfix}.onnx")

    if not os.path.exists(ONNX_DIR):
        print(
            f"Error: Directory '{ONNX_DIR}' not found. Please run 'export_onnx.py' first."
        )
        sys.exit(1)

    # Load the ONNX models from the new location
    print(f"Loading ONNX Image Encoder from {IMAGE_ONNX_PATH}...")
    onnx_img_model = onnx.load(IMAGE_ONNX_PATH)

    # Check the model for errors
    try:
        onnx.checker.check_model(onnx_img_model)
        print("Image ONNX model is valid ✅")
    except onnx.checker.ValidationError as e:
        print("Image ONNX model validation failed ❌")
        print(e)

    print(f"\nLoading ONNX Text Encoder from {TEXT_ONNX_PATH}...")
    onnx_txt_model = onnx.load(TEXT_ONNX_PATH)

    # Check the model for errors
    try:
        onnx.checker.check_model(onnx_txt_model)
        print("Text ONNX model is valid ✅")
    except onnx.checker.ValidationError as e:
        print("Text ONNX model validation failed ❌")
        print(e)

    target_device = qai_hub.Device("XR2 Gen 2 (Proxy)")

    # Submit compilation jobs in parallel
    print("\nSubmitting compilation jobs to QAI Hub...")
    # Only enable force_channel_last for quantized image variants.
    image_force_channel_last = "image" if postfix else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        img_compile_future = executor.submit(
            compile_model,
            onnx_img_model,
            model_name + f"_image_encoder{postfix}",
            target_device,
            {"image": (1, 3, 224, 224)},
            force_channel_last_input_name=image_force_channel_last,
        )
        txt_compile_future = executor.submit(
            compile_model,
            onnx_txt_model,
            model_name + f"_text_encoder{postfix}",
            target_device,
            {"text": ((1, 77), "int64")},
        )
        img_id = img_compile_future.result()
        txt_id = txt_compile_future.result()

    print(f"Image compilation job ID: {img_id}")
    print(f"Text compilation job ID: {txt_id}")

    # Wait for both compile jobs to finish, then submit profile jobs in parallel
    print("\nWaiting for compilation and submitting profiling jobs to QAI Hub...")

    def _wait_and_profile(job_id: str, name: str) -> str:
        target_model = qai_hub.get_job(job_id).get_target_model()
        profile_job_id = run_profile(model=target_model, name=name, device=target_device)
        print(f"Profile job submitted for {name}: {profile_job_id}")
        return profile_job_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        img_profile_future = executor.submit(
            _wait_and_profile, img_id, model_name + f"_image_encoder{postfix}"
        )
        txt_profile_future = executor.submit(
            _wait_and_profile, txt_id, model_name + f"_text_encoder{postfix}"
        )
        img_profile_id = img_profile_future.result()
        txt_profile_id = txt_profile_future.result()

    print("Profiling jobs submitted for both models.")

    if args.ids_file:
        payload = {
            "model_name": model_name,
            "postfix": postfix,
            "image_compile_id": img_id,
            "text_compile_id": txt_id,
            "image_profile_id": img_profile_id,
            "text_profile_id": txt_profile_id,
        }
        with open(args.ids_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved job ids to: {args.ids_file}")


if __name__ == "__main__":
    main()

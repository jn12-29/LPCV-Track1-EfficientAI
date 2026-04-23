"""Compile and profile MobileCLIP2 encoders on QAI Hub.

Supports two input modes:
  A) Local FP32 ONNX files (original flow, uses --float_bitwidth 16)
  B) QAI Hub quantized model IDs (from ptq/quantize.py, no --float_bitwidth)

Mode B is preferred for INT8 deployment — avoids the ORT PTQ → qairt-converter
incompatibilities (MatMul→Gemm rewrite, QDQ fusion failure, bias format, etc.).
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qai_hub
import onnx
import os
import argparse
import json

from onnx_utils import sanitize_value_info_clashing_with_io


def run_profile(model, name, device) -> str:
    """Submit a profile job for the model."""
    profile_job = qai_hub.submit_profile_job(
        model=model, name=name, device=device, options="--max_profiler_iterations 100"
    )
    return profile_job.job_id


def compile_model(model, name, device, input_specs, quantized: bool = False) -> str:
    """Submits a compile job for the model and returns the job ID.

    When quantized=True the model is already INT8 from QAI Hub's quantizer,
    so --float_bitwidth is omitted (would conflict with existing quant params).
    """
    options = "--target_runtime qnn_dlc --truncate_64bit_io"
    if not quantized:
        options += " --float_bitwidth 16"

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
    parser = argparse.ArgumentParser(
        description="Compile & profile MobileCLIP2 on QAI Hub"
    )
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--ids-file", type=str, default=None)

    # Mode B: QAI Hub quantized model IDs (from ptq/quantize.py --ids-file)
    parser.add_argument(
        "--quantized-image-model-id", type=str, default=None,
        help="QAI Hub model ID for quantized image encoder (skips local ONNX loading)"
    )
    parser.add_argument(
        "--quantized-text-model-id", type=str, default=None,
        help="QAI Hub model ID for quantized text encoder (skips local ONNX loading)"
    )
    parser.add_argument(
        "--quantize-ids-file", type=str, default=None,
        help="JSON file from ptq/quantize.py --ids-file (reads model IDs automatically)"
    )
    return parser.parse_args()


def _load_local_onnx(onnx_path: str, label: str):
    """Load and validate a local ONNX file."""
    print(f"Loading ONNX {label} from {onnx_path}...")
    model = sanitize_value_info_clashing_with_io(onnx.load(onnx_path))
    try:
        onnx.checker.check_model(model)
        print(f"{label} ONNX model is valid ✅")
    except onnx.checker.ValidationError as e:
        print(f"{label} ONNX model validation failed ❌")
        print(e)
    return model


def _load_hub_model(model_id: str, label: str):
    """Fetch a model from QAI Hub by ID."""
    print(f"Loading {label} from QAI Hub model ID: {model_id}")
    model = qai_hub.get_model(model_id)
    print(f"  {label} loaded: {model.name or model_id}")
    return model


def main():
    args = parse_args()
    model_name = args.model_name
    postfix = args.postfix

    # Resolve quantized model IDs from --quantize-ids-file if provided
    q_img_id = args.quantized_image_model_id
    q_txt_id = args.quantized_text_model_id
    if args.quantize_ids_file:
        with open(args.quantize_ids_file, "r") as f:
            q_ids = json.load(f)
        q_img_id = q_img_id or q_ids.get("image_quantized_model_id")
        q_txt_id = q_txt_id or q_ids.get("text_quantized_model_id")

    is_quantized = bool(q_img_id or q_txt_id)

    # --- Load models ---
    if q_img_id:
        onnx_img_model = _load_hub_model(q_img_id, "Image Encoder (quantized)")
    else:
        ONNX_DIR = f"exported_{model_name}_onnx"
        IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, f"image_encoder{postfix}.onnx")
        if not os.path.exists(ONNX_DIR):
            print(f"Error: Directory '{ONNX_DIR}' not found. Run 'export_onnx.py' first.")
            sys.exit(1)
        onnx_img_model = _load_local_onnx(IMAGE_ONNX_PATH, "Image Encoder")

    if q_txt_id:
        onnx_txt_model = _load_hub_model(q_txt_id, "Text Encoder (quantized)")
    else:
        ONNX_DIR = f"exported_{model_name}_onnx"
        TEXT_ONNX_PATH = os.path.join(ONNX_DIR, f"text_encoder{postfix}.onnx")
        onnx_txt_model = _load_local_onnx(TEXT_ONNX_PATH, "Text Encoder")

    target_device = qai_hub.Device("XR2 Gen 2 (Proxy)")

    # Determine quantized status per encoder
    img_quantized = bool(q_img_id)
    txt_quantized = bool(q_txt_id)

    # Build name suffixes
    img_suffix = postfix if not img_quantized else f"{postfix}_int8"
    txt_suffix = postfix if not txt_quantized else f"{postfix}_int8"

    # Submit compilation jobs in parallel
    mode_label = "quantized (INT8)" if is_quantized else "FP32 (→FP16 on-device)"
    print(f"\nSubmitting compilation jobs to QAI Hub ({mode_label})...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        img_compile_future = executor.submit(
            compile_model,
            onnx_img_model,
            model_name + f"_image_encoder{img_suffix}",
            target_device,
            {"image": (1, 3, 224, 224)},
            img_quantized,
        )
        txt_compile_future = executor.submit(
            compile_model,
            onnx_txt_model,
            model_name + f"_text_encoder{txt_suffix}",
            target_device,
            {"text": ((1, 77), "int64")},
            txt_quantized,
        )
        img_id = img_compile_future.result()
        txt_id = txt_compile_future.result()

    print(f"Image compilation job ID: {img_id}")
    print(f"Text compilation job ID: {txt_id}")

    # Wait for both compile jobs to finish, then submit profile jobs in parallel
    print("\nWaiting for compilation and submitting profiling jobs to QAI Hub...")

    def _wait_and_profile(job_id: str, name: str) -> str:
        target_model = qai_hub.get_job(job_id).get_target_model()
        if target_model is None:
            raise RuntimeError(
                f"Compile failed (job {job_id}); cannot submit profile job. "
                "Please inspect compile logs on QAI Hub."
            )
        profile_job_id = run_profile(model=target_model, name=name, device=target_device)
        print(f"Profile job submitted for {name}: {profile_job_id}")
        return profile_job_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        img_profile_future = executor.submit(
            _wait_and_profile, img_id, model_name + f"_image_encoder{img_suffix}"
        )
        txt_profile_future = executor.submit(
            _wait_and_profile, txt_id, model_name + f"_text_encoder{txt_suffix}"
        )
        img_profile_id = img_profile_future.result()
        txt_profile_id = txt_profile_future.result()

    print("Profiling jobs submitted for both models.")

    if args.ids_file:
        payload = {
            "model_name": model_name,
            "postfix": postfix,
            "quantized": is_quantized,
            "image_compile_id": img_id,
            "text_compile_id": txt_id,
            "image_profile_id": img_profile_id,
            "text_profile_id": txt_profile_id,
        }
        if q_img_id:
            payload["image_quantized_model_id"] = q_img_id
        if q_txt_id:
            payload["text_quantized_model_id"] = q_txt_id
        with open(args.ids_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved job ids to: {args.ids_file}")


if __name__ == "__main__":
    main()

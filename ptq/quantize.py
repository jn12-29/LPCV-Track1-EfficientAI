"""QAI Hub native Post-Training Quantization for MobileCLIP2 ONNX encoders.

Bypasses ORT quantize_static entirely — the four fatal incompatibilities with
qairt-converter (MatMul→Gemm rewrite, QDQ fusion failure, bias format mismatch,
per-channel zero_point corruption) are all eliminated by using Qualcomm's own
quantization pipeline.

Flow:
  1. Upload FP32 ONNX to QAI Hub
  2. submit_quantize_job with real calibration data → INT8 model (hub-side)
  3. Optionally compile + profile in one shot

The quantized model lives on QAI Hub; use its model ID in compile_and_profile.py
or pass --compile here to do everything in one go.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import qai_hub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptq.dataset import (
    ImageCalibReader,
    TextCalibReader,
    load_jsonl_records,
    split_calib_val,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QAI Hub native PTQ for MobileCLIP2 ONNX encoders"
    )
    p.add_argument("--onnx-dir", type=str, required=True,
                    help="Directory containing image_encoder.onnx and text_encoder.onnx")
    p.add_argument("--output-suffix", type=str, default="_ptq_qdq_u8s8_pct",
                    help="Suffix for locally saved quantized ONNX files")
    p.add_argument("--jsonl-path", type=str, required=True,
                    help="Contrastive JSONL dataset for calibration")
    p.add_argument("--image-base-dir", type=str, required=True,
                    help="Root directory for resolving image_path in JSONL")
    p.add_argument("--calib-size", type=int, default=500,
                    help="Number of calibration samples (default: 500)")
    p.add_argument("--val-size", type=int, default=1,
                    help="Validation split size used only for split compatibility")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--quantize-image", action="store_true", default=True)
    p.add_argument("--no-quantize-image", dest="quantize_image", action="store_false")
    p.add_argument("--quantize-text", action="store_true", default=False)
    p.add_argument("--no-quantize-text", dest="quantize_text", action="store_false")

    p.add_argument("--weights-dtype", type=str, default="int8",
                    choices=["int8", "int4"],
                    help="Weight quantization dtype (default: int8)")
    p.add_argument("--activations-dtype", type=str, default="int8",
                    choices=["int8", "int16"],
                    help="Activation quantization dtype (default: int8)")
    p.add_argument("--calib-method", type=str, default="percentile",
                    help="Compatibility arg; QAI Hub uses backend-native PTQ method.")
    p.add_argument("--quant-format", type=str, default="qdq",
                    help="Compatibility arg; ignored in QAI Hub native quantization.")
    p.add_argument("--activation-type", type=str, default="quint8",
                    help="Compatibility arg; mapped by --activations-dtype instead.")
    p.add_argument("--weight-type", type=str, default="qint8",
                    help="Compatibility arg; mapped by --weights-dtype instead.")
    p.add_argument("--op-types", type=str, default="",
                    help="Compatibility arg; QAI Hub quantization is op-agnostic.")

    p.add_argument("--compile", action="store_true", default=False,
                    help="Also compile + profile after quantization")
    p.add_argument("--model-name", type=str, default="MobileCLIP2-B",
                    help="Model name prefix for QAI Hub job naming")
    p.add_argument("--ids-file", type=str, default=None,
                    help="Path to save job/model IDs as JSON")
    return p.parse_args()


DTYPE_MAP = {
    "int8": qai_hub.QuantizeDtype.INT8,
    "int4": qai_hub.QuantizeDtype.INT4,
    "int16": qai_hub.QuantizeDtype.INT16,
}


def build_image_calib_data(
    records: list[dict], max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Build calibration data dict for image encoder from JSONL records."""
    reader = ImageCalibReader(records)
    batches = []
    count = 0
    while True:
        batch = reader.get_next()
        if batch is None:
            break
        batches.append(batch["image"])  # (1, 3, H, W)
        count += 1
        if max_samples and count >= max_samples:
            break
    data = np.concatenate(batches, axis=0)  # (N, 3, H, W)
    print(f"  Image calibration data: {data.shape}, dtype={data.dtype}")
    return {"image": data}


def build_text_calib_data(
    records: list[dict], tokenizer, max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Build calibration data dict for text encoder from JSONL records."""
    reader = TextCalibReader(records, tokenizer)
    batches = []
    count = 0
    while True:
        batch = reader.get_next()
        if batch is None:
            break
        batches.append(batch["text"])  # (1, 77)
        count += 1
        if max_samples and count >= max_samples:
            break
    data = np.concatenate(batches, axis=0)  # (N, 77)
    print(f"  Text calibration data: {data.shape}, dtype={data.dtype}")
    return {"text": data}


def quantize_on_hub(
    onnx_path: str,
    calib_data: dict[str, np.ndarray],
    weights_dtype: qai_hub.QuantizeDtype,
    activations_dtype: qai_hub.QuantizeDtype,
    name: str,
) -> tuple[str, object]:
    """Upload FP32 ONNX, run QAI Hub quantization, return (job_id, quantized_model)."""
    print(f"  Uploading {onnx_path} to QAI Hub...")
    fp32_model = qai_hub.upload_model(onnx_path)
    print(f"  Uploaded model ID: {fp32_model.model_id}")

    print(f"  Submitting quantize job: weights={weights_dtype}, activations={activations_dtype}")
    quantize_job = qai_hub.submit_quantize_job(
        model=fp32_model,
        calibration_data=calib_data,
        weights_dtype=weights_dtype,
        activations_dtype=activations_dtype,
        name=f"{name}_quantize",
    )
    print(f"  Quantize job ID: {quantize_job.job_id}  (waiting...)")
    quantize_job.wait()

    quantized_model = quantize_job.get_target_model()
    if quantized_model is None:
        raise RuntimeError(
            f"Quantize job {quantize_job.job_id} failed. "
            "Check QAI Hub dashboard for details."
        )
    print(f"  Quantized model ID: {quantized_model.model_id}")
    return quantize_job.job_id, quantized_model


def _copy_if_needed(src_path: str, dst_path: str) -> None:
    if os.path.abspath(src_path) != os.path.abspath(dst_path):
        shutil.copy2(src_path, dst_path)


def compile_quantized(
    quantized_model,
    name: str,
    device,
    input_specs: dict,
) -> str:
    """Compile a QAI Hub quantized model (no --float_bitwidth, already INT8)."""
    compile_job = qai_hub.submit_compile_job(
        model=quantized_model,
        name=name,
        device=device,
        input_specs=input_specs,
        options="--target_runtime qnn_dlc --truncate_64bit_io",
    )
    compile_job.modify_sharing(add_emails=["lowpowervision@gmail.com"])
    print(f"  Compile job {compile_job.job_id} shared with lowpowervision@gmail.com")
    return compile_job.job_id


def main() -> None:
    args = parse_args()
    onnx_dir = args.onnx_dir

    image_src = os.path.join(onnx_dir, "image_encoder.onnx")
    text_src = os.path.join(onnx_dir, "text_encoder.onnx")
    image_dst = os.path.join(onnx_dir, f"image_encoder{args.output_suffix}.onnx")
    text_dst = os.path.join(onnx_dir, f"text_encoder{args.output_suffix}.onnx")

    w_dtype = DTYPE_MAP[args.weights_dtype]
    a_dtype = DTYPE_MAP[args.activations_dtype]

    print(f"QAI Hub PTQ: weights={args.weights_dtype}, activations={args.activations_dtype}")
    print(f"  Calibration size: {args.calib_size}, seed: {args.seed}")

    # Load calibration data
    records = load_jsonl_records(args.jsonl_path, args.image_base_dir)
    calib_records, _ = split_calib_val(
        records, calib_size=args.calib_size, val_size=args.val_size, seed=args.seed,
    )
    print(f"Loaded {len(calib_records)} calibration records")

    results = {}
    target_device = qai_hub.Device("XR2 Gen 2 (Proxy)")

    # --- Image encoder ---
    if args.quantize_image:
        if not os.path.exists(image_src):
            raise FileNotFoundError(f"Missing {image_src}")
        print("\n=== Image Encoder ===")
        img_calib = build_image_calib_data(calib_records)
        img_q_job_id, img_q_model = quantize_on_hub(
            image_src, img_calib, w_dtype, a_dtype,
            name=f"{args.model_name}_image_encoder",
        )
        results["image_quantize_job_id"] = img_q_job_id
        results["image_quantized_model_id"] = img_q_model.model_id
        img_q_model.download(image_dst)
        results["image_quantized_onnx_path"] = image_dst

        if args.compile:
            print("  Compiling quantized image encoder...")
            img_c_id = compile_quantized(
                img_q_model,
                f"{args.model_name}_image_encoder_int8",
                target_device,
                {"image": (1, 3, 224, 224)},
            )
            results["image_compile_job_id"] = img_c_id
    else:
        _copy_if_needed(image_src, image_dst)
        results["image_quantized_onnx_path"] = image_dst

    # --- Text encoder ---
    if args.quantize_text:
        if not os.path.exists(text_src):
            raise FileNotFoundError(f"Missing {text_src}")
        print("\n=== Text Encoder ===")
        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        txt_calib = build_text_calib_data(calib_records, tokenizer)
        txt_q_job_id, txt_q_model = quantize_on_hub(
            text_src, txt_calib, w_dtype, a_dtype,
            name=f"{args.model_name}_text_encoder",
        )
        results["text_quantize_job_id"] = txt_q_job_id
        results["text_quantized_model_id"] = txt_q_model.model_id
        txt_q_model.download(text_dst)
        results["text_quantized_onnx_path"] = text_dst

        if args.compile:
            print("  Compiling quantized text encoder...")
            txt_c_id = compile_quantized(
                txt_q_model,
                f"{args.model_name}_text_encoder_int8",
                target_device,
                {"text": ((1, 77), "int64")},
            )
            results["text_compile_job_id"] = txt_c_id
    else:
        _copy_if_needed(text_src, text_dst)
        results["text_quantized_onnx_path"] = text_dst

    # Save results
    if args.ids_file:
        results["model_name"] = args.model_name
        results["weights_dtype"] = args.weights_dtype
        results["activations_dtype"] = args.activations_dtype
        results["calib_size"] = args.calib_size
        with open(args.ids_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved IDs to: {args.ids_file}")

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("\nQAI Hub PTQ complete.")


if __name__ == "__main__":
    main()

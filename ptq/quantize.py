"""Post-Training Quantization for MobileCLIP2 ONNX encoders.

Condensed from the LPCVTrack1 mobileclipv2_quant.py reference, tailored for
MobileCLIP2-B (ViT-based vision encoder).

Default configuration:
  - Format:      QDQ (QuantizeLinear + DequantizeLinear node pairs)
  - Activation:  QUInt8 (asymmetric, HTP preferred)
  - Weight:      QInt8 (symmetric, per-channel)
  - Calibration: Percentile (99.9%, robust against attention logit outliers)

Quantized ops (INT8):
  - MatMul (72 nodes): MHSA projections + MLP FCs, main acceleration target
  - Conv   (3 nodes):  Patch embedding backbone, memory bandwidth bottleneck
  - Gemm   (1 node):   Final projection head

FP32 islands (kept unquantized):
  - GELU (14 Erf nodes): No native INT8 on QNN HTP, would fall back to CPU
  - Softmax (12 nodes):  Attention distribution sensitive to quantization noise
  - LayerNorm (25 nodes): QNN HTP has native FP16/FP32 LN support

QNN EP will auto-partition: INT8 MatMul/Conv/Gemm -> HTP HMX units,
FP32 GELU/Softmax/LN -> CPU/HVX fallback.

This script supports three PTQ schemes:
  1) int8_ort      : local ONNX Runtime static PTQ (default; existing flow)
  2) w8a16_vit_hub : Qualcomm AI Hub W8A16 PTQ via direct quantize job
  3) w8a8_vit_hub  : Qualcomm AI Hub W8A8 PTQ via direct quantize job
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import open_clip
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
    shape_inference,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptq.dataset import (
    ImageCalibReader,
    TextCalibReader,
    load_jsonl_records,
    split_calib_val,
)
from onnx_utils import sanitize_value_info_clashing_with_io
from utils.preprocess import preprocess_image

CALIB_METHODS: Dict[str, CalibrationMethod] = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
    "distribution": CalibrationMethod.Distribution,
}

QUANT_FORMATS: Dict[str, QuantFormat] = {
    "qdq": QuantFormat.QDQ,
    "qoperator": QuantFormat.QOperator,
}

QUANT_TYPES: Dict[str, QuantType] = {
    "qint8": QuantType.QInt8,
    "quint8": QuantType.QUInt8,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTQ for MobileCLIP2 ONNX encoders")
    p.add_argument(
        "--ptq-scheme",
        type=str,
        default="int8_ort",
        choices=["int8_ort", "w8a16_vit_hub", "w8a8_vit_hub"],
        help=(
            "Quantization backend. int8_ort uses local ORT PTQ; "
            "w8a16_vit_hub/w8a8_vit_hub use AI Hub PTQ."
        ),
    )
    p.add_argument("--onnx-dir", type=str, required=True,
                    help="Directory containing image_encoder.onnx and text_encoder.onnx")
    p.add_argument("--output-suffix", type=str, default="_ptq_qdq_u8s8_mixed")
    p.add_argument("--jsonl-path", type=str, required=True)
    p.add_argument("--image-base-dir", type=str, required=True)
    p.add_argument("--calib-size", type=int, default=1000)
    p.add_argument("--val-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--quantize-image", action="store_true", default=True)
    p.add_argument("--no-quantize-image", dest="quantize_image", action="store_false")
    p.add_argument("--quantize-text", action="store_true", default=False)
    p.add_argument("--no-quantize-text", dest="quantize_text", action="store_false")
    p.add_argument("--copy-fp32-for-skipped-branch", action="store_true", default=True)
    p.add_argument(
        "--no-copy-fp32-for-skipped-branch",
        dest="copy_fp32_for_skipped_branch",
        action="store_false",
    )

    p.add_argument("--calib-method", type=str, default="percentile",
                    choices=sorted(CALIB_METHODS.keys()))
    p.add_argument("--quant-format", type=str, default="qdq",
                    choices=sorted(QUANT_FORMATS.keys()))
    p.add_argument("--activation-type", type=str, default="quint8",
                    choices=sorted(QUANT_TYPES.keys()))
    p.add_argument("--weight-type", type=str, default="qint8",
                    choices=sorted(QUANT_TYPES.keys()))
    p.add_argument("--per-channel", action="store_true", default=True)
    p.add_argument("--no-per-channel", dest="per_channel", action="store_false")

    p.add_argument("--op-types", type=str, default="MatMul,Gemm,Conv",
                    help="Comma-separated op types to quantize. Default includes "
                         "MatMul (72 MHSA+MLP), Conv (3 patch embed), Gemm (1 proj). "
                         "GELU/Softmax/LayerNorm are intentionally excluded as FP32 islands.")
    p.add_argument("--model-name", type=str, default="MobileCLIP2-B",
                    help="Used for tokenizer selection only")
    return p.parse_args()


def _parse_op_types(csv: str) -> List[str]:
    return list(dict.fromkeys(t.strip() for t in csv.split(",") if t.strip()))


def _quantize_encoder(
    input_path: str,
    output_path: str,
    reader: CalibrationDataReader,
    calib_method: CalibrationMethod,
    quant_format: QuantFormat,
    activation_type: QuantType,
    weight_type: QuantType,
    per_channel: bool,
    op_types: List[str],
) -> None:
    prep_path = input_path.replace(".onnx", "_prep.onnx")
    try:
        print(f"  Pre-processing: {input_path}")
        shape_inference.quant_pre_process(input_path, prep_path)
        quantize_static(
            model_input=prep_path,
            model_output=output_path,
            calibration_data_reader=reader,
            quant_format=quant_format,
            weight_type=weight_type,
            activation_type=activation_type,
            op_types_to_quantize=op_types,
            per_channel=per_channel,
            calibrate_method=calib_method,
            extra_options={
                "ActivationSymmetric": False,   # U8 asymmetric for activations
                "WeightSymmetric": True,        # S8 symmetric for weights
                "EnableSubgraph": False,        # Avoid subgraph quantization errors
                "ForceQuantizeNoInputCheck": False,  # Respect input type constraints
                # For older Hexagon DSP (HTP v68 and below), uncomment:
                # "ReduceRange": True,          # Use 7-bit effective range
            },
        )
        print(f"  Quantized → {output_path}")
        model = onnx.load(output_path)
        sanitize_value_info_clashing_with_io(model)
        onnx.checker.check_model(model)
        onnx.save(model, output_path)
    finally:
        if os.path.exists(prep_path):
            os.remove(prep_path)


def _copy_if_needed(src_path: str, dst_path: str) -> None:
    if os.path.abspath(src_path) != os.path.abspath(dst_path):
        shutil.copy2(src_path, dst_path)


def _materialize_downloaded_onnx(downloaded_path: str, output_path: str) -> str:
    """Normalize Hub download artifact to a plain ONNX file at output_path."""
    src = Path(downloaded_path)
    dst = Path(output_path)

    def _copy_onnx_with_external_data(src_onnx: Path, dst_onnx: Path) -> None:
        dst_onnx.parent.mkdir(parents=True, exist_ok=True)
        if src_onnx.resolve() != dst_onnx.resolve():
            shutil.copy2(src_onnx, dst_onnx)

        # Keep external tensor data files (e.g., model.data) next to the copied ONNX.
        model = onnx.load(str(src_onnx), load_external_data=False)
        copied_locations = set()
        for tensor in model.graph.initializer:
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            location = None
            for entry in tensor.external_data:
                if entry.key == "location":
                    location = entry.value
                    break
            if not location or location in copied_locations:
                continue
            copied_locations.add(location)
            rel_path = Path(location)
            src_data = src_onnx.parent / rel_path
            dst_data = dst_onnx.parent / rel_path
            if not src_data.exists():
                raise RuntimeError(
                    f"Missing external tensor data referenced by ONNX: {src_data}"
                )
            dst_data.parent.mkdir(parents=True, exist_ok=True)
            if src_data.resolve() != dst_data.resolve():
                shutil.copy2(src_data, dst_data)

        # Fallback: copy all sibling sidecar files. Some exports keep external
        # tensors outside graph.initializer references.
        for sidecar in src_onnx.parent.rglob("*"):
            if not sidecar.is_file() or sidecar.suffix.lower() == ".onnx":
                continue
            rel = sidecar.relative_to(src_onnx.parent)
            dst_sidecar = dst_onnx.parent / rel
            dst_sidecar.parent.mkdir(parents=True, exist_ok=True)
            if sidecar.resolve() != dst_sidecar.resolve():
                shutil.copy2(sidecar, dst_sidecar)

    # Direct ONNX artifact.
    if src.suffix.lower() == ".onnx":
        _copy_onnx_with_external_data(src, dst)
        return str(dst)

    # Zipped ONNX artifact from AI Hub.
    if src.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="w8a16_onnx_") as tmpdir:
            with zipfile.ZipFile(src, "r") as zf:
                zf.extractall(tmpdir)
            onnx_files = sorted(Path(tmpdir).rglob("*.onnx"))
            if not onnx_files:
                raise RuntimeError(f"No .onnx file found inside downloaded archive: {downloaded_path}")
            _copy_onnx_with_external_data(onnx_files[0], dst)
        return str(dst)

    raise RuntimeError(f"Unsupported downloaded artifact type: {downloaded_path}")


def _quantize_encoder_vit_hub(
    input_path: str,
    output_path: str,
    job_name: str,
    calib_records: List[Dict],
    num_calibration_samples: int,
    scheme: str,
) -> None:
    """Quantize an ONNX model through AI Hub quantize job."""
    try:
        import qai_hub as hub
        from qai_hub.client import QuantizeDtype
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            f"{scheme} requires qai_hub to be installed/configured."
        ) from exc

    if scheme == "w8a16_vit_hub":
        activations_dtype = QuantizeDtype.INT16
    elif scheme == "w8a8_vit_hub":
        activations_dtype = QuantizeDtype.INT8
    else:
        raise ValueError(f"Unsupported hub PTQ scheme: {scheme}")

    print(f"  Uploading ONNX to AI Hub: {input_path}")
    uploaded_model = hub.upload_model(input_path)

    model_proto = onnx.load(input_path)
    if not model_proto.graph.input:
        raise RuntimeError("Input ONNX has no graph inputs; cannot build calibration dataset.")
    input_name = model_proto.graph.input[0].name

    capped = max(1, min(num_calibration_samples, len(calib_records)))
    selected = calib_records[:capped]
    if not selected:
        raise ValueError("No calibration records available for hub quantization.")

    print(f"  Preparing calibration dataset (input='{input_name}', samples={len(selected)})")
    calib_images: List[np.ndarray] = []
    for rec in selected:
        img = Image.open(rec["_resolved_path"]).convert("RGB")
        tensor = preprocess_image(img).unsqueeze(0).numpy().astype(np.float32)
        calib_images.append(tensor)
    calib_dataset = hub.upload_dataset({input_name: calib_images}, name=f"{job_name}_calib")

    print(f"  Submitting {scheme} quantize job (calibration samples={len(selected)})")
    quantize_job = hub.submit_quantize_job(
        model=uploaded_model,
        calibration_data=calib_dataset,
        weights_dtype=QuantizeDtype.INT8,
        activations_dtype=activations_dtype,
        name=f"{job_name}_{scheme}",
    )
    wait_result = quantize_job.wait()
    if not wait_result.success:
        raise RuntimeError(f"{scheme} quantize job failed: {quantize_job.url}")

    quantized_model = quantize_job.get_target_model()
    if quantized_model is None:
        raise RuntimeError("W8A16 quantize job finished without target model.")

    downloaded_path = quantized_model.download(output_path)
    normalized_onnx_path = _materialize_downloaded_onnx(downloaded_path, output_path)

    # Keep Hub-exported external-data ONNX layout intact.
    _ = normalized_onnx_path
    print(f"  {scheme} quantized -> {output_path}")


def main() -> None:
    args = parse_args()
    onnx_dir = args.onnx_dir
    suffix = args.output_suffix
    op_types = _parse_op_types(args.op_types)

    image_src = os.path.join(onnx_dir, "image_encoder.onnx")
    text_src = os.path.join(onnx_dir, "text_encoder.onnx")
    image_dst = os.path.join(onnx_dir, f"image_encoder{suffix}.onnx")
    text_dst = os.path.join(onnx_dir, f"text_encoder{suffix}.onnx")

    print(f"PTQ scheme: {args.ptq_scheme}")
    if args.ptq_scheme == "int8_ort":
        print(f"PTQ config: format={args.quant_format} act={args.activation_type} "
              f"wt={args.weight_type} calib={args.calib_method} per_channel={args.per_channel}")
        print(f"  INT8 op types: {op_types}")
        print(f"  FP32 islands:  GELU(Erf), Softmax, LayerNormalization (auto-skipped)")

    records = load_jsonl_records(args.jsonl_path, args.image_base_dir)
    calib_records, _ = split_calib_val(
        records, calib_size=args.calib_size, val_size=args.val_size, seed=args.seed,
    )
    print(f"Calibration records: {len(calib_records)}")

    calib_method = CALIB_METHODS[args.calib_method]
    quant_format = QUANT_FORMATS[args.quant_format]
    act_type = QUANT_TYPES[args.activation_type]
    wt_type = QUANT_TYPES[args.weight_type]

    if args.quantize_image:
        if not os.path.exists(image_src):
            raise FileNotFoundError(f"Missing {image_src}")
        print("Quantizing image encoder ...")
        if args.ptq_scheme == "int8_ort":
            reader = ImageCalibReader(calib_records)
            _quantize_encoder(
                image_src, image_dst, reader,
                calib_method, quant_format, act_type, wt_type,
                args.per_channel, op_types,
            )
        else:
            _quantize_encoder_vit_hub(
                image_src,
                image_dst,
                job_name=f"{args.model_name}_image_encoder",
                calib_records=calib_records,
                num_calibration_samples=args.calib_size,
                scheme=args.ptq_scheme,
            )
    elif args.copy_fp32_for_skipped_branch:
        _copy_if_needed(image_src, image_dst)

    if args.quantize_text:
        if not os.path.exists(text_src):
            raise FileNotFoundError(f"Missing {text_src}")
        if args.ptq_scheme != "int8_ort":
            raise ValueError(
                f"{args.ptq_scheme} currently supports image encoder only. "
                "Use --no-quantize-text or switch to --ptq-scheme int8_ort."
            )
        print("Quantizing text encoder ...")
        tokenizer = open_clip.get_tokenizer(args.model_name)
        reader = TextCalibReader(calib_records, tokenizer)
        _quantize_encoder(
            text_src, text_dst, reader,
            calib_method, quant_format, act_type, wt_type,
            args.per_channel, op_types,
        )
    elif args.copy_fp32_for_skipped_branch:
        _copy_if_needed(text_src, text_dst)

    print("PTQ complete.")


if __name__ == "__main__":
    main()

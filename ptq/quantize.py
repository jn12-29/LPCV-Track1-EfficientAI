"""Post-Training Quantization for MobileCLIP2 ONNX encoders.

Condensed from the LPCVTrack1 mobileclipv2_quant.py reference, tailored for
MobileCLIP2-B (ViT-based vision encoder).

Default configuration:
  - Format:      QDQ
  - Activation:  QUInt8
  - Weight:      QInt8 (U8S8)
  - Calibration: Percentile
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
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
    p.add_argument("--onnx-dir", type=str, required=True,
                    help="Directory containing image_encoder.onnx and text_encoder.onnx")
    p.add_argument("--output-suffix", type=str, default="_ptq_qdq_u8s8_pct")
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

    p.add_argument("--op-types", type=str, default="MatMul",
                    help="Comma-separated op types to quantize (ViT is MatMul-heavy)")
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
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
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


def main() -> None:
    args = parse_args()
    onnx_dir = args.onnx_dir
    suffix = args.output_suffix
    op_types = _parse_op_types(args.op_types)

    image_src = os.path.join(onnx_dir, "image_encoder.onnx")
    text_src = os.path.join(onnx_dir, "text_encoder.onnx")
    image_dst = os.path.join(onnx_dir, f"image_encoder{suffix}.onnx")
    text_dst = os.path.join(onnx_dir, f"text_encoder{suffix}.onnx")

    print(f"PTQ config: format={args.quant_format} act={args.activation_type} "
          f"wt={args.weight_type} calib={args.calib_method} ops={op_types}")

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
        print("Quantizing image encoder …")
        reader = ImageCalibReader(calib_records)
        _quantize_encoder(
            image_src, image_dst, reader,
            calib_method, quant_format, act_type, wt_type,
            args.per_channel, op_types,
        )
    elif args.copy_fp32_for_skipped_branch:
        _copy_if_needed(image_src, image_dst)

    if args.quantize_text:
        if not os.path.exists(text_src):
            raise FileNotFoundError(f"Missing {text_src}")
        print("Quantizing text encoder …")
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

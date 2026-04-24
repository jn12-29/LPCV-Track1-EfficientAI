from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import qai_hub as hub
from PIL import Image
from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.args import export_parser

import sys
sys.path.append("/mnt/sata1/charles/LPCV-Track1-EfficientAI")
from onnx_utils import sanitize_value_info_clashing_with_io
from ptq.dataset import load_jsonl_records, split_calib_val
from utils.preprocess import preprocess_image
from vit.export import (
    compile_model,
    link_model,
    profile_model,
    quantize_model,  # imported intentionally; see note in _quantize_for_mobileclip
)


class MobileClipImageAdapter:
    """Minimal adapter to reuse vit.export helpers on image-only ONNX."""

    def __init__(self, input_name: str) -> None:
        self.input_name = input_name

    def get_input_spec(self, **_: Any) -> dict[str, tuple[tuple[int, ...], str]]:
        return {self.input_name: ((1, 3, 224, 224), "float32")}

    def get_output_spec(self) -> dict[str, tuple[tuple[int, ...], str]]:
        # Metadata-only; actual value is not required by compile/profile flow.
        return {"output_0": ((1, 512), "float32")}

    def sample_inputs(self, input_spec=None, use_channel_last_format: bool = False):
        spec = input_spec or self.get_input_spec()
        (shape, dtype_str) = next(iter(spec.values()))
        dtype = np.float32 if dtype_str == "float32" else np.float32
        return {self.input_name: [np.zeros(shape, dtype=dtype)]}

    def calibration_dataset_name(self):
        return None

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        extra_options: str,
        device,
        model_id: str,
    ) -> str:
        _ = (precision, device, model_id)
        base = f"--target_runtime {target_runtime.value}"
        return f"{base} {extra_options}".strip()

    def get_hub_profile_options(self, target_runtime: TargetRuntime, profile_options: str) -> str:
        _ = target_runtime
        return profile_options

    def get_hub_link_options(self, target_runtime: TargetRuntime, extra_options: str = "") -> str:
        _ = target_runtime
        return extra_options

    def get_hub_quantize_options(self, precision: Precision, extra_options: str = "") -> str:
        _ = precision
        return extra_options

    def write_supplementary_files(self, dst_path: Path, model_metadata) -> None:
        _ = (dst_path, model_metadata)
        return


class MobileCLIP2BImageModelConfig:
    """Parser-only model config for MobileCLIP2-B image export CLI."""

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "MobileCLIP2-B",
    ) -> "MobileCLIP2BImageModelConfig":
        _ = model_name
        return cls()


def _parse_args() -> argparse.Namespace:
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.float: [
            TargetRuntime.TFLITE,
            TargetRuntime.QNN_DLC,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
        Precision.w8a8_mixed_int16: [
            TargetRuntime.ONNX,
        ],
        Precision.w8a16: [
            TargetRuntime.QNN_DLC,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
        Precision.w8a8: [
            TargetRuntime.TFLITE,
            TargetRuntime.QNN_DLC,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
    }
    # Keep CLI behavior aligned with vit/export.py.
    p = export_parser(
        model_cls=MobileCLIP2BImageModelConfig,
        export_fn=export_model,
        supported_precision_runtimes=supported_precision_runtimes,
        default_export_device="XR2 Gen 2 (Proxy)",
    )
    # Extra args required by this image-only ONNX pipeline.
    p.add_argument("--onnx-path", type=str, default="exported_MobileCLIP2-B_onnx/image_encoder.onnx")
    p.add_argument("--ids-file", type=str, default="ptq_compile_ids.json")
    p.add_argument("--jsonl-path", type=str, default="./build_datasets/data/vg_llm_contrastive_v1/vg_llm_contrastive.jsonl")
    p.add_argument("--image-base-dir", type=str, default="./")
    p.add_argument("--calib-size", type=int, default=10)
    p.add_argument("--val-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-compile-id", type=str, default="jp01nr62g")
    return p.parse_args()


def _to_precision(name: str) -> Precision:
    if isinstance(name, Precision):
        return name
    if name == "float":
        return Precision.float
    if name == "w8a8":
        return Precision.w8a8
    if name == "w8a16":
        return Precision.w8a16
    raise ValueError(f"Unsupported precision: {name}")


def _to_runtime(name: str) -> TargetRuntime:
    if isinstance(name, TargetRuntime):
        return name
    for rt in TargetRuntime:
        if rt.value == name:
            return rt
    raise ValueError(f"Unsupported runtime: {name}")


def _build_calibration_dataset(
    *,
    jsonl_path: str,
    image_base_dir: str,
    input_name: str,
    calib_size: int,
    num_calibration_samples: int | None,
    val_size: int,
    seed: int,
) -> hub.Dataset:
    records = load_jsonl_records(jsonl_path=jsonl_path, image_base_dir=image_base_dir)
    effective_calib_size = (
        num_calibration_samples if num_calibration_samples is not None else calib_size
    )
    calib_records, _ = split_calib_val(
        records,
        calib_size=effective_calib_size,
        val_size=val_size,
        seed=seed,
    )
    if num_calibration_samples is not None:
        calib_records = calib_records[:effective_calib_size]
    samples: list[np.ndarray] = []
    for rec in calib_records:
        img = Image.open(rec["_resolved_path"]).convert("RGB")
        samples.append(preprocess_image(img).unsqueeze(0).numpy().astype(np.float32))
    return hub.upload_dataset({input_name: samples})


def _quantize_for_mobileclip(
    *,
    source_onnx_model: hub.Model,
    adapter: MobileClipImageAdapter,
    model_name: str,
    precision: Precision,
    jsonl_path: str,
    image_base_dir: str,
    calib_size: int,
    num_calibration_samples: int | None,
    val_size: int,
    seed: int,
    quantize_options: str,
) -> hub.Model:
    if precision == Precision.float:
        return source_onnx_model

    # NOTE: vit.export.quantize_model relies on qai_hub_models quantization helpers that
    # fetch calibration data from BaseModel contracts. For MobileCLIP image ONNX we need
    # explicit JSONL-based calibration samples, so direct submit_quantize_job is required.
    _ = quantize_model
    dataset = _build_calibration_dataset(
        jsonl_path=jsonl_path,
        image_base_dir=image_base_dir,
        input_name=adapter.input_name,
        calib_size=calib_size,
        num_calibration_samples=num_calibration_samples,
        val_size=val_size,
        seed=seed,
    )
    if precision == Precision.w8a16:
        act_dtype = hub.QuantizeDtype.INT16
    else:
        act_dtype = hub.QuantizeDtype.INT8
    qjob = hub.submit_quantize_job(
        model=source_onnx_model,
        calibration_data=dataset,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=act_dtype,
        name=f"{model_name}_{str(precision)}_quantize",
        options=quantize_options,
    )
    qjob.wait()
    if qjob.get_status().failure:
        raise RuntimeError(f"Quantize failed: {qjob.url}")
    quantized = qjob.get_target_model()
    if quantized is None:
        raise RuntimeError(f"Quantize target model unavailable: {qjob.url}")
    return quantized


def export_model(
    device: hub.Device,
    precision: Precision = Precision.float,
    target_runtime: TargetRuntime = TargetRuntime.QNN_DLC,
    compile_options: str = "--truncate_64bit_io",
    quantize_options: str = "",
    profile_options: str = "--max_profiler_iterations 100",
    skip_compiling: bool = False,
    skip_profiling: bool = False,
    model_name: str = "MobileCLIP2-B",
    onnx_path: str = "exported_MobileCLIP2-B_onnx/image_encoder.onnx",
    ids_file: str = "ptq_compile_ids.json",
    jsonl_path: str = "./build_datasets/data/vg_llm_contrastive_v1/vg_llm_contrastive.jsonl",
    image_base_dir: str = "./",
    calib_size: int = 10,
    num_calibration_samples: int | None = None,
    val_size: int = 100,
    seed: int = 42,
    text_compile_id: str = "jp01nr62g",
    **additional_model_kwargs: Any,
) -> None:
    """
    Export MobileCLIP2-B image encoder via ONNX->QAI Hub pipeline.

    Parameters
    ----------
    device
        Device for export/compile/profile (for example, hub.Device("XR2 Gen 2 (Proxy)")).
    precision
        Quantization precision. Use float to skip quantization.
    target_runtime
        Runtime target for compile/link.
    compile_options
        Additional options passed to compile submission.
    quantize_options
        Additional options passed to quantize submission.
    profile_options
        Additional options passed to profile submission.
    skip_compiling
        If set, does compiling after optional quantization.
    skip_profiling
        If set, skips profiling on target device.
    model_name
        Base model name used in job naming and metadata payload.
    onnx_path
        Path to image encoder ONNX model.
    ids_file
        Output JSON path for compile IDs.
    jsonl_path
        JSONL dataset path used to build calibration samples.
    image_base_dir
        Base directory for resolving image paths in JSONL records.
    calib_size
        Calibration split size used when num_calibration_samples is not provided.
    num_calibration_samples
        Optional explicit calibration sample count override.
    val_size
        Validation split size used by split helper.
    seed
        Random seed for calibration/validation split.
    text_compile_id
        Text compile ID written alongside image compile ID in ids_file.
    """
    _ = additional_model_kwargs
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX: {onnx_path}")

    model_proto = onnx.load(str(onnx_path), load_external_data=False)
    sanitize_value_info_clashing_with_io(model_proto)
    input_name = model_proto.graph.input[0].name

    adapter = MobileClipImageAdapter(input_name=input_name)
    device = device if isinstance(device, hub.Device) else hub.Device(device)
    precision = _to_precision(precision)
    target_runtime = _to_runtime(target_runtime)
    model_name = f"{model_name}_image"

    src_model = hub.upload_model(str(onnx_path), name=f"{model_name}_onnx")
    quantized_model: hub.Model | None = None
    if precision != Precision.float:
        quantized_model = _quantize_for_mobileclip(
            source_onnx_model=src_model,
            adapter=adapter,
            model_name=model_name,
            precision=precision,
            jsonl_path=jsonl_path,
            image_base_dir=image_base_dir,
            calib_size=calib_size,
            num_calibration_samples=num_calibration_samples,
            val_size=val_size,
            seed=seed,
            quantize_options=quantize_options,
        )

    if skip_compiling:
        print("Skipping compile due to --skip-compiling.")
        return

    # Keep the compile call style aligned with vit.export.export_model().
    compile_job = compile_model(
        adapter,
        f"{model_name}_{str(precision)}",
        device,
        target_runtime,
        precision,
        source_model=quantized_model or src_model,
        input_spec=adapter.get_input_spec(),
        extra_options=compile_options,
    )
    compiled_model = compile_job.get_target_model()
    if compiled_model is None:
        raise RuntimeError(f"Compile failed: {compile_job.url}")

    link_job = None
    target_model = compiled_model
    if getattr(target_runtime, 'uses_hub_link', False):
        link_job = link_model(
            compiled_model=compiled_model,
            device=device,
            model_name=f"{model_name}_{str(precision)}",
            model=adapter,
            target_runtime=target_runtime,
        )
        linked = link_job.get_target_model()
        if linked is None:
            raise RuntimeError(f"Link failed: {link_job.url}")
        target_model = linked

    if not skip_profiling:
        profile_model(
            model_name=f"{model_name}_{str(precision)}",
            device=device,
            options=profile_options,
            target_model=target_model,
        )

    payload = {
        "model_name": model_name.removesuffix("_image"),
        "precision": str(precision),
        "target_runtime": target_runtime.value,
        "image_compile_id": compile_job.job_id,
        "text_compile_id": text_compile_id,
    }
    with open(ids_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved ids to {ids_file}")


def main() -> None:
    args = _parse_args()
    export_model(**vars(args))


if __name__ == "__main__":
    main()
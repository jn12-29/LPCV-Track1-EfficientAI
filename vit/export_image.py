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

import sys
sys.path.append("/mnt/sata1/charles/LPCV-Track1-EfficientAI")
from onnx_utils import sanitize_value_info_clashing_with_io
from ptq.dataset import load_jsonl_records, split_calib_val
from utils.preprocess import preprocess_image
from vit.export import (
    compile_model,
    download_model,
    inference_model,
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MobileCLIP2-B image-only export pipeline via vit.export helpers")
    p.add_argument("--onnx-path", type=str, default="exported_MobileCLIP2-B_onnx/image_encoder.onnx")
    p.add_argument("--model-name", type=str, default="MobileCLIP2-B")
    p.add_argument("--device", type=str, default="XR2 Gen 2 (Proxy)")
    p.add_argument("--precision", type=str, default="w8a8", choices=["float", "w8a8", "w8a16"])
    p.add_argument(
        "--target-runtime",
        type=str,
        default="qnn_dlc",
        choices=["onnx", "qnn_dlc", "qnn_context_binary", "precompiled_qnn_onnx", "tflite"],
    )
    p.add_argument("--compile-options", type=str, default="--truncate_64bit_io")
    p.add_argument("--profile-options", type=str, default="--max_profiler_iterations 100")
    p.add_argument("--skip-profiling", action="store_true")
    p.add_argument("--skip-inferencing", action="store_true")
    p.add_argument("--skip-downloading", action="store_true")
    p.add_argument("--output-dir", type=str, default="export_assets_mobileclip_image")
    p.add_argument("--ids-file", type=str, default="ptq_compile_ids.json")
    p.add_argument("--jsonl-path", type=str, default="./build_datasets/data/vg_llm_contrastive_v1/vg_llm_contrastive.jsonl")
    p.add_argument("--image-base-dir", type=str, default="./")
    p.add_argument("--calib-size", type=int, default=10)
    p.add_argument("--val-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-compile-id", type=str, default="jp01nr62g")
    return p.parse_args()


def _to_precision(name: str) -> Precision:
    if name == "float":
        return Precision.float
    if name == "w8a8":
        return Precision.w8a8
    if name == "w8a16":
        return Precision.w8a16
    raise ValueError(f"Unsupported precision: {name}")


def _to_runtime(name: str) -> TargetRuntime:
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
    val_size: int,
    seed: int,
) -> hub.Dataset:
    records = load_jsonl_records(jsonl_path=jsonl_path, image_base_dir=image_base_dir)
    calib_records, _ = split_calib_val(records, calib_size=calib_size, val_size=val_size, seed=seed)
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
    args: argparse.Namespace,
) -> hub.Model:
    if precision == Precision.float:
        return source_onnx_model

    # NOTE: vit.export.quantize_model relies on qai_hub_models quantization helpers that
    # fetch calibration data from BaseModel contracts. For MobileCLIP image ONNX we need
    # explicit JSONL-based calibration samples, so direct submit_quantize_job is required.
    _ = quantize_model
    dataset = _build_calibration_dataset(
        jsonl_path=args.jsonl_path,
        image_base_dir=args.image_base_dir,
        input_name=adapter.input_name,
        calib_size=args.calib_size,
        val_size=args.val_size,
        seed=args.seed,
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
    )
    qjob.wait()
    if qjob.get_status().failure:
        raise RuntimeError(f"Quantize failed: {qjob.url}")
    quantized = qjob.get_target_model()
    if quantized is None:
        raise RuntimeError(f"Quantize target model unavailable: {qjob.url}")
    return quantized


def main() -> None:
    args = _parse_args()
    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX: {onnx_path}")

    model_proto = onnx.load(str(onnx_path), load_external_data=False)
    sanitize_value_info_clashing_with_io(model_proto)
    input_name = model_proto.graph.input[0].name

    adapter = MobileClipImageAdapter(input_name=input_name)
    device = hub.Device(args.device)
    precision = _to_precision(args.precision)
    target_runtime = _to_runtime(args.target_runtime)
    model_name = f"{args.model_name}_image"

    src_model = hub.upload_model(str(onnx_path), name=f"{model_name}_onnx")
    quantized_or_source = _quantize_for_mobileclip(
        source_onnx_model=src_model,
        adapter=adapter,
        model_name=model_name,
        precision=precision,
        args=args,
    )

    compile_job = compile_model(
        model=adapter,
        model_name=f"{model_name}_{precision.value}",
        device=device,
        target_runtime=target_runtime,
        precision=precision,
        source_model=quantized_or_source,
        input_spec=adapter.get_input_spec(),
        extra_options=args.compile_options,
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
            model_name=f"{model_name}_{precision.value}",
            model=adapter,
            target_runtime=target_runtime,
        )
        linked = link_job.get_target_model()
        if linked is None:
            raise RuntimeError(f"Link failed: {link_job.url}")
        target_model = linked

    profile_job = None
    if not args.skip_profiling:
        profile_job = profile_model(
            model_name=f"{model_name}_{precision.value}",
            device=device,
            options=args.profile_options,
            target_model=target_model,
        )

    inference_job = None
    if not args.skip_inferencing:
        inference_job = inference_model(
            inputs=adapter.sample_inputs(adapter.get_input_spec()),
            model_name=f"{model_name}_{precision.value}",
            device=device,
            options=args.profile_options,
            target_model=target_model,
        )

    tool_versions = None
    if profile_job is not None and profile_job.wait():
        tool_versions = ToolVersions.from_job(profile_job)
    elif inference_job is not None and inference_job.wait():
        tool_versions = ToolVersions.from_job(inference_job)
    elif compile_job.wait():
        tool_versions = ToolVersions.from_job(compile_job)

    download_path = None
    if (not args.skip_downloading) and (tool_versions is not None):
        download_path = download_model(
            output_dir=Path(args.output_dir) / f"{model_name}_{precision.value}_{target_runtime.value}",
            model=adapter,
            runtime=target_runtime,
            precision=precision,
            tool_versions=tool_versions,
            target_model=target_model,
            model_name=model_name,
            zip_assets=False,
        )

    payload = {
        "model_name": args.model_name,
        "precision": args.precision,
        "target_runtime": args.target_runtime,
        "image_compile_id": compile_job.job_id,
        "image_profile_id": profile_job.job_id if profile_job else None,
        "image_inference_id": inference_job.job_id if inference_job else None,
        "text_compile_id": args.text_compile_id,
        "download_path": str(download_path) if download_path else None,
    }
    with open(args.ids_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved ids to {args.ids_file}")


if __name__ == "__main__":
    main()
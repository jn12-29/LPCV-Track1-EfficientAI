from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
import torch.nn as nn
from timm.utils import reparameterize_model

from utils.clip_utils import _load_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--output-postfix",
        type=str,
        default="",
        help="Suffix appended to exported ONNX filenames and output directory.",
    )
    return parser.parse_args()


class OpenClipVisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image)


class OpenClipTextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(dtype=torch.int64)
        eot_pos = token_ids.argmax(dim=-1, keepdim=True)
        positions = torch.arange(
            token_ids.shape[-1], device=token_ids.device
        ).unsqueeze(0)
        mask = (positions <= eot_pos).to(token_ids.dtype)
        return self.model.encode_text(token_ids * mask)


def verify_onnx(
    onnx_path: str,
    input_dict: dict,
    pt_output: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {k: v.numpy() for k, v in input_dict.items()}
    ort_out = sess.run(None, ort_inputs)[0]
    pt_out = pt_output.numpy()
    max_diff = np.abs(ort_out - pt_out).max()
    status = "PASS" if np.allclose(ort_out, pt_out, rtol=rtol, atol=atol) else "FAIL"
    print(f"  Max abs diff (ONNX vs PyTorch): {max_diff:.6f}  {status}")
    if status == "FAIL":
        raise RuntimeError(
            f"ONNX output mismatch for {onnx_path}. Max diff={max_diff:.6f}"
        )


def replace_gelu_with_tanh_approx(model: nn.Module) -> None:
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.GELU) and child.approximate == "none":
                setattr(parent, name, nn.GELU(approximate="tanh"))


def _simplify_onnx(onnx_path: str) -> None:
    model = onnx.load(onnx_path)
    before_nodes = len(model.graph.node)
    before_size = os.path.getsize(onnx_path) / 1024 / 1024
    simplified, ok = onnxsim.simplify(model)
    if ok:
        onnx.save(simplified, onnx_path)
        after_nodes = len(simplified.graph.node)
        after_size = os.path.getsize(onnx_path) / 1024 / 1024
        print(
            f"  Simplified: nodes {before_nodes} → {after_nodes} "
            f"({before_nodes - after_nodes:+d}), "
            f"size {before_size:.2f} → {after_size:.2f} MB "
            f"({after_size - before_size:+.2f} MB)"
        )
    else:
        print(f"  Simplification failed (kept original): {onnx_path}")


def export_encoders_to_onnx(clip_model: nn.Module, output_dir: str) -> None:
    """Export image and text encoders to ONNX.

    clip_model must already be reparameterized and in eval mode.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(clip_model.parameters()).device

    image_encoder = OpenClipVisionEncoder(clip_model).eval()
    text_encoder = OpenClipTextEncoder(clip_model).eval()

    dummy_image = torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device)
    dummy_text = torch.randint(0, 49408, (1, 77), dtype=torch.int64, device=device)

    print("\nCalculating PyTorch baseline outputs for validation...")
    with torch.no_grad():
        pt_img_feat = image_encoder(dummy_image).cpu()
        pt_txt_feat = text_encoder(dummy_text).cpu()

    image_onnx_path = os.path.join(output_dir, "image_encoder.onnx")
    text_onnx_path = os.path.join(output_dir, "text_encoder.onnx")

    print(f"\nExporting Image Encoder to {image_onnx_path}...")
    torch.onnx.export(
        image_encoder.cpu(),
        dummy_image.cpu(),
        image_onnx_path,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    _simplify_onnx(image_onnx_path)
    verify_onnx(image_onnx_path, {"image": dummy_image.cpu()}, pt_img_feat)

    print(f"\nExporting Text Encoder to {text_onnx_path}...")
    torch.onnx.export(
        text_encoder.cpu(),
        dummy_text.cpu(),
        text_onnx_path,
        input_names=["text"],
        output_names=["text_embedding"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    _simplify_onnx(text_onnx_path)
    verify_onnx(text_onnx_path, {"text": dummy_text.cpu()}, pt_txt_feat)

    print(f"\nExport complete → {output_dir}")


def main() -> None:
    args = parse_args()

    if args.output_postfix:
        output_dir_name = f"exported_{args.model_name}{args.output_postfix}_onnx"
    else:
        output_dir_name = f"exported_{args.model_name}_onnx"
    print(f"Saving ONNX files to directory: {os.path.abspath(output_dir_name)}")

    device = torch.device("cpu")

    checkpoint_path = args.checkpoint_path
    is_qat_ckpt = False
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        is_qat_ckpt = isinstance(ckpt, dict) and ckpt.get("qat_enabled", False)

    if is_qat_ckpt:
        # QAT checkpoint: load into reparameterized base model by filtering keys.
        from utils.qat_utils import extract_base_model_state_dict
        clip_model, _, _ = _load_clip(model_name=args.model_name, device=device)
        clip_model = reparameterize_model(clip_model)
        qat_state_dict = ckpt.get("model_state_dict", ckpt)
        filtered = extract_base_model_state_dict(qat_state_dict, clip_model)
        missing, unexpected = clip_model.load_state_dict(filtered, strict=False)
        print(f"Loaded QAT checkpoint: {checkpoint_path}")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        clip_model, _, _ = _load_clip(
            model_name=args.model_name,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        clip_model = reparameterize_model(clip_model)

    clip_model.eval()
    export_encoders_to_onnx(clip_model, output_dir_name)

    if checkpoint_path:
        print(f"Exported from checkpoint: {Path(checkpoint_path).resolve()}")


if __name__ == "__main__":
    main()

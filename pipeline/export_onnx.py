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
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=77,
        help="Maximum number of tokens for text input. Can speed up inference for short text scene",
    )
    return parser.parse_args()


class OpenClipVisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image)


class OpenClipTextEncoder(nn.Module):
    def __init__(self, model, max_text_len: int = 77):
        super().__init__()
        self.model = model
        self.max_text_len = max_text_len
        if (getattr(model.text, "attn_mask", None) is not None) and (max_text_len < 77):
            self.model.text.attn_mask = self.model.text.attn_mask[
                :max_text_len, :max_text_len
            ]
            print(f"Truncated text attn_mask to [{max_text_len}, {max_text_len}]")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(dtype=torch.int64)
        eot_pos = token_ids.argmax(dim=-1, keepdim=True)
        positions = torch.arange(
            token_ids.shape[-1], device=token_ids.device
        ).unsqueeze(0)
        mask = (positions <= eot_pos).to(token_ids.dtype)
        token_ids = token_ids * mask
        if self.max_text_len < token_ids.shape[-1]:
            token_ids = token_ids[:, : self.max_text_len]
        return self.model.encode_text(token_ids)


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


def _export_encoder_onnx(
    encoder: nn.Module,
    dummy: torch.Tensor,
    onnx_path: str,
    input_name: str,
    output_name: str,
    pt_feat: torch.Tensor,
) -> None:
    """Export one encoder to ONNX, simplify, and verify against PyTorch output."""
    print(f"\nExporting to {onnx_path}...")
    torch.onnx.export(
        encoder.cpu(),
        dummy.cpu(),
        onnx_path,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=True,
    )
    # _simplify_onnx(onnx_path)
    verify_onnx(onnx_path, {input_name: dummy.cpu()}, pt_feat)


def _split_clip_onnx(
    full_onnx_path: str,
    inputs: list,
    outputs: list,
    image_onnx_path: str,
    text_onnx_path: str,
) -> None:
    """Split a combined CLIP ONNX (from AIMET sim.export) into separate encoder files.

    Determines which graph output belongs to the image vs text path by BFS from each input.
    """
    model_proto = onnx.load(full_onnx_path)
    graph = model_proto.graph

    consumer_map: dict = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumer_map.setdefault(inp, []).extend(o for o in node.output if o)

    output_set = {o.name for o in graph.output}

    def reachable_graph_outputs(start: str) -> set:
        visited, queue, reached = set(), [start], set()
        while queue:
            t = queue.pop()
            if t in visited:
                continue
            visited.add(t)
            if t in output_set:
                reached.add(t)
            queue.extend(consumer_map.get(t, []))
        return reached

    img_outs = reachable_graph_outputs(inputs[0])
    txt_outs = reachable_graph_outputs(inputs[1])
    img_out = next(iter(img_outs)) if img_outs else outputs[0]
    txt_out = next(iter(txt_outs)) if txt_outs else outputs[1]

    print(f"  Image path: {inputs[0]} → {img_out}")
    print(f"  Text  path: {inputs[1]} → {txt_out}")

    onnx.utils.extract_model(full_onnx_path, image_onnx_path, [inputs[0]], [img_out])
    onnx.utils.extract_model(full_onnx_path, text_onnx_path, [inputs[1]], [txt_out])
    print(f"  Saved {image_onnx_path}")
    print(f"  Saved {text_onnx_path}")


def export_quantized_encoders_to_onnx(sim, output_dir: str) -> None:
    """Export image and text encoders with Q/DQ nodes via AIMET sim.export().

    AIMET converts fake-quant (QuantizeDequantize) nodes into real ONNX
    QuantizeLinear + DequantizeLinear ops.  The combined ONNX is then split
    into image_encoder.onnx and text_encoder.onnx using graph connectivity.
    """
    import shutil
    import tempfile

    os.makedirs(output_dir, exist_ok=True)
    dummy_image = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    dummy_text = torch.zeros(1, 77, dtype=torch.long)

    # AIMET export requires model and dummy inputs on the same device.
    model_device = next(sim.model.parameters()).device
    sim.model.cpu()
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            print("\nExporting quantized ONNX via AIMET (Q/DQ nodes preserved)...")
            sim.export(tmp_dir, "model", (dummy_image, dummy_text))

            full_onnx = os.path.join(tmp_dir, "model.onnx")
            model_proto = onnx.load(full_onnx)
            inputs = [i.name for i in model_proto.graph.input]
            outputs = [o.name for o in model_proto.graph.output]
            print(f"  Inputs : {inputs}")
            print(f"  Outputs: {outputs}")

            if len(inputs) >= 2 and len(outputs) >= 2:
                image_onnx = os.path.join(output_dir, "image_encoder.onnx")
                text_onnx = os.path.join(output_dir, "text_encoder.onnx")
                _split_clip_onnx(full_onnx, inputs, outputs, image_onnx, text_onnx)
            else:
                combined = os.path.join(output_dir, "combined_quantized.onnx")
                shutil.copy(full_onnx, combined)
                print(
                    f"  Cannot split (inputs={inputs}, outputs={outputs}), "
                    f"saved combined → {combined}"
                )
    finally:
        sim.model.to(model_device)

    print(f"\nExport complete → {output_dir}")


def export_encoders_to_onnx(
    clip_model: nn.Module, output_dir: str, max_text_len: int = 77
) -> None:
    """Export image and text encoders to ONNX.

    clip_model must already be reparameterized and in eval mode.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(clip_model.parameters()).device

    image_encoder = OpenClipVisionEncoder(clip_model).eval()
    text_encoder = OpenClipTextEncoder(clip_model, max_text_len).eval()

    dummy_image = torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device)
    dummy_text = torch.randint(0, 49408, (1, 77), dtype=torch.int64, device=device)

    print("\nCalculating PyTorch baseline outputs for validation...")
    with torch.no_grad():
        pt_img_feat = image_encoder(dummy_image).cpu()
        pt_txt_feat = text_encoder(dummy_text).cpu()

    image_onnx_path = os.path.join(output_dir, "image_encoder.onnx")
    text_onnx_path = os.path.join(output_dir, "text_encoder.onnx")

    _export_encoder_onnx(
        image_encoder, dummy_image, image_onnx_path, "image", "embedding", pt_img_feat
    )
    _export_encoder_onnx(
        text_encoder, dummy_text, text_onnx_path, "text", "text_embedding", pt_txt_feat
    )

    print(f"\nExport complete → {output_dir}")


def main() -> None:
    args = parse_args()

    output_dir_name = (
        f"exported_{args.model_name}{args.output_postfix}_onnx"
        if args.output_postfix
        else f"exported_{args.model_name}_onnx"
    )
    print(f"Saving ONNX files to directory: {os.path.abspath(output_dir_name)}")

    clip_model, _, _ = _load_clip(
        model_name=args.model_name,
        device=torch.device("cpu"),
        checkpoint_path=args.checkpoint_path,
    )

    sim = getattr(clip_model, "_qat_sim", None)
    if sim is not None:
        export_quantized_encoders_to_onnx(sim, output_dir_name)
    else:
        clip_model = reparameterize_model(clip_model)
        clip_model.eval()
        export_encoders_to_onnx(clip_model, output_dir_name, args.max_text_len)

    if args.checkpoint_path:
        print(f"Exported from checkpoint: {Path(args.checkpoint_path).resolve()}")


if __name__ == "__main__":
    main()

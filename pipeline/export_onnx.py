from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
from typing import Tuple

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Downsample image inside the ONNX graph to this resolution (external input stays 224x224).",
    )
    parser.add_argument(
        "--image-mode",
        type=str,
        default="resize",
        choices=["resize", "crop"],
        help="Deprecated compatibility flag. Prefer --resize / --crop rings.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Shrink by N ViT patch rings using bilinear resize. 0 keeps 224.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=0,
        help="Shrink by N ViT patch rings using center crop. Applied after resize when both are set.",
    )
    return parser.parse_args()


def _infer_vit_patch_size(model: nn.Module, orig_size: int = 224) -> int:
    trunk = getattr(getattr(model, "visual", None), "trunk", model)
    patch_embed = getattr(trunk, "patch_embed", None)
    if patch_embed is not None:
        img_size = getattr(patch_embed, "img_size", None)
        grid_size = getattr(patch_embed, "grid_size", None)
        if (
            isinstance(img_size, tuple) and len(img_size) == 2 and img_size[0] == img_size[1]
            and isinstance(grid_size, tuple) and len(grid_size) == 2 and grid_size[0] == grid_size[1]
            and grid_size[0] > 0 and img_size[0] % grid_size[0] == 0
        ):
            return img_size[0] // grid_size[0]

    pos_embed_param = getattr(trunk, "pos_embed", None)
    if pos_embed_param is None:
        raise ValueError("Cannot infer ViT patch size: model has no pos_embed.")

    num_tokens = pos_embed_param.shape[1]
    sqrt_n = int(num_tokens**0.5)
    if sqrt_n * sqrt_n == num_tokens:
        old_grid = sqrt_n
    else:
        sqrt_n = int((num_tokens - 1) ** 0.5)
        if sqrt_n * sqrt_n != num_tokens - 1:
            raise ValueError(f"Cannot infer ViT grid from pos_embed shape {tuple(pos_embed_param.shape)}.")
        old_grid = sqrt_n

    patch_size = orig_size // old_grid
    if patch_size <= 0 or orig_size % old_grid != 0:
        raise ValueError(f"Invalid inferred patch size: orig_size={orig_size}, old_grid={old_grid}.")
    return patch_size


def resolve_image_rings(
    *,
    image_size: int,
    image_mode: str,
    resize_rings: int,
    crop_rings: int,
    patch_size: int,
    base_size: int = 224,
) -> Tuple[int, int, int]:
    if resize_rings < 0 or crop_rings < 0:
        raise ValueError("--resize and --crop must be >= 0.")

    if resize_rings == 0 and crop_rings == 0 and image_size != base_size:
        delta = base_size - image_size
        if delta <= 0 or delta % (2 * patch_size) != 0:
            raise ValueError(
                f"--image-size {image_size} is incompatible with patch_size={patch_size}; "
                f"expected base_size - 2*k*patch_size."
            )
        legacy_rings = delta // (2 * patch_size)
        if image_mode == "crop":
            crop_rings = legacy_rings
        else:
            resize_rings = legacy_rings

    resized_size = base_size - 2 * patch_size * resize_rings
    final_size = resized_size - 2 * patch_size * crop_rings
    min_size = patch_size
    if resized_size < min_size or final_size < min_size:
        raise ValueError(
            f"Requested resize/crop is too aggressive for base_size={base_size}, patch_size={patch_size}: "
            f"resize->{resized_size}, final->{final_size}."
        )
    return resize_rings, crop_rings, final_size


def apply_image_resize_crop(
    image: torch.Tensor,
    *,
    resize_rings: int,
    crop_rings: int,
    patch_size: int,
) -> torch.Tensor:
    if resize_rings > 0:
        resized_size = image.shape[-1] - 2 * patch_size * resize_rings
        image = F.interpolate(
            image,
            size=(resized_size, resized_size),
            mode="bilinear",
            align_corners=False,
        )
    if crop_rings > 0:
        crop_size = image.shape[-1] - 2 * patch_size * crop_rings
        h, w = image.shape[-2], image.shape[-1]
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        image = image[:, :, top:top + crop_size, left:left + crop_size]
    return image


def _resize_vit_pos_embed(model: nn.Module, image_size: int, orig_size: int = 224) -> None:
    """Bicubic-interpolate ViT positional embeddings for a new image resolution.
    Grid size and cls-token presence are inferred from pos_embed shape.
    No-op for CNN backbones (no pos_embed) or when image_size == orig_size.
    """
    import math
    if image_size == orig_size:
        return
    trunk = getattr(getattr(model, "visual", None), "trunk", model)
    pos_embed_param = getattr(trunk, "pos_embed", None)
    if pos_embed_param is None:
        return
    pos_embed = pos_embed_param.data
    num_tokens = pos_embed.shape[1]
    sqrt_n = math.isqrt(num_tokens)
    if sqrt_n * sqrt_n == num_tokens:
        has_cls, old_grid = False, sqrt_n
    elif math.isqrt(num_tokens - 1) ** 2 == num_tokens - 1:
        has_cls, old_grid = True, math.isqrt(num_tokens - 1)
    else:
        print(f"  _resize_vit_pos_embed: cannot infer grid from {pos_embed.shape}, skipping.")
        return
    patch_size = orig_size // old_grid
    new_grid = image_size // patch_size
    if old_grid == new_grid:
        return
    with torch.no_grad():
        dim = pos_embed.shape[-1]
        cls_token = pos_embed[:, :1] if has_cls else None
        patch_pos = pos_embed[:, 1:] if has_cls else pos_embed
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos.float(), size=(new_grid, new_grid), mode="bicubic", align_corners=False).to(pos_embed.dtype)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
        new_pos_embed = torch.cat([cls_token, patch_pos], dim=1) if has_cls else patch_pos
        trunk.pos_embed = nn.Parameter(new_pos_embed)
    print(f"  Resized ViT pos_embed: {num_tokens} → {new_pos_embed.shape[1]} tokens ({old_grid}×{old_grid} → {new_grid}×{new_grid}, patch={patch_size})")


class OpenClipVisionEncoder(nn.Module):
    def __init__(
        self,
        model,
        image_size: int = 224,
        image_mode: str = "resize",
        resize_rings: int = 0,
        crop_rings: int = 0,
    ):
        super().__init__()
        self.model = model
        self.patch_size = _infer_vit_patch_size(model)
        self.resize_rings, self.crop_rings, self.image_size = resolve_image_rings(
            image_size=image_size,
            image_mode=image_mode,
            resize_rings=resize_rings,
            crop_rings=crop_rings,
            patch_size=self.patch_size,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.resize_rings > 0 or self.crop_rings > 0:
            image = apply_image_resize_crop(
                image,
                resize_rings=self.resize_rings,
                crop_rings=self.crop_rings,
                patch_size=self.patch_size,
            )
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
        # token_ids = token_ids.to(dtype=torch.int32)
        # eot_pos = token_ids.argmax(dim=-1, keepdim=True)
        # positions = torch.arange(
        #     token_ids.shape[-1], device=token_ids.device
        # ).unsqueeze(0)
        # mask = (positions <= eot_pos).to(token_ids.dtype)
        # token_ids = token_ids * mask
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
    clip_model: nn.Module, output_dir: str, max_text_len: int = 77,
    image_size: int = 224, image_mode: str = "resize",
    resize_rings: int = 0, crop_rings: int = 0,
) -> None:
    """Export image and text encoders to ONNX.

    clip_model must already be reparameterized and in eval mode.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(clip_model.parameters()).device

    patch_size = _infer_vit_patch_size(clip_model)
    resize_rings, crop_rings, final_image_size = resolve_image_rings(
        image_size=image_size,
        image_mode=image_mode,
        resize_rings=resize_rings,
        crop_rings=crop_rings,
        patch_size=patch_size,
    )
    if final_image_size != 224:
        print(
            f"\nResizing ViT positional embeddings for image_size={final_image_size} "
            f"(resize_rings={resize_rings}, crop_rings={crop_rings}, patch={patch_size})..."
        )
        _resize_vit_pos_embed(clip_model, final_image_size)

    image_encoder = OpenClipVisionEncoder(
        clip_model,
        image_size=image_size,
        image_mode=image_mode,
        resize_rings=resize_rings,
        crop_rings=crop_rings,
    ).eval()
    text_encoder = OpenClipTextEncoder(clip_model, max_text_len).eval()

    dummy_image = torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device)
    dummy_text = torch.randint(0, 49408, (1, 77), dtype=torch.int32, device=device)

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
        export_encoders_to_onnx(
            clip_model,
            output_dir_name,
            args.max_text_len,
            args.image_size,
            args.image_mode,
            args.resize,
            args.crop,
        )

    if args.checkpoint_path:
        print(f"Exported from checkpoint: {Path(args.checkpoint_path).resolve()}")


if __name__ == "__main__":
    main()

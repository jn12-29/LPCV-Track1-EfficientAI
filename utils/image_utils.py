from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

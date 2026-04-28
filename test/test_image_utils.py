from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from utils.image_utils import resolve_image_rings, apply_image_resize_crop


def test_resolve_rings_resize_only():
    r, c, final = resolve_image_rings(
        image_size=224, image_mode="resize",
        resize_rings=3, crop_rings=0, patch_size=16,
    )
    assert r == 3 and c == 0 and final == 128


def test_resolve_rings_crop_only():
    r, c, final = resolve_image_rings(
        image_size=224, image_mode="resize",
        resize_rings=0, crop_rings=1, patch_size=16,
    )
    assert r == 0 and c == 1 and final == 192


def test_resolve_rings_chain():
    r, c, final = resolve_image_rings(
        image_size=224, image_mode="resize",
        resize_rings=1, crop_rings=2, patch_size=16,
    )
    assert r == 1 and c == 2 and final == 128


def test_resolve_rings_noop():
    r, c, final = resolve_image_rings(
        image_size=224, image_mode="resize",
        resize_rings=0, crop_rings=0, patch_size=16,
    )
    assert r == 0 and c == 0 and final == 224


def test_apply_resize_shape():
    x = torch.ones(1, 3, 224, 224)
    out = apply_image_resize_crop(x, resize_rings=3, crop_rings=0, patch_size=16)
    assert out.shape == (1, 3, 128, 128)


def test_apply_crop_shape():
    x = torch.ones(1, 3, 224, 224)
    out = apply_image_resize_crop(x, resize_rings=0, crop_rings=1, patch_size=16)
    assert out.shape == (1, 3, 192, 192)


def test_apply_noop():
    x = torch.rand(1, 3, 224, 224)
    out = apply_image_resize_crop(x, resize_rings=0, crop_rings=0, patch_size=16)
    assert torch.equal(out, x)


def test_monkey_patch_preserves_model_structure():
    """Monkey-patching encode_image must not change named_modules or named_parameters."""
    class FakeModel(nn.Module):
        def encode_image(self, x: torch.Tensor) -> torch.Tensor:
            return x.flatten(1)

    model = FakeModel()
    modules_before = list(model.named_modules())
    params_before = list(model.named_parameters())

    resize_rings, crop_rings, patch_size = 1, 0, 16
    _orig = model.encode_image
    def _enc(image: torch.Tensor, _o=_orig, _r=resize_rings, _c=crop_rings, _p=patch_size) -> torch.Tensor:
        image = apply_image_resize_crop(image, resize_rings=_r, crop_rings=_c, patch_size=_p)
        return _o(image)
    model.encode_image = _enc

    assert list(model.named_modules()) == modules_before
    assert list(model.named_parameters()) == params_before

    x = torch.ones(1, 3, 224, 224)
    out = model.encode_image(x)
    # resize_rings=1, patch=16 → 224-32=192; flatten gives 3*192*192
    assert out.shape == (1, 3 * 192 * 192)

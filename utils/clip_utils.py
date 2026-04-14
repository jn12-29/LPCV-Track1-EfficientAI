from __future__ import annotations

from typing import Optional, Tuple
import torch
import open_clip


def _load_clip(
    model_name: str,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    pretrained: Optional[str] = None,
) -> Tuple:
    """Load a CLIP model, optionally from a fine-tuned checkpoint.

    Args:
        model_name: open_clip model name, e.g. 'MobileCLIP2-S0'.
        device: target device.
        checkpoint_path: path to a fine-tuned .pt file (optional).
        pretrained: explicit pretrained tag; auto-selected when None.

    Returns:
        (model, preprocess, tokenizer)
    """
    print(f"Loading CLIP model '{model_name}'...")
    available_models_tuple = open_clip.list_pretrained()
    available_model_names = {name for name, _ in available_models_tuple}
    if model_name not in available_model_names:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {open_clip.list_pretrained()}"
        )

    if pretrained:
        pretrained_tag = pretrained
    else:
        available_tags = [
            ckpt for name, ckpt in available_models_tuple if name == model_name
        ]
        pretrained_tag = available_tags[0]

    # MobileCLIP2 variants expect [0, 1] RGB — disable internal normalization so
    # the competition-style preprocessing (resize + /255 only) passes through unchanged.
    model_kwargs: dict = {}
    model_kwargs = {"image_mean": (0.0, 0.0, 0.0), "image_std": (1.0, 1.0, 1.0)}

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_tag, **model_kwargs
    )

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")

    model.eval().to(device)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return model, preprocess, tokenizer

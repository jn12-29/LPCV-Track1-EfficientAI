from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
import torch
import open_clip

if TYPE_CHECKING:
    from utils.qat_utils import QATConfig


def _load_clip(
    model_name: str,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    pretrained: Optional[str] = None,
    qat_config: Optional["QATConfig"] = None,
) -> Tuple:
    """Load a CLIP model, optionally from a fine-tuned or QAT checkpoint.

    When qat_config.enabled is True the model is reparameterized and wrapped
    with AIMET QuantizationSimModel.  The returned model is sim.model and
    carries _qat_sim / _qat_needs_calibration attributes.

    Checkpoint auto-detection:
      - Regular checkpoint (no "qat_enabled" key): weights loaded before QAT wrap.
      - QAT checkpoint ("qat_enabled": True): weights and encodings loaded after
        QAT wrap so the reparameterized structure matches.

    Args:
        model_name: open_clip model name, e.g. 'MobileCLIP2-S0'.
        device: target device.
        checkpoint_path: path to a .pt file (regular or QAT, auto-detected).
        pretrained: explicit pretrained tag; auto-selected when None.
        qat_config: QATConfig instance; when provided the model is QAT-wrapped.

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
    model_kwargs: dict = {"image_mean": (0.0, 0.0, 0.0), "image_std": (1.0, 1.0, 1.0)}

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_tag, **model_kwargs
    )

    # --- Checkpoint loading ---
    qat_state_dict = None
    qat_encodings = None

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        is_qat_ckpt = isinstance(checkpoint, dict) and checkpoint.get("qat_enabled", False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        if is_qat_ckpt:
            # Defer loading until after QAT wrap (structure is reparameterized).
            qat_state_dict = state_dict
            qat_encodings = checkpoint.get("qat_encodings")
            print(f"Detected QAT checkpoint: {checkpoint_path}")
        else:
            # Regular checkpoint: load into base (non-reparameterized) model now.
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")

    model.eval().to(device)

    # --- QAT wrapping ---
    if qat_config is not None and qat_config.enabled:
        from utils.qat_utils import wrap_model_for_qat
        model = wrap_model_for_qat(model, qat_config, device, qat_encodings)

        if qat_state_dict is not None:
            # Load QAT weights into reparameterized + QuantSim model.
            missing, unexpected = model.load_state_dict(qat_state_dict, strict=False)
            print("Restored QAT model weights from checkpoint")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
    elif checkpoint_path and qat_state_dict is not None:
        # QAT checkpoint loaded without QAT mode: fall back to base model keys.
        from utils.qat_utils import extract_base_model_state_dict
        filtered = extract_base_model_state_dict(qat_state_dict, model)
        model.load_state_dict(filtered, strict=False)
        print("Loaded base weights from QAT checkpoint (QAT mode disabled)")

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return model, preprocess, tokenizer

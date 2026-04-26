from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
import torch
import torch.nn as nn
import open_clip

if TYPE_CHECKING:
    from utils.qat_utils import QATConfig


def _log_state_dict_load(label: str, missing: list, unexpected: list) -> None:
    print(label)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")


def _load_clip(
    model_name: str,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    pretrained: Optional[str] = None,
    qat_config: Optional["QATConfig"] = None,
    relu_image: bool = False,
    relu_text: bool = False,
) -> Tuple:
    """Load a CLIP model, optionally from a fine-tuned or QAT checkpoint.

    When qat_config.enabled is True the model is reparameterized and wrapped
    with AIMET QuantizationSimModel.  The returned model is sim.model and
    carries _qat_sim / _qat_needs_calibration attributes.

    Checkpoint auto-detection:
      - Regular checkpoint (no "qat_enabled" key): weights loaded before QAT wrap.
      - QAT checkpoint ("qat_enabled": True): weights and encodings loaded after
        QAT wrap so the reparameterized structure matches.
      - MLP-reconstruction checkpoint ("relu_blocks" key): ReLU structure applied
        before weight loading; relu_image / relu_text flags set accordingly.

    Args:
        model_name: open_clip model name, e.g. 'MobileCLIP2-S0'.
        device: target device.
        checkpoint_path: path to a .pt file (regular, QAT, or MLP-reconstruction,
            auto-detected).
        pretrained: explicit pretrained tag; auto-selected when None.
        qat_config: QATConfig instance; when provided the model is QAT-wrapped.
        relu_image: replace GELU with ReLU in all visual-encoder MLP blocks.
        relu_text: replace GELU with ReLU in all text-encoder MLP blocks.

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
    elif checkpoint_path:
        pretrained_tag = None
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
    is_relu_ckpt = False

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        is_qat_ckpt = isinstance(checkpoint, dict) and checkpoint.get("qat_enabled", False)
        is_relu_ckpt = isinstance(checkpoint, dict) and "relu_blocks" in checkpoint
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        # MLP-reconstruction checkpoint: apply ReLU structure before loading weights.
        if is_relu_ckpt:
            from mlp_reconstruction.mlp_blocks import apply_relu_blocks
            relu_labels = checkpoint["relu_blocks"]
            ckpt_relu_image = any(lbl.startswith("visual[") for lbl in relu_labels)
            ckpt_relu_text = any(lbl.startswith("text[") for lbl in relu_labels)
            # Warn on mismatches between checkpoint and caller flags.
            if relu_image and not ckpt_relu_image:
                print("WARNING: --relu-image set but checkpoint has no visual ReLU blocks; "
                      "fresh ReLU will be applied without distillation recovery.")
            if relu_text and not ckpt_relu_text:
                print("WARNING: --relu-text set but checkpoint has no text ReLU blocks; "
                      "fresh ReLU will be applied without distillation recovery.")
            if ckpt_relu_image and not relu_image:
                print("WARNING: checkpoint contains visual ReLU blocks but --relu-image was not set; "
                      "applying from checkpoint.")
            if ckpt_relu_text and not relu_text:
                print("WARNING: checkpoint contains text ReLU blocks but --relu-text was not set; "
                      "applying from checkpoint.")
            apply_relu_blocks(model, relu_labels)
            # OR with CLI flags so both sources are honoured.
            relu_image = relu_image or ckpt_relu_image
            relu_text = relu_text or ckpt_relu_text
            print(f"Detected MLP-reconstruction checkpoint: {len(relu_labels)} ReLU blocks "
                  f"(image={relu_image}, text={relu_text})")

        if is_qat_ckpt:
            # Defer loading until after QAT wrap (structure is reparameterized).
            qat_state_dict = state_dict
            qat_encodings = checkpoint.get("qat_encodings")
            print(f"Detected QAT checkpoint: {checkpoint_path}")
        else:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            _log_state_dict_load(f"Loaded checkpoint from {checkpoint_path}", missing, unexpected)

    # Track relu_blocks on model for downstream use.
    # For relu checkpoints, apply_relu_blocks already set the correct structure — collect labels
    # from actual activation state rather than re-applying, which would wrongly override
    # intentionally-excluded blocks (e.g. --no-relu-stem).
    # For plain checkpoints with relu_image/relu_text flags, apply and collect.
    if is_relu_ckpt:
        from mlp_reconstruction.mlp_blocks import iter_mlp_blocks, is_relu_active
        model._relu_blocks = [info.label for info in iter_mlp_blocks(model) if is_relu_active(info)]
    elif relu_image or relu_text:
        from mlp_reconstruction.mlp_blocks import iter_mlp_blocks, replace_gelu_with_relu, is_relu_active
        labels = []
        for info in iter_mlp_blocks(model):
            if (info.encoder == "visual" and relu_image) or (info.encoder == "text" and relu_text):
                replace_gelu_with_relu(info)
            if is_relu_active(info):
                labels.append(info.label)
        model._relu_blocks = labels
        print(f"Applied ReLU activations: image={relu_image}, text={relu_text}")

    model.eval().to(device)

    # --- QAT wrapping ---
    if qat_config is not None and qat_config.enabled:
        from utils.qat_utils import wrap_model_for_qat
        model = wrap_model_for_qat(model, qat_config, device, qat_encodings)
        if qat_state_dict is not None:
            missing, unexpected = model.load_state_dict(qat_state_dict, strict=False)
            _log_state_dict_load("Restored QAT model weights from checkpoint", missing, unexpected)
    elif checkpoint_path and qat_state_dict is not None:
        # QAT checkpoint without explicit qat_config: auto-reconstruct QATConfig from checkpoint.
        from utils.qat_utils import wrap_model_for_qat, qat_config_from_checkpoint
        auto_qat_config = qat_config_from_checkpoint(checkpoint)
        model = wrap_model_for_qat(model, auto_qat_config, device, qat_encodings)
        missing, unexpected = model.load_state_dict(qat_state_dict, strict=False)
        _log_state_dict_load("Loaded QAT checkpoint with quantization enabled (auto-detected)", missing, unexpected)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return model, preprocess, tokenizer

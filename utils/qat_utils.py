from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Exclusion rule Protocol and concrete implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class ExclusionRule(Protocol):
    """Structural protocol: any object with should_exclude(name, module) -> bool."""

    def should_exclude(self, name: str, module: nn.Module) -> bool: ...


class GroupConvRule:
    """Exclude nn.Conv2d where groups > 1 (any grouped or depthwise conv)."""

    def should_exclude(self, name: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Conv2d) and module.groups > 1


class DepthwiseConvRule:
    """Exclude nn.Conv2d where groups == in_channels (pure depthwise conv)."""

    def should_exclude(self, name: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Conv2d) and module.groups == module.in_channels


class NamePatternRule:
    """Exclude modules whose fully-qualified name matches any of the given regex patterns.

    Uses re.search (partial match): r"stem" matches "visual.trunk.stem.conv_dw".
    """

    def __init__(self, patterns: List[str]) -> None:
        self._compiled = [re.compile(p) for p in patterns]

    def should_exclude(self, name: str, module: nn.Module) -> bool:
        return any(pat.search(name) for pat in self._compiled)


class TypeRule:
    """Exclude modules that are instances of any of the given types."""

    def __init__(self, *types: type) -> None:
        self._types = tuple(types)

    def should_exclude(self, name: str, module: nn.Module) -> bool:
        return isinstance(module, self._types)


@dataclass
class QATConfig:
    enabled: bool = True
    weight_bw: int = 8
    act_bw: int = 8
    quant_scheme: str = "tf_enhanced"
    calib_samples: int = 1024
    exclusion_rules: List[ExclusionRule] = field(default_factory=list)


def wrap_model_for_qat(
    model: nn.Module,
    config: QATConfig,
    device: torch.device,
    qat_encodings: Optional[str] = None,
) -> nn.Module:
    """Reparameterize then wrap model with AIMET QuantizationSimModel.

    Always calls reparameterize_model() first so the structure is consistent
    whether starting from pretrained, a regular checkpoint, or a QAT checkpoint.
    When qat_encodings is provided the calibration step can be skipped.

    Returns sim.model with _qat_sim and _qat_needs_calibration attached.
    """
    from timm.utils import reparameterize_model
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.common.defs import QuantScheme
    from aimet_torch.nn import QuantizationMixin
    import timm.layers.norm as _timm_norm
    import open_clip.transformer as _oc_transformer

    # AIMET v2 requires registration for subclasses of nn.LayerNorm it doesn't know.
    # Ignoring them keeps LayerNorm at fp32, which is acceptable (often excluded in practice).
    QuantizationMixin.ignore(_timm_norm.LayerNorm)
    QuantizationMixin.ignore(_oc_transformer.LayerNorm)

    _SCHEME_MAP = {
        "tf_enhanced": QuantScheme.post_training_tf_enhanced,
        "percentile": QuantScheme.post_training_percentile,
    }

    model = reparameterize_model(model)
    model.eval()

    dummy_image = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device)
    dummy_text = torch.zeros(1, 77, dtype=torch.long, device=device)

    sim = QuantizationSimModel(
        model=model,
        dummy_input=(dummy_image, dummy_text),
        quant_scheme=_SCHEME_MAP.get(config.quant_scheme, QuantScheme.post_training_tf_enhanced),
        default_output_bw=config.act_bw,
        default_param_bw=config.weight_bw,
    )

    apply_exclusion_rules(sim, config.exclusion_rules)

    needs_calibration = True
    if qat_encodings is not None:
        try:
            sim.load_encodings(json.loads(qat_encodings), strict=False, partial=False)
            needs_calibration = False
            print("Loaded QAT encodings from checkpoint (skipping calibration)")
        except Exception as exc:
            print(f"Warning: could not load QAT encodings ({exc}), will re-calibrate")

    sim.model._qat_sim = sim
    sim.model._qat_needs_calibration = needs_calibration
    return sim.model


def apply_exclusion_rules(sim, rules: List[ExclusionRule]) -> None:
    """Remove fake-quant wrappers from modules matching any rule.

    Must be called after QuantizationSimModel creation, before compute_encodings.
    Empty rules list is a safe no-op.
    """
    if not rules:
        return

    to_exclude = [
        module
        for name, module in sim.model.named_modules()
        if any(rule.should_exclude(name, module) for rule in rules)
    ]

    if to_exclude:
        print(f"[QAT] Excluding {len(to_exclude)} module(s) from quantization.")
        sim.exclude_layers_from_quantization(to_exclude)


def calibrate_quantsim(
    sim,
    dataloader,
    n_samples: int,
    device: torch.device,
) -> None:
    """Run compute_encodings() with up to n_samples calibration samples."""

    def calibration_fn(model, _):
        model.eval()
        seen = 0
        with torch.no_grad():
            for batch in dataloader:
                if seen >= n_samples:
                    break
                images = batch["images"].to(device, non_blocking=True)
                tokens = batch["positive_tokens"].to(device, non_blocking=True)
                model(images, tokens)
                seen += images.shape[0]

    sim.compute_encodings(calibration_fn, forward_pass_callback_args=None)
    sim.model.train()


def get_qat_encodings_json(sim) -> str:
    """Export quantization encodings to a JSON string (no ONNX needed)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # AIMET 1.x exposes save_encodings_to_json directly (no ONNX export).
        if hasattr(sim, "save_encodings_to_json"):
            sim.save_encodings_to_json(tmp_dir, "encodings")
            return (Path(tmp_dir) / "encodings.json").read_text()
        # Fallback: full export (also writes ONNX, which we discard).
        dummy_image = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
        dummy_text = torch.zeros(1, 77, dtype=torch.long)
        sim.export(tmp_dir, "model", (dummy_image, dummy_text))
        return (Path(tmp_dir) / "model.encodings").read_text()


def extract_base_model_state_dict(qat_state_dict: dict, base_model: nn.Module) -> dict:
    """Filter a QAT state dict to only the keys present in base_model.

    AIMET's QuantSim wrappers delegate state_dict() to the wrapped module so
    key names are identical, but this guard handles any version differences.
    """
    base_keys = set(base_model.state_dict().keys())
    return {k: v for k, v in qat_state_dict.items() if k in base_keys}

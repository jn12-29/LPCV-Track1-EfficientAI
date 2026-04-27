from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn as nn


@dataclass
class MLPBlockInfo:
    label: str                               # 'text[0]', 'visual[s0b0]', 'visual[b0]', 'visual[stem]'
    mlp: nn.Module
    kind: Literal['text_sequential', 'fastvit_convmlp', 'vit_mlp', 'conv_stem']
    fc1: nn.Module | None                    # None for conv_stem
    fc2: nn.Module | None                    # None for conv_stem
    act_attr: str                            # 'gelu' / 'act'; '' for conv_stem
    encoder: str                             # 'text' or 'visual'

    # ------------------------------------------------------------------
    # Activation management
    # ------------------------------------------------------------------

    def set_activation(self, act_cls: type[nn.Module]) -> None:
        """Replace activation in-place (conv_stem: all bn.act layers; others: single attr)."""
        if self.kind == 'conv_stem':
            for layer in self.mlp:
                if (hasattr(layer, 'bn') and hasattr(layer.bn, 'act')
                        and isinstance(layer.bn.act, (nn.GELU, nn.ReLU))):
                    layer.bn.act = act_cls()
        else:
            setattr(self.mlp, self.act_attr, act_cls())

    def is_relu(self) -> bool:
        """Return True if the block's activation is currently ReLU."""
        if self.kind == 'conv_stem':
            return any(
                hasattr(l, 'bn') and hasattr(l.bn, 'act') and isinstance(l.bn.act, nn.ReLU)
                for l in self.mlp
            )
        return isinstance(getattr(self.mlp, self.act_attr), nn.ReLU)

    # ------------------------------------------------------------------
    # Hook registration for input collection
    # ------------------------------------------------------------------

    def register_x_hook(self, callback: Callable[[torch.Tensor], None]):
        """Register a hook that fires callback(X) on each forward pass.

        fastvit_convmlp: X = output of mlp.conv (direct input to fc1).
        All others: X = raw MLP input (pre-hook on mlp).
        """
        if self.kind == 'fastvit_convmlp':
            return self.mlp.conv.register_forward_hook(
                lambda m, inp, out: callback(out)
            )
        return self.mlp.register_forward_pre_hook(
            lambda m, inp: callback(inp[0])
        )

    # ------------------------------------------------------------------
    # Forward / parameter access
    # ------------------------------------------------------------------

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """MLP forward using the currently installed activation."""
        if self.fc1 is None:  # conv_stem: whole Sequential is one unit
            return self.mlp(X)
        return self.fc2(getattr(self.mlp, self.act_attr)(self.fc1(X)))

    def trainable_params(self) -> list[nn.Parameter]:
        """Parameters to optimize during distillation."""
        if self.fc1 is None:  # conv_stem
            return list(self.mlp.parameters())
        return list(self.fc1.parameters()) + list(self.fc2.parameters())

    # ------------------------------------------------------------------
    # Weight snapshot for revert
    # ------------------------------------------------------------------

    def save_weights(self) -> dict:
        if self.fc1 is None:  # conv_stem
            return {'mlp': {k: v.clone() for k, v in self.mlp.state_dict().items()}}
        return {
            'fc1': {k: v.clone() for k, v in self.fc1.state_dict().items()},
            'fc2': {k: v.clone() for k, v in self.fc2.state_dict().items()},
        }

    def restore_weights(self, saved: dict) -> None:
        if self.fc1 is None:  # conv_stem
            self.mlp.load_state_dict(saved['mlp'])
        else:
            self.fc1.load_state_dict(saved['fc1'])
            self.fc2.load_state_dict(saved['fc2'])


# ----------------------------------------------------------------------
# Block enumeration
# ----------------------------------------------------------------------

def iter_mlp_blocks(model: nn.Module) -> list[MLPBlockInfo]:
    blocks: list[MLPBlockInfo] = []
    # Text encoder (same for all model variants)
    for i, rb in enumerate(model.text.transformer.resblocks):
        blocks.append(MLPBlockInfo(
            label=f'text[{i}]',
            mlp=rb.mlp,
            kind='text_sequential',
            fc1=rb.mlp.c_fc,
            fc2=rb.mlp.c_proj,
            act_attr='gelu',
            encoder='text',
        ))
    # Visual encoder: FastVit (S0/S2) vs ViT (B)
    trunk = model.visual.trunk
    if hasattr(trunk, 'stages'):  # FastVit
        for si, stage in enumerate(trunk.stages):
            for bi, block in enumerate(stage.blocks):
                blocks.append(MLPBlockInfo(
                    label=f'visual[s{si}b{bi}]',
                    mlp=block.mlp,
                    kind='fastvit_convmlp',
                    fc1=block.mlp.fc1,
                    fc2=block.mlp.fc2,
                    act_attr='act',
                    encoder='visual',
                ))
    else:  # ViT (B)
        # ConvStem: distilled as one joint unit because individual Conv→BN→ReLU
        # layers have no downstream projection to compensate for GELU vs ReLU.
        if hasattr(trunk, 'patch_embed') and hasattr(trunk.patch_embed, 'backbone'):
            backbone = trunk.patch_embed.backbone
            backbone_has_bn_act = isinstance(backbone, nn.Sequential) and any(
                hasattr(l, 'bn') and hasattr(l.bn, 'act')
                and isinstance(l.bn.act, (nn.GELU, nn.ReLU))
                for l in backbone
            )
            if backbone_has_bn_act:
                blocks.append(MLPBlockInfo(
                    label='visual[stem]',
                    mlp=backbone,
                    kind='conv_stem',
                    fc1=None,
                    fc2=None,
                    act_attr='',
                    encoder='visual',
                ))
        for bi, block in enumerate(trunk.blocks):
            blocks.append(MLPBlockInfo(
                label=f'visual[b{bi}]',
                mlp=block.mlp,
                kind='vit_mlp',
                fc1=block.mlp.fc1,
                fc2=block.mlp.fc2,
                act_attr='act',
                encoder='visual',
            ))
    return blocks


def apply_relu_blocks(model: nn.Module, relu_labels: list[str]) -> None:
    """Apply ReLU to the listed blocks (called before loading a reconstructed state_dict)."""
    label_set = set(relu_labels)
    for info in iter_mlp_blocks(model):
        if info.label in label_set:
            info.set_activation(nn.ReLU)


# ----------------------------------------------------------------------
# Module-level wrappers — preserved for clip_utils.py compatibility
# ----------------------------------------------------------------------

def replace_gelu_with_relu(info: MLPBlockInfo) -> None:
    info.set_activation(nn.ReLU)


def restore_gelu(info: MLPBlockInfo) -> None:
    info.set_activation(nn.GELU)


def is_relu_active(info: MLPBlockInfo) -> bool:
    return info.is_relu()

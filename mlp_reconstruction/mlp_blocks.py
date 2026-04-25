from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch.nn as nn


@dataclass
class MLPBlockInfo:
    label: str                               # 'text[0]', 'visual[s0b0]', 'visual[b0]', 'visual[stem]'
    mlp: nn.Module
    kind: Literal['text_sequential', 'fastvit_convmlp', 'vit_mlp', 'conv_stem']
    fc1: nn.Module | None                    # None for conv_stem (optimize all mlp params jointly)
    fc2: nn.Module | None                    # None for conv_stem
    act_attr: str                            # 'gelu' (text) or 'act' (FastVit/ViT); unused for conv_stem
    encoder: str                             # 'text' or 'visual'


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
        # ConvStem (HybridEmbed backbone) — only in ViT-B; distilled as a single unit.
        # ConvStem[0/1] have GELU; ConvStem[2] (Conv 192→768, no act) acts as the compensating
        # projection — analogous to fc2 in MLP blocks. Joint distillation is necessary because
        # a single Conv→BN→ReLU has no downstream projection to compensate for GELU vs ReLU.
        if hasattr(trunk, 'patch_embed') and hasattr(trunk.patch_embed, 'backbone'):
            backbone = trunk.patch_embed.backbone
            # backbone is a ConvStem Sequential; add it as one joint block.
            # The condition guards against unexpected backbone variants that lack bn.act layers.
            backbone_has_bn_act = isinstance(backbone, nn.Sequential) and any(
                hasattr(l, 'bn') and hasattr(l.bn, 'act') and isinstance(l.bn.act, (nn.GELU, nn.ReLU))
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


def replace_gelu_with_relu(info: MLPBlockInfo) -> None:
    if info.kind == 'conv_stem':
        for layer in info.mlp:
            if hasattr(layer, 'bn') and hasattr(layer.bn, 'act') and isinstance(layer.bn.act, nn.GELU):
                layer.bn.act = nn.ReLU()
    else:
        setattr(info.mlp, info.act_attr, nn.ReLU())


def is_relu_active(info: MLPBlockInfo) -> bool:
    """Return True if this block's activation has been replaced with ReLU."""
    if info.kind == 'conv_stem':
        return any(
            hasattr(l, 'bn') and hasattr(l.bn, 'act') and isinstance(l.bn.act, nn.ReLU)
            for l in info.mlp
        )
    return isinstance(getattr(info.mlp, info.act_attr), nn.ReLU)


def apply_relu_blocks(model: nn.Module, relu_labels: list[str]) -> None:
    """Restore ReLU structure before loading a reconstructed checkpoint's state_dict."""
    label_set = set(relu_labels)
    for info in iter_mlp_blocks(model):
        if info.label in label_set:
            replace_gelu_with_relu(info)

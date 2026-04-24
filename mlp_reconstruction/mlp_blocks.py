from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch.nn as nn


@dataclass
class MLPBlockInfo:
    label: str                               # 'text[0]', 'visual[s0b0]', 'visual[b0]'
    mlp: nn.Module
    kind: Literal['text_sequential', 'fastvit_convmlp', 'vit_mlp']
    fc1: nn.Module                           # nn.Linear (text/ViT) or nn.Conv2d (FastVit)
    fc2: nn.Module
    act_attr: str                            # 'gelu' (text) or 'act' (FastVit/ViT)
    encoder: str                             # 'text' or 'visual'


def iter_mlp_blocks(model: nn.Module, model_name: str) -> list[MLPBlockInfo]:
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
    setattr(info.mlp, info.act_attr, nn.ReLU())


def apply_relu_blocks(model: nn.Module, model_name: str, relu_labels: list[str]) -> None:
    """Restore ReLU structure before loading a reconstructed checkpoint's state_dict."""
    label_set = set(relu_labels)
    for info in iter_mlp_blocks(model, model_name):
        if info.label in label_set:
            replace_gelu_with_relu(info)

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from utils.preprocess import preprocess_image


class VGCalibrationLoader:
    """Load image and text calibration batches from vg_llm_contrastive.jsonl."""

    def __init__(
        self,
        jsonl_path: str,
        project_root: str,
        tokenizer,
        n_calib: int = 1024,
        batch_size: int = 32,
    ) -> None:
        self.project_root = Path(project_root)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        records: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                if len(records) >= n_calib:
                    break
                row = json.loads(line)
                if row.get('positives') and row.get('image_path'):
                    records.append(row)
        self.records = records

    def get_image_batches(self) -> list[torch.Tensor]:
        """Returns list of (B,3,224,224) float32 image tensors."""
        all_imgs: list[torch.Tensor] = []
        for rec in self.records:
            img = Image.open(self.project_root / rec['image_path']).convert('RGB')
            all_imgs.append(preprocess_image(img))
        batches: list[torch.Tensor] = []
        for i in range(0, len(all_imgs), self.batch_size):
            batches.append(torch.stack(all_imgs[i : i + self.batch_size]))
        return batches

    def get_text_batches(self) -> list[torch.Tensor]:
        """Returns list of tokenized (B,77) int64 tensors."""
        texts = [rec['positives'][0] for rec in self.records]
        batches: list[torch.Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            batches.append(self.tokenizer(texts[i : i + self.batch_size]))
        return batches


@torch.no_grad()
def collect_mlp_io(
    model: nn.Module,
    info,
    image_batches: list[torch.Tensor] | None,
    text_batches: list[torch.Tensor] | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect (X_all, O_all) for an MLP block via forward hooks.

    For fastvit_convmlp: X_all = conv_out (direct input to fc1).
    For text_sequential / vit_mlp: X_all = raw MLP input.
    Returns CPU tensors.
    """
    X_list: list[torch.Tensor] = []
    O_list: list[torch.Tensor] = []

    if info.kind == 'fastvit_convmlp':
        h_x = info.mlp.conv.register_forward_hook(
            lambda m, a, o: X_list.append(o.detach().cpu())
        )
    else:
        h_x = info.mlp.register_forward_pre_hook(
            lambda m, a: X_list.append(a[0].detach().cpu())
        )
    h_o = info.mlp.register_forward_hook(
        lambda m, a, o: O_list.append(o.detach().cpu())
    )

    model.eval()
    batches = text_batches if info.encoder == 'text' else image_batches
    try:
        for batch in batches:
            batch = batch.to(device)
            if info.encoder == 'text':
                model.encode_text(batch)
            else:
                model.encode_image(batch)
    finally:
        h_x.remove()
        h_o.remove()
    return torch.cat(X_list, dim=0), torch.cat(O_list, dim=0)

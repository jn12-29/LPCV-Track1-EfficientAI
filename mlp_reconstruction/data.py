from __future__ import annotations

import json
from pathlib import Path

import torch
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
                if row.get("positives") and row.get("image_path"):
                    records.append(row)
        self.records = records

    def get_image_batches(self) -> list[torch.Tensor]:
        """Returns list of (B, 3, 224, 224) float32 image tensors."""
        all_imgs = [
            preprocess_image(
                Image.open(self.project_root / rec["image_path"]).convert("RGB")
            )
            for rec in self.records
        ]
        return [
            torch.stack(all_imgs[i : i + self.batch_size])
            for i in range(0, len(all_imgs), self.batch_size)
        ]

    def get_text_batches(self) -> list[torch.Tensor]:
        """Returns list of tokenized (B, 77) int32 tensors."""
        texts = [rec["positives"][0] for rec in self.records]
        return [
            self.tokenizer(
                list(texts[i : i + self.batch_size]),
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )["input_ids"]
            for i in range(0, len(texts), self.batch_size)
        ]

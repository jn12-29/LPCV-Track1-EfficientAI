"""PTQ dataset utilities for MobileCLIP2-B.

Builds deterministic calibration/validation splits from contrastive JSONL:
  - calibration: random 1k samples (default)
  - validation : random 100 samples from the remaining pool (default)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open_clip
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.preprocess import preprocess_image


def load_jsonl_records(
    jsonl_path: str | Path,
    image_base_dir: str | Path,
    max_records: Optional[int] = None,
) -> List[Dict]:
    """Read JSONL lines and resolve image paths.

    Each line is expected to contain at least ``image_path`` and ``positives``.
    """
    jsonl_path = Path(jsonl_path)
    image_base_dir = Path(image_base_dir)
    records: List[Dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img_path = image_base_dir / rec["image_path"]
            if not img_path.exists():
                continue
            rec["_resolved_path"] = str(img_path)
            records.append(rec)
            if max_records and len(records) >= max_records:
                break
    return records


def split_calib_val(
    records: List[Dict],
    calib_size: int = 1000,
    val_size: int = 100,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    if calib_size <= 0 or val_size <= 0:
        raise ValueError("calib_size and val_size must be positive.")
    if len(records) < calib_size + val_size:
        raise ValueError(
            "Not enough records for requested split: "
            f"need {calib_size + val_size}, got {len(records)}."
        )

    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    calib = pool[:calib_size]
    val = pool[calib_size : calib_size + val_size]
    return calib, val


# ---------------------------------------------------------------------------
# ONNX Runtime CalibrationDataReader implementations
# ---------------------------------------------------------------------------

class ImageCalibReader(CalibrationDataReader):
    """Feeds preprocessed images to the image encoder during calibration."""

    def __init__(self, records: List[Dict]):
        self.records = records
        self.idx = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.idx >= len(self.records):
            return None
        rec = self.records[self.idx]
        self.idx += 1
        img = Image.open(rec["_resolved_path"]).convert("RGB")
        tensor = preprocess_image(img).unsqueeze(0).numpy().astype(np.float32)
        return {"image": tensor}


class TextCalibReader(CalibrationDataReader):
    """Feeds tokenised texts to the text encoder during calibration."""

    def __init__(self, records: List[Dict], tokenizer):
        self.texts: List[str] = []
        for rec in records:
            self.texts.extend(rec.get("positives", []))
        if not self.texts:
            raise ValueError("No positive texts found in calibration records.")
        self.tokenizer = tokenizer
        self.idx = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.idx >= len(self.texts):
            return None
        tok = self.tokenizer(self.texts[self.idx]).numpy().astype(np.int64)
        self.idx += 1
        return {"text": tok}

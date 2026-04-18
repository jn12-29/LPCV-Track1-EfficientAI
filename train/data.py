from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.preprocess import preprocess_image
from train.train_utils import log_message


class ContrastiveRecordDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        max_records: Optional[int] = None,
        max_positives_per_image: Optional[int] = None,
        max_hard_negatives_per_image: Optional[int] = None,
        shuffle_texts_on_load: bool = False,
        shuffle_seed: Optional[int] = None,
        log_progress: bool = True,
    ) -> None:
        self.jsonl_path = Path(jsonl_path).resolve()
        self.max_records = max_records
        self.max_positives_per_image = max_positives_per_image
        self.max_hard_negatives_per_image = max_hard_negatives_per_image
        self.shuffle_texts_on_load = shuffle_texts_on_load
        self.shuffle_seed = shuffle_seed
        self.log_progress = log_progress
        self.records = self._load_records()

    def _load_records(self) -> List[Dict[str, object]]:
        records = []
        rng = (
            random.Random(self.shuffle_seed) if self.shuffle_seed is not None else None
        )
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                record = json.loads(line)

                positives = list(record["positives"])
                hard_negatives = list(record["hard_negatives"])

                if self.shuffle_texts_on_load:
                    if rng is None:
                        random.shuffle(positives)
                        random.shuffle(hard_negatives)
                    else:
                        rng.shuffle(positives)
                        rng.shuffle(hard_negatives)

                if self.max_positives_per_image is not None:
                    positives = positives[: self.max_positives_per_image]
                if self.max_hard_negatives_per_image is not None:
                    hard_negatives = hard_negatives[: self.max_hard_negatives_per_image]

                records.append(
                    {
                        "image_path": record["image_path"],
                        "positives": positives,
                        "hard_negatives": hard_negatives,
                    }
                )

                if self.max_records is not None and len(records) >= self.max_records:
                    break

                if self.log_progress and line_idx % 50000 == 0:
                    log_message(f"Loaded {line_idx} rows from {self.jsonl_path}")

        if not records:
            raise ValueError(f"No usable training records found in {self.jsonl_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        image_tensor = preprocess_image(image)
        return {
            "image": image_tensor,
            "positives": list(record["positives"]),
            "hard_negatives": list(record["hard_negatives"]),
        }


def choose_texts(
    texts: Sequence[str],
    count: int,
    strategy: str,
) -> List[str]:
    if not texts or count <= 0:
        return []
    texts = list(texts)
    if strategy == "first":
        if len(texts) >= count:
            return texts[:count]
        repeats = [texts[i % len(texts)] for i in range(count)]
        return repeats
    if strategy == "random":
        if len(texts) >= count:
            return random.sample(texts, count)
        return [random.choice(texts) for _ in range(count)]
    raise ValueError(f"Unsupported sampling strategy: {strategy}")


def create_collate_fn(tokenizer, num_hard_negatives: int, text_sampling: str):
    # num_hard_negatives == 0: use all hard negatives in the record (dynamic per batch)
    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch], dim=0)

        positive_texts = [
            choose_texts(item["positives"], 1, text_sampling)[0] for item in batch
        ]
        positive_tokens = tokenizer(positive_texts)

        use_all = num_hard_negatives == 0
        effective_count = (
            max(len(item["hard_negatives"]) for item in batch)
            if use_all
            else num_hard_negatives
        )

        hard_negative_texts = []
        hard_negative_mask = []
        for item in batch:
            negatives = item["hard_negatives"]
            if use_all:
                sampled = list(negatives)
            else:
                sampled = (
                    choose_texts(negatives, num_hard_negatives, text_sampling)
                    if negatives
                    else []
                )

            row_texts = []
            row_mask = []
            for text in sampled:
                row_texts.append(text)
                row_mask.append(1.0)

            while len(row_texts) < effective_count:
                row_texts.append("")
                row_mask.append(0.0)

            hard_negative_texts.extend(row_texts)
            hard_negative_mask.append(row_mask)

        if effective_count > 0:
            flat_negative_tokens = tokenizer(hard_negative_texts)
            negative_tokens = flat_negative_tokens.view(len(batch), effective_count, -1)
            negative_mask = torch.tensor(hard_negative_mask, dtype=torch.float32)
        else:
            negative_tokens = torch.empty((len(batch), 0, 77), dtype=torch.long)
            negative_mask = torch.empty((len(batch), 0), dtype=torch.float32)

        return {
            "images": images,
            "positive_tokens": positive_tokens,
            "negative_tokens": negative_tokens,
            "negative_mask": negative_mask,
        }

    return collate_fn

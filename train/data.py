from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.preprocess import preprocess_image
from train.train_utils import log_message


def load_jsonl_records(
    jsonl_path: str | Path,
    max_records: Optional[int] = None,
    max_positives_per_image: Optional[int] = None,
    max_hard_negatives_per_image: Optional[int] = None,
    shuffle_texts_on_load: bool = False,
    shuffle_seed: Optional[int] = None,
    log_progress: bool = True,
) -> List[Dict[str, object]]:
    """Load records from a JSONL file and return them as a plain list."""
    records: List[Dict[str, object]] = []
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    path = Path(jsonl_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            record = json.loads(line)
            positives = list(record["positives"])
            hard_negatives = list(record["hard_negatives"])
            if shuffle_texts_on_load:
                if rng is None:
                    random.shuffle(positives)
                    random.shuffle(hard_negatives)
                else:
                    rng.shuffle(positives)
                    rng.shuffle(hard_negatives)
            if max_positives_per_image is not None:
                positives = positives[:max_positives_per_image]
            if max_hard_negatives_per_image is not None:
                hard_negatives = hard_negatives[:max_hard_negatives_per_image]
            records.append(
                {
                    "image_path": record["image_path"],
                    "positives": positives,
                    "hard_negatives": hard_negatives,
                }
            )
            if max_records is not None and len(records) >= max_records:
                break
            if log_progress and line_idx % 50000 == 0:
                log_message(f"Loaded {line_idx} rows from {path}")
    if not records:
        raise ValueError(f"No usable training records found in {path}")
    return records


def split_val_records(
    records: List[Dict],
    val_size: int,
    random_seed: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Split records into (train_records, val_records).

    Default: deterministic tail split (last val_size records become val).
    If random_seed is given: random split with a fixed seed.
    val_size=0 returns (records, []) unchanged.
    """
    if val_size <= 0:
        return records, []
    if val_size >= len(records):
        raise ValueError(
            f"val_size={val_size} must be smaller than total records={len(records)}"
        )
    if random_seed is not None:
        rng = random.Random(random_seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        val_records = [records[i] for i in indices[:val_size]]
        train_records = [records[i] for i in indices[val_size:]]
    else:
        train_records = records[:-val_size]
        val_records = records[-val_size:]
    return train_records, val_records


class ContrastiveRecordDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path | None = None,
        records: Optional[List[Dict[str, object]]] = None,
        max_records: Optional[int] = None,
        max_positives_per_image: Optional[int] = None,
        max_hard_negatives_per_image: Optional[int] = None,
        shuffle_texts_on_load: bool = False,
        shuffle_seed: Optional[int] = None,
        log_progress: bool = True,
    ) -> None:
        if records is not None:
            self.records = list(records)
        elif jsonl_path is not None:
            self.records = load_jsonl_records(
                jsonl_path=jsonl_path,
                max_records=max_records,
                max_positives_per_image=max_positives_per_image,
                max_hard_negatives_per_image=max_hard_negatives_per_image,
                shuffle_texts_on_load=shuffle_texts_on_load,
                shuffle_seed=shuffle_seed,
                log_progress=log_progress,
            )
        else:
            raise ValueError("Either jsonl_path or records must be provided")

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

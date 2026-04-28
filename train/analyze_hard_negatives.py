from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.clip_utils import _load_clip
from utils.data_utils import _batched
from utils.preprocess import preprocess_image


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path]):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return preprocess_image(image), idx


def _get_image_output_dim(model, device: torch.device) -> int:
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model.encode_image(dummy)
    return out.shape[-1]


@torch.no_grad()
def encode_all_images(
    model,
    image_paths: List[Path],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    output_dim = _get_image_output_dim(model, device)
    dataset = ImageDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if num_workers > 0 else None,
    )
    features = torch.zeros(len(image_paths), output_dim)
    for batch_images, indices in loader:
        batch_images = batch_images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            feats = model.encode_image(batch_images)
        feats = F.normalize(feats.float(), dim=-1).cpu()
        features[indices] = feats
    return features


@torch.no_grad()
def encode_all_texts(
    model, tokenizer, texts: List[str], device: torch.device, batch_size: int
) -> torch.Tensor:
    outputs = []
    for _, batch_texts in _batched(texts, batch_size):
        tokens = tokenizer(
            list(batch_texts),
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )["input_ids"].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            feats = model.encode_text(tokens)
        outputs.append(F.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze positive vs hard negative similarity distributions"
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="./build_datasets/data/dataset_raw_contrastive.jsonl",
    )
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--top-k-hard-cases", type=int, default=50)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model for faster inference",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_hard_negatives",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, tokenizer = _load_clip(
        model_name=args.model_name,
        device=device,
        checkpoint_path=args.checkpoint_path,
    )
    model.eval()

    if args.compile:
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)

    # --- Pass 1: read all records, collect unique texts and image paths ---
    print("Reading JSONL ...")
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if not record["hard_negatives"]:
                continue
            records.append(record)
            if args.max_records is not None and len(records) >= args.max_records:
                break

    if not records:
        raise ValueError("No valid records found.")

    print(f"Loaded {len(records)} records.")

    # Deduplicate texts globally — encode each unique text only once
    all_texts_ordered: List[str] = []
    text_to_idx: Dict[str, int] = {}
    for rec in records:
        for t in rec["positives"] + rec["hard_negatives"]:
            if t not in text_to_idx:
                text_to_idx[t] = len(all_texts_ordered)
                all_texts_ordered.append(t)

    image_paths = [Path(rec["image_path"]) for rec in records]

    print(f"Encoding {len(image_paths)} images ...")
    image_features = encode_all_images(
        model, image_paths, device, args.image_batch_size, args.num_workers
    )

    print(f"Encoding {len(all_texts_ordered)} unique texts (deduped) ...")
    text_features = encode_all_texts(
        model, tokenizer, all_texts_ordered, device, args.text_batch_size
    )

    # --- Pass 2: compute per-record stats from cached features ---
    positive_sims: List[float] = []
    negative_sims: List[float] = []
    per_record_stats: List[Dict[str, object]] = []

    for i, rec in enumerate(records):
        img_feat = image_features[i]

        pos_indices = [text_to_idx[t] for t in rec["positives"]]
        neg_indices = [text_to_idx[t] for t in rec["hard_negatives"]]

        pos_feats = text_features[pos_indices]
        neg_feats = text_features[neg_indices]

        pos_sims = torch.mv(pos_feats, img_feat).numpy()
        neg_sims = torch.mv(neg_feats, img_feat).numpy()

        positive_sims.extend(pos_sims.tolist())
        negative_sims.extend(neg_sims.tolist())

        min_pos = float(np.min(pos_sims))
        max_neg = float(np.max(neg_sims))
        gap = min_pos - max_neg

        hardest_neg_idx = int(np.argmax(neg_sims))
        weakest_pos_idx = int(np.argmin(pos_sims))
        image_path = Path(rec["image_path"])

        per_record_stats.append(
            {
                "image_id": image_path.name,
                "image_path": str(image_path),
                "num_positives": len(rec["positives"]),
                "num_hard_negatives": len(rec["hard_negatives"]),
                "mean_positive_similarity": float(np.mean(pos_sims)),
                "mean_hard_negative_similarity": float(np.mean(neg_sims)),
                "min_positive_similarity": min_pos,
                "max_hard_negative_similarity": max_neg,
                "gap_min_pos_minus_max_neg": gap,
                "weakest_positive_text": rec["positives"][weakest_pos_idx],
                "weakest_positive_similarity": float(pos_sims[weakest_pos_idx]),
                "hardest_negative_text": rec["hard_negatives"][hardest_neg_idx],
                "hardest_negative_similarity": float(neg_sims[hardest_neg_idx]),
                "positives": rec["positives"],
                "hard_negatives": rec["hard_negatives"],
            }
        )

        if (i + 1) % 100 == 0:
            print(f"Computed stats for {i + 1} records ...")

    stats_path = output_dir / "similarity_stats.jsonl"
    with stats_path.open("w", encoding="utf-8") as f:
        for item in per_record_stats:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    hard_cases = sorted(per_record_stats, key=lambda x: x["gap_min_pos_minus_max_neg"])[
        : args.top_k_hard_cases
    ]
    hard_cases_path = output_dir / "hard_cases.jsonl"
    with hard_cases_path.open("w", encoding="utf-8") as f:
        for item in hard_cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    plt.figure(figsize=(10, 6))
    plt.hist(positive_sims, bins=60, alpha=0.6, label="positive", density=True)
    plt.hist(negative_sims, bins=60, alpha=0.6, label="hard_negative", density=True)
    plt.xlabel("cosine similarity")
    plt.ylabel("density")
    plt.title("Positive vs Hard Negative Similarity Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "positive_vs_hard_negative_similarity.png", dpi=160)
    plt.close()

    gaps = [item["gap_min_pos_minus_max_neg"] for item in per_record_stats]
    plt.figure(figsize=(10, 6))
    plt.hist(gaps, bins=60, alpha=0.8, color="tab:red")
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("min_positive_similarity - max_hard_negative_similarity")
    plt.ylabel("count")
    plt.title("Gap Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "gap_distribution.png", dpi=160)
    plt.close()

    summary = {
        "num_records": len(per_record_stats),
        "num_positive_pairs": len(positive_sims),
        "num_hard_negative_pairs": len(negative_sims),
        "positive_similarity_mean": float(np.mean(positive_sims)),
        "positive_similarity_std": float(np.std(positive_sims)),
        "hard_negative_similarity_mean": float(np.mean(negative_sims)),
        "hard_negative_similarity_std": float(np.std(negative_sims)),
        "gap_mean": float(np.mean(gaps)),
        "gap_std": float(np.std(gaps)),
        "num_gap_below_zero": int(sum(g < 0 for g in gaps)),
        "ratio_gap_below_zero": float(sum(g < 0 for g in gaps) / len(gaps)),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Saved summary     → {summary_path}")
    print(f"Saved full stats  → {stats_path}")
    print(f"Saved hard cases  → {hard_cases_path}")
    print(f"Saved plots       → {output_dir}")


if __name__ == "__main__":
    main()

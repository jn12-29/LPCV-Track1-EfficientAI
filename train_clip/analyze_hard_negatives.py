from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.clip_utils import _load_clip
from utils.data_utils import _batched
from utils.preprocess import preprocess_image
from train_clip.record_utils import normalize_record, resolve_image_path


@torch.no_grad()
def encode_texts(
    model, tokenizer, texts: List[str], device: torch.device, batch_size: int
) -> torch.Tensor:
    outputs = []
    for _, batch_texts in _batched(list(texts), batch_size):
        text_tokens = tokenizer(list(batch_texts)).to(device)
        text_features = model.encode_text(text_tokens)
        outputs.append(F.normalize(text_features, dim=-1).cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_image(model, image_path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image_tensor)
    return F.normalize(image_features, dim=-1).cpu().squeeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze positive vs hard negative similarity distributions")
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="./build_datasets/data/dataset_raw_contrastive.jsonl",
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--top-k-hard-cases", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_hard_negatives",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl_path).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, tokenizer = _load_clip(
        model_name=args.model_name,
        device=device,
        checkpoint_path=args.checkpoint_path,
    )
    model.eval()

    positive_sims: List[float] = []
    negative_sims: List[float] = []
    per_record_stats: List[Dict[str, object]] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            raw_record = json.loads(line)
            record = normalize_record(raw_record)
            if record is None or not record["hard_negatives"]:
                continue

            image_path = resolve_image_path(record["image_path"], jsonl_path, repo_root)
            image_feature = encode_image(model, image_path, device)

            pos_features = encode_texts(
                model, tokenizer, record["positives"], device, args.text_batch_size
            )
            neg_features = encode_texts(
                model, tokenizer, record["hard_negatives"], device, args.text_batch_size
            )

            pos_sims = torch.mv(pos_features, image_feature).numpy()
            neg_sims = torch.mv(neg_features, image_feature).numpy()

            positive_sims.extend(pos_sims.tolist())
            negative_sims.extend(neg_sims.tolist())

            min_pos = float(np.min(pos_sims))
            max_neg = float(np.max(neg_sims))
            mean_pos = float(np.mean(pos_sims))
            mean_neg = float(np.mean(neg_sims))
            gap = min_pos - max_neg

            hardest_neg_idx = int(np.argmax(neg_sims))
            weakest_pos_idx = int(np.argmin(pos_sims))

            per_record_stats.append(
                {
                    "image_id": record["image_id"],
                    "image_path": str(image_path),
                    "num_positives": len(record["positives"]),
                    "num_hard_negatives": len(record["hard_negatives"]),
                    "mean_positive_similarity": mean_pos,
                    "mean_hard_negative_similarity": mean_neg,
                    "min_positive_similarity": min_pos,
                    "max_hard_negative_similarity": max_neg,
                    "gap_min_pos_minus_max_neg": gap,
                    "weakest_positive_text": record["positives"][weakest_pos_idx],
                    "weakest_positive_similarity": float(pos_sims[weakest_pos_idx]),
                    "hardest_negative_text": record["hard_negatives"][hardest_neg_idx],
                    "hardest_negative_similarity": float(neg_sims[hardest_neg_idx]),
                    "positives": record["positives"],
                    "hard_negatives": record["hard_negatives"],
                }
            )

            if args.max_records is not None and len(per_record_stats) >= args.max_records:
                break

            if len(per_record_stats) % 100 == 0:
                print(f"Processed {len(per_record_stats)} records...")

    if not per_record_stats:
        raise ValueError("No valid records processed.")

    stats_path = output_dir / "similarity_stats.jsonl"
    with stats_path.open("w", encoding="utf-8") as f:
        for item in per_record_stats:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    hard_cases = sorted(per_record_stats, key=lambda x: x["gap_min_pos_minus_max_neg"])[: args.top_k_hard_cases]
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
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved summary to: {summary_path}")
    print(f"Saved full stats to: {stats_path}")
    print(f"Saved hard cases to: {hard_cases_path}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()

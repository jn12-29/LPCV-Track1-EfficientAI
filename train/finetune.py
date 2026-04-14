from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import open_clip

from utils.clip_utils import _load_clip
from utils.preprocess import preprocess_image
from train.record_utils import dedupe_keep_order, normalize_record, resolve_image_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ContrastiveRecordDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        repo_root: str | Path | None = None,
        max_records: Optional[int] = None,
        max_positives_per_image: Optional[int] = None,
        max_hard_negatives_per_image: Optional[int] = None,
        shuffle_texts_on_load: bool = False,
    ) -> None:
        self.jsonl_path = Path(jsonl_path).resolve()
        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self.max_records = max_records
        self.max_positives_per_image = max_positives_per_image
        self.max_hard_negatives_per_image = max_hard_negatives_per_image
        self.shuffle_texts_on_load = shuffle_texts_on_load
        self.records = self._load_records()

    def _load_records(self) -> List[Dict[str, object]]:
        records = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                raw_record = json.loads(line)
                record = normalize_record(raw_record)
                if record is None:
                    continue

                positives = list(record["positives"])
                hard_negatives = list(record["hard_negatives"])

                if self.shuffle_texts_on_load:
                    random.shuffle(positives)
                    random.shuffle(hard_negatives)

                if self.max_positives_per_image is not None:
                    positives = positives[: self.max_positives_per_image]
                if self.max_hard_negatives_per_image is not None:
                    hard_negatives = hard_negatives[: self.max_hard_negatives_per_image]

                resolved_image_path = resolve_image_path(
                    image_path=record["image_path"],
                    jsonl_path=self.jsonl_path,
                    repo_root=self.repo_root,
                )

                records.append(
                    {
                        "image_path": str(resolved_image_path),
                        "image_id": record["image_id"],
                        "positives": positives,
                        "hard_negatives": hard_negatives,
                        "challenge_tags": record["challenge_tags"],
                    }
                )

                if self.max_records is not None and len(records) >= self.max_records:
                    break

                if line_idx % 50000 == 0:
                    print(f"Loaded {line_idx} rows from {self.jsonl_path}")

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
            "image_path": record["image_path"],
            "image_id": record["image_id"],
            "positives": list(record["positives"]),
            "hard_negatives": list(record["hard_negatives"]),
            "challenge_tags": list(record["challenge_tags"]),
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
    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch], dim=0)

        positive_texts = [
            choose_texts(item["positives"], 1, text_sampling)[0]
            for item in batch
        ]
        positive_tokens = tokenizer(positive_texts)

        hard_negative_texts = []
        hard_negative_mask = []
        for item in batch:
            negatives = item["hard_negatives"]
            sampled = choose_texts(
                negatives,
                num_hard_negatives,
                text_sampling,
            ) if negatives else []

            row_texts = []
            row_mask = []
            for text in sampled:
                row_texts.append(text)
                row_mask.append(1.0)

            while len(row_texts) < num_hard_negatives:
                row_texts.append("")
                row_mask.append(0.0)

            hard_negative_texts.extend(row_texts)
            hard_negative_mask.append(row_mask)

        if num_hard_negatives > 0:
            flat_negative_tokens = tokenizer(hard_negative_texts)
            negative_tokens = flat_negative_tokens.view(
                len(batch), num_hard_negatives, -1
            )
            negative_mask = torch.tensor(hard_negative_mask, dtype=torch.float32)
        else:
            negative_tokens = torch.empty((len(batch), 0, 77), dtype=torch.long)
            negative_mask = torch.empty((len(batch), 0), dtype=torch.float32)

        return {
            "images": images,
            "positive_tokens": positive_tokens,
            "negative_tokens": negative_tokens,
            "negative_mask": negative_mask,
            "positive_texts": positive_texts,
            "image_paths": [item["image_path"] for item in batch],
            "image_ids": [item["image_id"] for item in batch],
        }

    return collate_fn


def build_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    total_steps = max(total_steps, 1)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (current_step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def compute_hard_negative_loss(
    image_features: torch.Tensor,
    negative_features: torch.Tensor,
    negative_mask: torch.Tensor,
    positive_scores: torch.Tensor,
    margin: float,
    strategy: str,
) -> torch.Tensor:
    if negative_features.numel() == 0 or negative_mask.numel() == 0:
        return image_features.new_zeros(())

    negative_scores = torch.einsum("bd,bnd->bn", image_features, negative_features)
    active_mask = negative_mask > 0
    if not active_mask.any():
        return image_features.new_zeros(())

    if strategy == "hinge":
        margins = margin + negative_scores - positive_scores.unsqueeze(1)
        losses = F.relu(margins) * negative_mask
        return losses.sum() / negative_mask.sum().clamp_min(1.0)

    if strategy == "logsigmoid":
        diffs = positive_scores.unsqueeze(1) - negative_scores
        losses = -F.logsigmoid(diffs / max(margin, 1e-6)) * negative_mask
        return losses.sum() / negative_mask.sum().clamp_min(1.0)

    raise ValueError(f"Unsupported hard negative loss strategy: {strategy}")


def append_metrics_row(csv_path: Path, row: Dict[str, float | int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_metrics_jsonl(jsonl_path: Path, row: Dict[str, float | int]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_training_curves(metrics_history: List[Dict[str, float | int]], output_dir: Path) -> None:
    if not metrics_history:
        return

    steps = [row["global_step"] for row in metrics_history]
    total_loss = [row["train_total_loss"] for row in metrics_history]
    clip_loss = [row["train_clip_loss"] for row in metrics_history]
    hard_neg_loss = [row["train_hard_negative_loss"] for row in metrics_history]
    lr = [row["lr"] for row in metrics_history]

    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(steps, total_loss, label="total_loss")
    ax1.plot(steps, clip_loss, label="clip_loss")
    ax1.plot(steps, hard_neg_loss, label="hard_negative_loss")
    ax1.set_xlabel("global_step")
    ax1.set_ylabel("loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(steps, lr, label="lr", color="tab:orange")
    ax2.set_xlabel("global_step")
    ax2.set_ylabel("learning_rate")
    ax2.set_title("Learning Rate")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close()


def save_checkpoint(
    save_path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    metrics_history: List[Dict[str, float | int]],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "args": vars(args),
            "metrics_history": metrics_history,
        },
        save_path,
    )


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    clip_loss_fn,
    device: torch.device,
    epoch: int,
    global_step: int,
    log_every_n_steps: int,
    grad_clip_norm: Optional[float],
    accum_freq: int,
    hard_negative_weight: float,
    hard_negative_margin: float,
    hard_negative_loss_type: str,
) -> Tuple[Dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda"
    start_time = time.time()
    num_batches = len(dataloader)

    running_total_loss = 0.0
    running_clip_loss = 0.0
    running_hard_negative_loss = 0.0

    for batch_idx, batch in enumerate(dataloader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        positive_tokens = batch["positive_tokens"].to(device, non_blocking=True)
        negative_tokens = batch["negative_tokens"].to(device, non_blocking=True)
        negative_mask = batch["negative_mask"].to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            image_features, positive_features, logit_scale = model(images, positive_tokens)
            clip_loss = clip_loss_fn(image_features, positive_features, logit_scale)

            image_features = F.normalize(image_features, dim=-1)
            positive_features = F.normalize(positive_features, dim=-1)
            positive_scores = (image_features * positive_features).sum(dim=-1)

            if negative_tokens.numel() > 0:
                flat_negative_tokens = negative_tokens.view(-1, negative_tokens.shape[-1])
                flat_negative_features = model.encode_text(flat_negative_tokens)
                flat_negative_features = F.normalize(flat_negative_features, dim=-1)
                negative_features = flat_negative_features.view(
                    negative_tokens.shape[0],
                    negative_tokens.shape[1],
                    -1,
                )
                hard_negative_loss = compute_hard_negative_loss(
                    image_features=image_features,
                    negative_features=negative_features,
                    negative_mask=negative_mask,
                    positive_scores=positive_scores,
                    margin=hard_negative_margin,
                    strategy=hard_negative_loss_type,
                )
            else:
                hard_negative_loss = clip_loss.new_zeros(())

            total_loss = clip_loss + hard_negative_weight * hard_negative_loss
            loss_for_backward = total_loss / accum_freq

        scaler.scale(loss_for_backward).backward()

        if batch_idx % accum_freq == 0 or batch_idx == num_batches:
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        running_total_loss += total_loss.item()
        running_clip_loss += clip_loss.item()
        running_hard_negative_loss += hard_negative_loss.item()

        if batch_idx % log_every_n_steps == 0 or batch_idx == num_batches:
            elapsed = time.time() - start_time
            throughput = (batch_idx * images.shape[0]) / max(elapsed, 1e-6)
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch} Step {batch_idx}/{num_batches} "
                f"Total {total_loss.item():.4f} "
                f"CLIP {clip_loss.item():.4f} "
                f"HardNeg {hard_negative_loss.item():.4f} "
                f"LR {current_lr:.6e} Throughput {throughput:.2f} samples/s"
            )

    return (
        {
            "train_total_loss": running_total_loss / max(num_batches, 1),
            "train_clip_loss": running_clip_loss / max(num_batches, 1),
            "train_hard_negative_loss": running_hard_negative_loss / max(num_batches, 1),
            "lr": scheduler.get_last_lr()[0],
        },
        global_step,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileCLIP2 on LPCV contrastive jsonl data"
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="./build_datasets/data/dataset_raw_contrastive.jsonl",
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_jsonl")
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--accum-freq", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--save-every-epoch", action="store_true")

    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-positives-per-image", type=int, default=None)
    parser.add_argument("--max-hard-negatives-per-image", type=int, default=None)
    parser.add_argument("--shuffle-texts-on-load", action="store_true")
    parser.add_argument(
        "--text-sampling",
        type=str,
        default="random",
        choices=["random", "first"],
    )

    parser.add_argument("--num-hard-negatives", type=int, default=4)
    parser.add_argument("--hard-negative-weight", type=float, default=0.5)
    parser.add_argument("--hard-negative-margin", type=float, default=0.2)
    parser.add_argument(
        "--hard-negative-loss-type",
        type=str,
        default="hinge",
        choices=["hinge", "logsigmoid"],
    )

    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, _, tokenizer = _load_clip(args.model_name, device, args.pretrained)
    if args.grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing()

    train_dataset = ContrastiveRecordDataset(
        jsonl_path=args.jsonl_path,
        repo_root=args.repo_root,
        max_records=args.max_records,
        max_positives_per_image=args.max_positives_per_image,
        max_hard_negatives_per_image=args.max_hard_negatives_per_image,
        shuffle_texts_on_load=args.shuffle_texts_on_load,
    )
    print(f"Loaded {len(train_dataset)} image records from {args.jsonl_path}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=create_collate_fn(
            tokenizer=tokenizer,
            num_hard_negatives=args.num_hard_negatives,
            text_sampling=args.text_sampling,
        ),
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, args.beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=max(math.ceil(len(train_dataloader) / max(args.accum_freq, 1)) * args.epochs, 1),
        warmup_steps=args.warmup_steps,
    )
    scaler = GradScaler(enabled=device.type == "cuda")
    clip_loss_fn = open_clip.ClipLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = output_dir / "metrics.csv"
    metrics_jsonl_path = output_dir / "metrics.jsonl"

    metrics_history: List[Dict[str, float | int]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_metrics, global_step = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            clip_loss_fn=clip_loss_fn,
            device=device,
            epoch=epoch,
            global_step=global_step,
            log_every_n_steps=args.log_every_n_steps,
            grad_clip_norm=args.grad_clip_norm,
            accum_freq=args.accum_freq,
            hard_negative_weight=args.hard_negative_weight,
            hard_negative_margin=args.hard_negative_margin,
            hard_negative_loss_type=args.hard_negative_loss_type,
        )

        epoch_row: Dict[str, float | int] = {
            "epoch": epoch,
            "global_step": global_step,
            "num_records": len(train_dataset),
            "batch_size": args.batch_size,
            "num_hard_negatives": args.num_hard_negatives,
            **epoch_metrics,
        }
        metrics_history.append(epoch_row)
        append_metrics_row(metrics_csv_path, epoch_row)
        append_metrics_jsonl(metrics_jsonl_path, epoch_row)
        plot_training_curves(metrics_history, output_dir)

        print(
            f"Epoch {epoch} summary: "
            f"total={epoch_metrics['train_total_loss']:.4f} "
            f"clip={epoch_metrics['train_clip_loss']:.4f} "
            f"hardneg={epoch_metrics['train_hard_negative_loss']:.4f}"
        )

        latest_path = output_dir / "checkpoint_latest.pt"
        save_checkpoint(
            save_path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            args=args,
            metrics_history=metrics_history,
        )

        if args.save_every_epoch:
            epoch_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(
                save_path=epoch_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                args=args,
                metrics_history=metrics_history,
            )

    final_weights_path = output_dir / f"{args.model_name}_finetuned.pt"
    torch.save(model.state_dict(), final_weights_path)
    print(f"Training complete. Final weights saved to {final_weights_path}")
    print(f"Metrics saved to {metrics_csv_path} and {metrics_jsonl_path}")
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")


if __name__ == "__main__":
    run_training(parse_args())

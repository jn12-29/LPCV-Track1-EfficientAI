from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import open_clip

from utils.clip_utils import _load_clip
from utils.preprocess import preprocess_image


MetricValue = float | int | str
MetricsRow = Dict[str, MetricValue]
TRAIN_LOG_PATH: Optional[Path] = None


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_tag(value: str) -> str:
    sanitized = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value
    )
    return sanitized.strip("-_") or "unknown"


def format_config_value(value: Any) -> str:
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e3):
            return format(value, ".0e")
        return format(value, "g")
    return str(value)


def build_config_stamp(args: argparse.Namespace) -> str:
    config_stamp = "_".join(
        [
            f"bs{args.batch_size}",
            f"ep{args.epochs}",
            f"lr{sanitize_tag(format_config_value(args.lr))}",
            f"wd{sanitize_tag(format_config_value(args.weight_decay))}",
            f"acc{args.accum_freq}",
            f"hn{args.num_hard_negatives}",
            f"hnw{sanitize_tag(format_config_value(args.hard_negative_weight))}",
            f"seed{args.seed}",
        ]
    )
    return sanitize_tag(config_stamp)


def build_run_name(args: argparse.Namespace, run_timestamp: str) -> str:
    return "__".join(
        [
            sanitize_tag(args.model_name),
            build_config_stamp(args),
            run_timestamp,
        ]
    )


def log_message(message: str, run_name: Optional[str] = None) -> None:
    prefix = f"[{current_timestamp()}]"
    if run_name:
        prefix = f"{prefix}[{run_name}]"
    line = f"{prefix} {message}"
    print(line)
    if TRAIN_LOG_PATH is not None:
        TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRAIN_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def save_run_config(config_path: Path, payload: Dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_tensorboard_scalars(
    writer: Optional[SummaryWriter],
    tag_prefix: str,
    metrics: Dict[str, MetricValue],
    step: int,
) -> None:
    if writer is None:
        return

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{tag_prefix}/{key}", value, step)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def parse_gpu_ids(gpu_ids: Optional[str]) -> Optional[List[int]]:
    if gpu_ids is None:
        return None
    parsed = [part.strip() for part in gpu_ids.split(",") if part.strip()]
    if not parsed:
        raise ValueError("--gpu-ids was provided but no valid GPU ids were found")
    return [int(part) for part in parsed]


def init_distributed_context(args: argparse.Namespace) -> Dict[str, Any]:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    env_local_rank = os.environ.get("LOCAL_RANK")
    local_rank = args.local_rank
    if env_local_rank is not None:
        local_rank = int(env_local_rank)
    elif local_rank < 0:
        local_rank = 0

    distributed = world_size > 1
    use_cuda = torch.cuda.is_available() and args.device != "cpu"

    if distributed:
        if not use_cuda:
            raise ValueError("Distributed training currently requires CUDA")
        if gpu_ids is None:
            if torch.cuda.device_count() < world_size:
                raise ValueError(
                    f"WORLD_SIZE={world_size} exceeds available CUDA devices ({torch.cuda.device_count()})"
                )
            gpu_ids = list(range(world_size))
        if len(gpu_ids) != world_size:
            raise ValueError(
                f"WORLD_SIZE={world_size} but --gpu-ids resolved to {len(gpu_ids)} GPUs: {gpu_ids}"
            )
        if local_rank < 0 or local_rank >= len(gpu_ids):
            raise ValueError(
                f"LOCAL_RANK={local_rank} is out of range for --gpu-ids {gpu_ids}"
            )

        device_index = gpu_ids[local_rank]
        torch.cuda.set_device(device_index)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", device_index)
    else:
        if gpu_ids is not None:
            if not use_cuda:
                raise ValueError("--gpu-ids requires CUDA")
            if len(gpu_ids) != 1:
                raise ValueError(
                    "Multiple --gpu-ids require torchrun, e.g. `torchrun --nproc_per_node=2 train/finetune.py --gpu-ids 0,1 ...`"
                )
            device = torch.device("cuda", gpu_ids[0])
            torch.cuda.set_device(device)
        else:
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            if device.type == "cuda" and device.index is not None:
                torch.cuda.set_device(device)

    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main_process": rank == 0,
        "device": device,
        "gpu_ids": gpu_ids,
    }


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_run_timestamp(run_timestamp: Optional[str], distributed: bool) -> str:
    if not distributed:
        if run_timestamp is None:
            raise ValueError("run_timestamp must be provided when not distributed")
        return run_timestamp

    payload = [run_timestamp]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("Failed to broadcast run timestamp from rank 0")
    return payload[0]


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
        rng = random.Random(self.shuffle_seed) if self.shuffle_seed is not None else None
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
    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch], dim=0)

        positive_texts = [
            choose_texts(item["positives"], 1, text_sampling)[0] for item in batch
        ]
        positive_tokens = tokenizer(positive_texts)

        hard_negative_texts = []
        hard_negative_mask = []
        for item in batch:
            negatives = item["hard_negatives"]
            sampled = (
                choose_texts(
                    negatives,
                    num_hard_negatives,
                    text_sampling,
                )
                if negatives
                else []
            )

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


def append_metrics_row(csv_path: Path, row: MetricsRow) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_metrics_jsonl(jsonl_path: Path, row: MetricsRow) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_training_curves(
    metrics_history: List[MetricsRow], output_dir: Path
) -> None:
    if not metrics_history:
        return

    steps = [row["global_step"] for row in metrics_history]
    total_loss = [row["train_total_loss"] for row in metrics_history]
    clip_loss = [row["train_loss"] for row in metrics_history]
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
    metrics_history: List[MetricsRow],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": unwrap_model(model).state_dict(),
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
    is_main_process: bool,
    distributed: bool,
    run_name: Optional[str] = None,
    writer: Optional[SummaryWriter] = None,
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

        with autocast(device_type=device.type, enabled=amp_enabled):
            image_features, positive_features, logit_scale = model(
                images, positive_tokens
            )
            clip_loss = clip_loss_fn(image_features, positive_features, logit_scale)

            image_features = F.normalize(image_features, dim=-1)
            positive_features = F.normalize(positive_features, dim=-1)
            positive_scores = (image_features * positive_features).sum(dim=-1)

            if negative_tokens.numel() > 0:
                flat_negative_tokens = negative_tokens.view(
                    -1, negative_tokens.shape[-1]
                )
                _, flat_negative_features, _ = model(text=flat_negative_tokens)
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
            previous_scale = scaler.get_scale() if amp_enabled else 1.0
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            current_scale = scaler.get_scale() if amp_enabled else 1.0
            if not amp_enabled or current_scale >= previous_scale:
                scheduler.step()
            global_step += 1

        running_total_loss += total_loss.item()
        running_clip_loss += clip_loss.item()
        running_hard_negative_loss += hard_negative_loss.item()

        if is_main_process and (batch_idx % log_every_n_steps == 0 or batch_idx == num_batches):
            elapsed = time.time() - start_time
            throughput = (batch_idx * images.shape[0]) / max(elapsed, 1e-6)
            current_lr = scheduler.get_last_lr()[0]
            tb_step = (epoch - 1) * num_batches + batch_idx
            write_tensorboard_scalars(
                writer,
                "train_step",
                {
                    "total_loss": total_loss.item(),
                    "clip_loss": clip_loss.item(),
                    "hard_negative_loss": hard_negative_loss.item(),
                    "lr": current_lr,
                    "throughput": throughput,
                },
                tb_step,
            )
            log_message(
                f"Epoch {epoch} Step {batch_idx}/{num_batches} "
                f"Total {total_loss.item():.4f} "
                f"CLIP {clip_loss.item():.4f} "
                f"HardNeg {hard_negative_loss.item():.4f} "
                f"LR {current_lr:.6e} Throughput {throughput:.2f} samples/s",
                run_name=run_name,
            )

    reduced_stats = torch.tensor(
        [
            running_total_loss,
            running_clip_loss,
            running_hard_negative_loss,
            float(num_batches),
        ],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        dist.all_reduce(reduced_stats, op=dist.ReduceOp.SUM)

    total_batches = max(reduced_stats[3].item(), 1.0)

    return (
        {
            "train_total_loss": reduced_stats[0].item() / total_batches,
            "train_loss": reduced_stats[1].item() / total_batches,
            "train_hard_negative_loss": reduced_stats[2].item() / total_batches,
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
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids. Single-GPU example: 2. Multi-GPU example with torchrun: 0,1,2,3",
    )
    parser.add_argument("--local-rank", type=int, default=-1)
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
    global TRAIN_LOG_PATH

    dist_ctx = init_distributed_context(args)
    distributed = dist_ctx["distributed"]
    world_size = dist_ctx["world_size"]
    rank = dist_ctx["rank"]
    is_main_process = dist_ctx["is_main_process"]
    device = dist_ctx["device"]
    gpu_ids = dist_ctx["gpu_ids"]

    set_seed(args.seed + rank)

    run_timestamp = make_run_timestamp() if is_main_process else None
    run_timestamp = broadcast_run_timestamp(run_timestamp, distributed)
    run_name = build_run_name(args, run_timestamp)
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = output_dir / "train.log"
    metrics_csv_path = output_dir / "metrics.csv"
    metrics_jsonl_path = output_dir / "metrics.jsonl"
    config_json_path = output_dir / "run_config.json"
    tensorboard_dir = output_dir / "tensorboard"

    previous_log_path = TRAIN_LOG_PATH
    TRAIN_LOG_PATH = train_log_path if is_main_process else None
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if is_main_process else None

    run_config: Dict[str, Any] = {
        "run_name": run_name,
        "run_timestamp": run_timestamp,
        "started_at": current_timestamp(),
        "output_dir": str(output_dir.resolve()),
        "train_log_path": str(train_log_path.resolve()),
        "tensorboard_dir": str(tensorboard_dir.resolve()),
        "args": vars(args),
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": dist_ctx["local_rank"],
        "gpu_ids": gpu_ids,
    }
    try:
        if is_main_process:
            save_run_config(config_json_path, run_config)

        if is_main_process:
            log_message(f"Using device: {device}", run_name=run_name)
            log_message(
                f"Distributed training: {distributed} (world_size={world_size}, gpu_ids={gpu_ids})",
                run_name=run_name,
            )
            log_message(f"Run outputs will be saved to {output_dir}", run_name=run_name)
            log_message(f"Train log saved to {train_log_path}", run_name=run_name)
            log_message(
                f"TensorBoard logs saved to {tensorboard_dir}",
                run_name=run_name,
            )
            log_message(f"Run config saved to {config_json_path}", run_name=run_name)

        model, _, tokenizer = _load_clip(
            args.model_name,
            device,
            pretrained=args.pretrained,
        )
        if args.grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing()
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
                static_graph=True,
            )

        train_dataset = ContrastiveRecordDataset(
            jsonl_path=args.jsonl_path,
            max_records=args.max_records,
            max_positives_per_image=args.max_positives_per_image,
            max_hard_negatives_per_image=args.max_hard_negatives_per_image,
            shuffle_texts_on_load=args.shuffle_texts_on_load,
            shuffle_seed=args.seed,
            log_progress=is_main_process,
        )
        if is_main_process:
            log_message(
                f"Loaded {len(train_dataset)} image records from {args.jsonl_path}",
                run_name=run_name,
            )
        run_config["num_records"] = len(train_dataset)
        run_config["effective_global_batch_size"] = args.batch_size * max(args.accum_freq, 1) * world_size
        if is_main_process:
            save_run_config(config_json_path, run_config)
            writer.add_text(
                "run/config_json",
                json.dumps(run_config, ensure_ascii=False, indent=2),
                0,
            )

        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
                seed=args.seed,
            )
            if distributed
            else None
        )
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(args.seed + rank)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=dataloader_generator,
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
            total_steps=max(
                math.ceil(len(train_dataloader) / max(args.accum_freq, 1))
                * args.epochs,
                1,
            ),
            warmup_steps=args.warmup_steps,
        )
        scaler = GradScaler(device=device.type, enabled=device.type == "cuda")
        clip_loss_fn = open_clip.ClipLoss(
            cache_labels=True,
            rank=rank,
            world_size=world_size,
        )

        metrics_history: List[MetricsRow] = []
        global_step = 0
        epoch_digits = max(len(str(args.epochs)), 2)
        previous_latest_path: Optional[Path] = None

        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

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
                is_main_process=is_main_process,
                distributed=distributed,
                run_name=run_name,
                writer=writer,
            )

            epoch_row: MetricsRow = {
                "logged_at": current_timestamp(),
                "run_timestamp": run_timestamp,
                "run_name": run_name,
                "model_name": args.model_name,
                "jsonl_path": str(Path(args.jsonl_path).resolve()),
                "epoch": epoch,
                "global_step": global_step,
                "num_records": len(train_dataset),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr_init": args.lr,
                "weight_decay": args.weight_decay,
                "accum_freq": args.accum_freq,
                "seed": args.seed,
                "num_hard_negatives": args.num_hard_negatives,
                "hard_negative_weight": args.hard_negative_weight,
                "hard_negative_margin": args.hard_negative_margin,
                "text_sampling": args.text_sampling,
                "world_size": world_size,
                **epoch_metrics,
            }
            if is_main_process:
                metrics_history.append(epoch_row)
                append_metrics_row(metrics_csv_path, epoch_row)
                append_metrics_jsonl(metrics_jsonl_path, epoch_row)
                plot_training_curves(metrics_history, output_dir)
                write_tensorboard_scalars(writer, "train_epoch", epoch_metrics, epoch)
                write_tensorboard_scalars(
                    writer,
                    "train_epoch_meta",
                    {"global_step": global_step},
                    epoch,
                )
                writer.flush()

                log_message(
                    f"Epoch {epoch} summary: "
                    f"total={epoch_metrics['train_total_loss']:.4f} "
                    f"clip={epoch_metrics['train_loss']:.4f} "
                    f"hardneg={epoch_metrics['train_hard_negative_loss']:.4f}",
                    run_name=run_name,
                )

                latest_path = output_dir / (
                    f"checkpoint_latest_epoch_{epoch:0{epoch_digits}d}.pt"
                )
                if previous_latest_path is not None and previous_latest_path.exists():
                    previous_latest_path.unlink()
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
                previous_latest_path = latest_path

                if args.save_every_epoch:
                    epoch_path = output_dir / f"checkpoint_epoch_{epoch:0{epoch_digits}d}.pt"
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
        if is_main_process:
            torch.save(unwrap_model(model).state_dict(), final_weights_path)
            run_config["finished_at"] = current_timestamp()
            run_config["final_weights_path"] = str(final_weights_path.resolve())
            save_run_config(config_json_path, run_config)
            writer.add_text(
                "run/final_weights_path",
                str(final_weights_path.resolve()),
                global_step,
            )
            writer.flush()
            log_message(
                f"Training complete. Final weights saved to {final_weights_path}",
                run_name=run_name,
            )
            log_message(
                f"Metrics saved to {metrics_csv_path} and {metrics_jsonl_path}",
                run_name=run_name,
            )
            log_message(
                f"Training curves saved to {output_dir / 'training_curves.png'}",
                run_name=run_name,
            )
    finally:
        if writer is not None:
            writer.close()
        TRAIN_LOG_PATH = previous_log_path
        cleanup_distributed()


if __name__ == "__main__":
    run_training(parse_args())

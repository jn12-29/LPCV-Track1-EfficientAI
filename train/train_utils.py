from __future__ import annotations

import contextlib
import json
import math
import re
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

MetricValue = float | int | str
MetricsRow = Dict[str, MetricValue]


class StepTimer(contextlib.AbstractContextManager):
    """Measures wall-clock time of a code block.

    With cuda_sync=True, calls torch.cuda.synchronize() on entry and exit
    so GPU kernels are fully drained before/after timing.
    """

    def __init__(self, name: str, cuda_sync: bool = False) -> None:
        self.name = name
        self.cuda_sync = cuda_sync
        self.elapsed: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "StepTimer":
        if self.cuda_sync:
            torch.cuda.synchronize()
        self._t0 = _time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        if self.cuda_sync:
            torch.cuda.synchronize()
        self.elapsed = _time.perf_counter() - self._t0


class StepTimeAccum:
    """Accumulates per-step timing for multiple named stages.

    Call update(name, elapsed) each step, then means() to read averages
    without resetting, or report() to read and reset.
    summary_str() formats means as a compact string for appending to log lines.
    """

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, name: str, elapsed: float) -> None:
        self._sums[name] = self._sums.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1

    def means(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def report(self) -> Dict[str, float]:
        result = self.means()
        self.reset()
        return result

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()

    def summary_str(self) -> str:
        m = self.means()
        if not m:
            return ""
        parts = [f"{k}={v * 1000:.1f}ms" for k, v in m.items()]
        return "  [" + " ".join(parts) + "]"

TRAIN_LOG_PATH: Optional[Path] = None


def set_log_path(path: Optional[Path]) -> None:
    global TRAIN_LOG_PATH
    TRAIN_LOG_PATH = path


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


def build_config_stamp(args) -> str:
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


def build_run_name(args, run_timestamp: str) -> str:
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


def save_checkpoint(
    save_path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    args,
    metrics_history: List[MetricsRow],
    sim=None,
    relu_labels: List[str] | None = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "metrics_history": metrics_history,
    }
    if relu_labels:
        payload["relu_blocks"] = relu_labels
    if sim is not None:
        from utils.qat_utils import get_qat_encodings_json
        payload["qat_enabled"] = True
        payload["qat_weight_bw"] = getattr(args, "qat_weight_bw", 8)
        payload["qat_act_bw"] = getattr(args, "qat_act_bw", 8)
        payload["qat_encodings"] = get_qat_encodings_json(sim)
    torch.save(payload, save_path)

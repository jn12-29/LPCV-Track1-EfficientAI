from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MetricValue = float | int | str
MetricsRow = Dict[str, MetricValue]


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


def plot_training_curves(metrics_history: List[MetricsRow], output_dir: Path) -> None:
    if not metrics_history:
        return

    steps = [row["global_step"] for row in metrics_history]
    total_loss = [row["train_total_loss"] for row in metrics_history]
    clip_loss = [row["train_loss"] for row in metrics_history]
    hard_neg_loss = [row["train_hard_negative_loss"] for row in metrics_history]
    lr = [row["lr"] for row in metrics_history]

    C_CLIP = "#2563EB"
    C_TOTAL = "#7C3AED"
    C_HN = "#DC2626"
    C_LR = "#D97706"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor("#F8FAFC")
    for ax in (ax1, ax2):
        ax.set_facecolor("#F1F5F9")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")
        ax.tick_params(colors="#475569", labelsize=10)
        ax.grid(True, color="white", linewidth=1.2, alpha=0.9)

    # --- Loss panel: clip/total on left axis, hard_neg on right axis ---
    lns1 = ax1.plot(steps, clip_loss, color=C_CLIP, linewidth=1.8, label="CLIP loss")
    lns2 = ax1.plot(
        steps,
        total_loss,
        color=C_TOTAL,
        linewidth=1.8,
        linestyle="--",
        label="Total loss",
    )
    ax1.set_ylabel("CLIP / Total loss", color="#334155", fontsize=11, labelpad=8)
    ax1.tick_params(axis="y", colors="#334155")

    ax1r = ax1.twinx()
    ax1r.set_facecolor("#F1F5F9")
    lns3 = ax1r.plot(
        steps,
        hard_neg_loss,
        color=C_HN,
        linewidth=1.8,
        alpha=0.85,
        label="Hard-neg loss",
    )
    ax1r.set_ylabel("Hard-neg loss", color=C_HN, fontsize=11, labelpad=8)
    ax1r.tick_params(axis="y", colors=C_HN, labelsize=10)
    ax1r.spines[["top", "left"]].set_visible(False)
    ax1r.spines["right"].set_color(C_HN)

    all_lines = lns1 + lns2 + lns3
    ax1.legend(
        all_lines,
        [l.get_label() for l in all_lines],
        loc="upper right",
        framealpha=0.85,
        fontsize=10,
        edgecolor="#CBD5E1",
    )
    ax1.set_title(
        "Training Loss", fontsize=13, fontweight="bold", color="#1E293B", pad=10
    )
    ax1.set_xlabel("Epoch", color="#475569", fontsize=11)

    # --- LR panel ---
    ax2.plot(steps, lr, color=C_LR, linewidth=1.8)
    ax2.set_ylabel("Learning rate", color="#334155", fontsize=11, labelpad=8)
    ax2.set_xlabel("Epoch", color="#475569", fontsize=11)
    ax2.set_title(
        "Learning Rate Schedule",
        fontsize=13,
        fontweight="bold",
        color="#1E293B",
        pad=10,
    )
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2e}"))

    plt.tight_layout(pad=2.5)
    plt.savefig(
        output_dir / "training_curves.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close()

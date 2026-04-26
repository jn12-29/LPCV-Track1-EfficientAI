from __future__ import annotations

import math
from pathlib import Path


def plot_reconstruction_metrics(
    metrics_history: list[dict],
    output_dir: Path,
) -> None:
    if not metrics_history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return

    labels = [m["block"] for m in metrics_history]
    cos_sims = [
        0.0 if math.isnan(m["cos_sim"]) else m["cos_sim"] for m in metrics_history
    ]
    losses = [
        0.0 if math.isnan(m["final_loss"]) else m["final_loss"] for m in metrics_history
    ]
    raw_gelu_cos = [
        m.get("gelu_cos_sim", float("nan")) for m in metrics_history
    ]
    gelu_cos_sims = [0.0 if math.isnan(v) else v for v in raw_gelu_cos]
    has_gelu_drift = any(not math.isnan(v) for v in raw_gelu_cos)

    n = len(labels)
    _kind_color = {"text_sequential": "#2563EB", "conv_stem": "#16A34A"}
    colors = [
        (
            "#9CA3AF"
            if m["status"] in ("REVERTED", "SKIPPED")
            else _kind_color.get(m["kind"], "#7C3AED")
        )
        for m in metrics_history
    ]
    drift_colors = [
        "#EA580C" if not math.isnan(raw_gelu_cos[i]) else "#E2E8F0"
        for i in range(n)
    ]

    n_rows = 3 if has_gelu_drift else 2
    fig_height = 11 if has_gelu_drift else 7
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(10, n * 0.45), fig_height))
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if has_gelu_drift else None
    fig.patch.set_facecolor("#F8FAFC")
    for ax in axes:
        ax.set_facecolor("#F1F5F9")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color="white", linewidth=1.2, alpha=0.9)

    ax1.bar(range(n), cos_sims, color=colors, alpha=0.85)
    ax1.axhline(
        0.99, color="#DC2626", linewidth=1.2, linestyle="--", label="0.99 threshold"
    )
    ax1.set_ylabel("Cosine similarity", fontsize=11)
    ax1.set_title(
        "MLP Reconstruction — Cosine Similarity per Block",
        fontsize=13,
        fontweight="bold",
    )
    y_min = max(0.0, min(cos_sims) - 0.02)
    ax1.set_ylim(y_min, 1.005)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    legend_elements = [
        Patch(color="#2563EB", label="text"),
        Patch(color="#16A34A", label="conv_stem"),
        Patch(color="#7C3AED", label="visual"),
        Patch(color="#9CA3AF", label="GELU (reverted/skipped)"),
        plt.Line2D([0], [0], color="#DC2626", linestyle="--", label="0.99 threshold"),
    ]
    ax1.legend(handles=legend_elements, fontsize=9)

    ax2.bar(range(n), losses, color=colors, alpha=0.85)
    ax2.set_ylabel("Final loss", fontsize=11)
    ax2.set_title(
        "MLP Reconstruction — Final Loss per Block", fontsize=13, fontweight="bold"
    )
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.legend(
        handles=[
            Patch(color="#2563EB", label="text"),
            Patch(color="#16A34A", label="conv_stem"),
            Patch(color="#7C3AED", label="visual"),
            Patch(color="#9CA3AF", label="GELU (reverted/skipped)"),
        ],
        fontsize=9,
    )

    if ax3 is not None:
        ax3.bar(range(n), gelu_cos_sims, color=drift_colors, alpha=0.85)
        ax3.set_ylabel("Cosine similarity", fontsize=11)
        ax3.set_title(
            "GELU Drift — Cosine Similarity vs Original (REVERTED blocks, non-greedy)",
            fontsize=13,
            fontweight="bold",
        )
        valid_vals = [v for v in gelu_cos_sims if v > 0.0]
        y3_min = max(0.0, min(valid_vals) - 0.02) if valid_vals else 0.0
        ax3.set_ylim(y3_min, 1.005)
        ax3.axhline(
            0.99, color="#DC2626", linewidth=1.2, linestyle="--", label="0.99 ref"
        )
        ax3.legend(
            handles=[
                Patch(color="#EA580C", label="GELU drift (reverted)"),
                Patch(color="#E2E8F0", label="N/A"),
                plt.Line2D([0], [0], color="#DC2626", linestyle="--", label="0.99 ref"),
            ],
            fontsize=9,
        )
        ax3.set_xticks(range(n))
        ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout(pad=2.5)
    out_path = output_dir / "reconstruction_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved: {out_path}")

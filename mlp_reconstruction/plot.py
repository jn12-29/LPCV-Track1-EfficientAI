from __future__ import annotations

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

    import math

    labels = [m["block"] for m in metrics_history]
    cos_sims = [
        0.0 if math.isnan(m["cos_sim"]) else m["cos_sim"] for m in metrics_history
    ]
    losses = [
        0.0 if math.isnan(m["final_loss"]) else m["final_loss"] for m in metrics_history
    ]
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n * 0.45), 8))
    fig.patch.set_facecolor("#F8FAFC")
    for ax in (ax1, ax2):
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

    plt.tight_layout(pad=2.5)
    out_path = output_dir / "reconstruction_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved: {out_path}")

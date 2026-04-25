from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm


def _make_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = f"lr{args.lr}_nit{args.n_iters}_bs{args.batch_size}_nc{args.n_calib}_{args.aph_mode}"
    model = args.model_name.replace("/", "-")
    do_visual = not args.no_relu_image
    do_text = not args.no_relu_text
    enc = "all" if (do_visual and do_text) else ("img" if do_visual else "txt")
    return f"{model}__{config}_{enc}__{timestamp}"


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.gpu_id is not None:
        device = torch.device("cuda", args.gpu_id)
        torch.cuda.set_device(device)
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(
    model: nn.Module,
    model_name: str,
    relu_labels: list[str],
    path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "relu_blocks": relu_labels,
            "relu_image": any(lbl.startswith("visual[") for lbl in relu_labels),
            "relu_text": any(lbl.startswith("text[") for lbl in relu_labels),
        },
        path,
    )


def _plot_reconstruction_metrics(
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
    cos_sims = [m["cos_sim"] for m in metrics_history]
    losses = [m["final_loss"] for m in metrics_history]
    n = len(labels)
    _kind_color = {"text_sequential": "#2563EB", "conv_stem": "#16A34A"}
    colors = [_kind_color.get(m["kind"], "#7C3AED") for m in metrics_history]

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
        ],
        fontsize=9,
    )

    plt.tight_layout(pad=2.5)
    out_path = output_dir / "reconstruction_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    from utils.clip_utils import _load_clip
    from mlp_reconstruction.mlp_blocks import iter_mlp_blocks, replace_gelu_with_relu
    from mlp_reconstruction.calibrate import VGCalibrationLoader, collect_mlp_io
    from mlp_reconstruction.distill import distill_mlp, verify_reconstruction
    from train.train_utils import set_log_path, log_message, save_run_config
    from train.metrics import append_metrics_row, append_metrics_jsonl

    if args.output is None:
        run_name = _make_run_name(args)
        args.output = str(Path(args.output_dir) / run_name / "mlp_relu.pt")

    output_path = Path(args.output)
    set_log_path(output_path.parent / "train.log")
    save_run_config(output_path.parent / "run_config.json", vars(args))
    log_message(f"Output: {args.output}")

    device = _resolve_device(args)
    log_message(f"Device: {device}")

    checkpoint_for_model = (
        args.resume_from if args.resume_from else args.checkpoint_path
    )
    model, _, tokenizer = _load_clip(
        args.model_name, device, checkpoint_path=checkpoint_for_model
    )
    model.eval()

    relu_labels: list[str] = []
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        relu_labels = list(ckpt["relu_blocks"])
        log_message(
            f"Resumed from {args.resume_from}, {len(relu_labels)} blocks already done"
        )

    loader = VGCalibrationLoader(
        args.calib_jsonl,
        args.project_root,
        tokenizer,
        n_calib=args.n_calib,
        batch_size=args.calib_batch_size,
    )
    img_batches = loader.get_image_batches()
    txt_batches = loader.get_text_batches()
    log_message(
        f"Calibration: {len(loader.records)} records "
        f"({len(img_batches)} image batches, {len(txt_batches)} text batches)"
    )

    all_blocks = iter_mlp_blocks(model)
    do_visual = not args.no_relu_image
    do_text = not args.no_relu_text
    blocks = [
        b
        for b in all_blocks
        if (b.encoder == "visual" and do_visual) or (b.encoder == "text" and do_text)
    ]
    log_message(
        f"Encoders: image={do_visual}, text={do_text} → {len(blocks)} blocks to reconstruct"
    )

    done_set = set(relu_labels)
    skip_mode = args.skip_to is not None and args.skip_to not in done_set

    if args.skip_to is not None and args.skip_to not in done_set:
        all_labels = {info.label for info in blocks}
        if args.skip_to not in all_labels:
            raise ValueError(
                f"--skip-to '{args.skip_to}' not found. "
                f"Available: {[b.label for b in blocks]}"
            )

    metrics_csv = output_path.parent / "metrics.csv"
    metrics_jsonl = output_path.parent / "metrics.jsonl"
    metrics_history: list[dict] = []
    run_start = time.time()

    for info in tqdm(blocks, desc="Reconstructing MLP blocks"):
        if skip_mode:
            if info.label == args.skip_to:
                skip_mode = False
            else:
                if info.label not in done_set:
                    replace_gelu_with_relu(info)
                    relu_labels.append(info.label)
                    done_set.add(info.label)
                continue

        log_message(f"=== {info.label} ({info.kind}) ===")

        X_all, O_all = collect_mlp_io(model, info, img_batches, txt_batches, device)
        log_message(f"  X={tuple(X_all.shape)}, O={tuple(O_all.shape)}")

        block_start = time.time()
        final_loss, steps_run = distill_mlp(
            model,
            info,
            X_all,
            O_all,
            lr=args.lr,
            batch_size=args.batch_size,
            n_iters=args.n_iters,
            alpha=args.alpha,
            device=device,
            log_every=args.log_every,
            aph_mode=args.aph_mode,
            log_fn=log_message,
            early_stop_patience=args.early_stop_patience,
            early_stop_delta=args.early_stop_delta,
        )
        elapsed_s = time.time() - block_start
        relu_labels.append(info.label)
        done_set.add(info.label)

        cos_sim = verify_reconstruction(info, X_all, O_all, device)
        status = "OK" if cos_sim >= 0.99 else "WARN"
        log_message(
            f"  [{status}] final_loss={final_loss:.6f}  cos_sim={cos_sim:.4f}"
            f"  steps={steps_run}/{args.n_iters}  elapsed={elapsed_s:.1f}s"
        )

        row = {
            "block": info.label,
            "kind": info.kind,
            "final_loss": round(final_loss, 6),
            "cos_sim": round(cos_sim, 4),
            "status": status,
            "steps_run": steps_run,
            "elapsed_s": round(elapsed_s, 1),
        }
        metrics_history.append(row)
        append_metrics_row(metrics_csv, row)
        append_metrics_jsonl(metrics_jsonl, row)

        _save_checkpoint(model, args.model_name, relu_labels, args.output + ".tmp")

    _save_checkpoint(model, args.model_name, relu_labels, args.output)
    total_elapsed = time.time() - run_start
    warn_blocks = [r["block"] for r in metrics_history if r["status"] == "WARN"]
    summary = (
        f"Done. {len(relu_labels)}/{len(blocks)} blocks reconstructed"
        f"  total={total_elapsed:.0f}s"
    )
    if warn_blocks:
        summary += f"  WARN: {warn_blocks}"
    log_message(summary)
    log_message(f"Saved: {args.output}")

    _plot_reconstruction_metrics(metrics_history, output_path.parent)


def cmd_eval(args: argparse.Namespace) -> None:
    from pipeline.eval_local import run_clip_retrieval_eval

    device = _resolve_device(args)
    ckpt_meta = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model_name = ckpt_meta["model_name"]
    metrics = run_clip_retrieval_eval(
        root_dir=args.root_dir,
        image_to_text_csv=args.image_to_text_csv,
        textnums_to_texts_csv=args.textnums_to_texts_csv,
        model_name=model_name,
        batch_size=args.batch_size,
        k=args.k,
        device=str(device),
        checkpoint_path=args.checkpoint_path,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP Reconstruction for MobileCLIP2")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Run MLP reconstruction distillation")
    t.add_argument("--model-name", default="MobileCLIP2-S0")
    t.add_argument("--checkpoint-path", default=None)
    t.add_argument(
        "--calib-jsonl", default="build_datasets/data/vg_llm_contrastive.jsonl"
    )
    t.add_argument("--project-root", default=".")
    t.add_argument("--n-calib", type=int, default=1024)
    t.add_argument("--calib-batch-size", type=int, default=32)
    t.add_argument("--output-dir", default="checkpoints")
    t.add_argument(
        "--output",
        default=None,
        help="Full output path. If omitted, auto-generated as "
        "{output-dir}/{model}__{config}__{timestamp}/mlp_relu.pt",
    )
    t.add_argument(
        "--gpu-id", type=int, default=None, metavar="N", help="GPU index (e.g. 2)"
    )
    t.add_argument(
        "--no-relu-image",
        action="store_true",
        help="Skip visual-encoder reconstruction (default: reconstruct both)",
    )
    t.add_argument(
        "--no-relu-text",
        action="store_true",
        help="Skip text-encoder reconstruction (default: reconstruct both)",
    )
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--n-iters", type=int, default=20000)
    t.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="L_Clamp loss weight (0=disable; use >0 only when targeting QAT/quantization)",
    )
    t.add_argument("--aph-mode", default="uniform", choices=["uniform", "magnitude"])
    t.add_argument("--log-every", type=int, default=500)
    t.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        metavar="N",
        help="Stop after N log-intervals without improvement (0=disable)",
    )
    t.add_argument(
        "--early-stop-delta",
        type=float,
        default=5e-4,
        help="Min relative EMA-loss improvement to reset patience counter",
    )
    t.add_argument("--skip-to", default=None)
    t.add_argument("--resume-from", default=None)

    e = sub.add_parser("eval", help="Evaluate a reconstructed checkpoint")
    e.add_argument("--checkpoint-path", required=True)
    e.add_argument("--root-dir", default="sample_data")
    e.add_argument("--image-to-text-csv", default="sample_data/img_list.csv")
    e.add_argument("--textnums-to-texts-csv", default="sample_data/txt_list.csv")
    e.add_argument("--k", type=int, default=10)
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument(
        "--gpu-id", type=int, default=None, metavar="N", help="GPU index (e.g. 2)"
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()

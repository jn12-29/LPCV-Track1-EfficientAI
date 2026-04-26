from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from mlp_reconstruction.plot import plot_reconstruction_metrics


def _make_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = f"lr{args.lr}_nit{args.n_iters}_bs{args.batch_size}_nc{args.n_calib}_{args.aph_mode}"
    if getattr(args, "greedy", False):
        config += "_greedy"
    model = args.model_name.replace("/", "-")
    do_visual = not args.no_relu_image
    do_text = not args.no_relu_text
    enc = "all" if (do_visual and do_text) else ("img" if do_visual else "txt")
    if args.no_relu_stem and do_visual:
        enc += "_nostem"
    return f"{model}__{config}_{enc}__{timestamp}"


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.gpu_id is not None:
        device = torch.device("cuda", args.gpu_id)
        torch.cuda.set_device(device)
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_checkpoint(
    model: nn.Module,
    model_name: str,
    relu_labels: list[str],
    path: str,
    checkpoint_path: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "relu_blocks": relu_labels,
            "relu_image": any(lbl.startswith("visual[") for lbl in relu_labels),
            "relu_text": any(lbl.startswith("text[") for lbl in relu_labels),
            "checkpoint_path": checkpoint_path,
        },
        path,
    )


def cmd_train(args: argparse.Namespace) -> None:
    from utils.clip_utils import _load_clip
    from mlp_reconstruction.mlp_blocks import (
        iter_mlp_blocks,
        replace_gelu_with_relu,
        apply_relu_blocks,
        restore_gelu,
    )
    from mlp_reconstruction.calibrate import VGCalibrationLoader, collect_mlp_io
    from mlp_reconstruction.distill import distill_mlp, verify_reconstruction, verify_gelu_drift
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

    # Load resume checkpoint metadata first to know which blocks are already done.
    resume_ckpt: dict | None = None
    relu_labels: list[str] = []
    if args.resume_from:
        resume_ckpt = torch.load(
            args.resume_from, map_location="cpu", weights_only=False
        )
        relu_labels = list(resume_ckpt["relu_blocks"])
        log_message(
            f"Resumed from {args.resume_from}, {len(relu_labels)} blocks already done"
        )
        # If --checkpoint-path was not given, fall back to the path recorded in the
        # reconstruction checkpoint so that pre-collection uses the correct base model
        # (e.g. a GELU fine-tuned checkpoint rather than the pretrained weights).
        if args.checkpoint_path is None:
            args.checkpoint_path = resume_ckpt.get("checkpoint_path")
            if args.checkpoint_path is not None:
                log_message(
                    f"Using base model from resume checkpoint: {args.checkpoint_path}"
                )

    # Always load the original model (checkpoint_path) for pre-collection so that
    # every block's teacher target O comes from the all-GELU original model.
    model, _, tokenizer = _load_clip(
        args.model_name, device, checkpoint_path=args.checkpoint_path
    )
    model.eval()

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
    skip_stem = args.no_relu_stem
    blocks = [
        b
        for b in all_blocks
        if (b.encoder == "visual" and do_visual) or (b.encoder == "text" and do_text)
        if not (skip_stem and b.label == "visual[stem]")
    ]
    stem_note = " (stem excluded)" if skip_stem and do_visual else ""
    log_message(
        f"Encoders: image={do_visual}{stem_note}, text={do_text} → {len(blocks)} blocks to reconstruct"
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

    block_io: dict[str, tuple[torch.Tensor, torch.Tensor, float | None]] = {}
    if not args.greedy:
        # Pre-collect (X, O, gelu_ub) for all remaining blocks from the original all-GELU model.
        # This ensures every teacher target is the true original GELU output, not a
        # drifted output produced after upstream blocks have already been converted to ReLU.
        blocks_to_run = [b for b in blocks if b.label not in done_set]
        log_message(
            f"Pre-collecting MLP I/O from original model "
            f"for {len(blocks_to_run)} remaining blocks ..."
        )
        for info in tqdm(blocks_to_run, desc="Pre-collecting"):
            X_all, O_all, gelu_ub = collect_mlp_io(model, info, img_batches, txt_batches, device)
            block_io[info.label] = (X_all, O_all, gelu_ub)
    else:
        log_message(
            "Greedy mode: I/O will be collected just-in-time from the partially-replaced model"
        )

    # Restore the partially-distilled model state so distillation continues correctly.
    if resume_ckpt is not None:
        apply_relu_blocks(model, relu_labels)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        resume_ckpt = None  # free memory
        log_message(f"Restored model state from {args.resume_from}")

    metrics_csv = output_path.parent / "metrics.csv"
    metrics_jsonl = output_path.parent / "metrics.jsonl"
    metrics_history: list[dict] = []

    keep_gelu_set = set(args.keep_gelu_blocks)
    if keep_gelu_set:
        log_message(f"Manual GELU blocks (skip distillation): {sorted(keep_gelu_set)}")
    gelu_threshold = args.gelu_threshold
    if gelu_threshold is not None:
        log_message(f"Auto-revert threshold: cos_sim < {gelu_threshold} → keep GELU")

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

        if info.label in done_set:
            continue

        # Manual GELU keep: skip distillation entirely, block remains GELU.
        if info.label in keep_gelu_set:
            log_message(f"  [SKIPPED] {info.label} kept as GELU (--keep-gelu-blocks)")
            row = {
                "block": info.label,
                "kind": info.kind,
                "final_loss": float("nan"),
                "cos_sim": float("nan"),
                "status": "SKIPPED",
                "steps_run": 0,
                "elapsed_s": 0.0,
                "gelu_cos_sim": float("nan"),
                "gelu_loss": float("nan"),
            }
            metrics_history.append(row)
            append_metrics_row(metrics_csv, row)
            append_metrics_jsonl(metrics_jsonl, row)
            continue

        log_message(f"=== {info.label} ({info.kind}) ===")

        if args.greedy:
            X_all, O_all, gelu_ub = collect_mlp_io(model, info, img_batches, txt_batches, device)
        else:
            X_all, O_all, gelu_ub = block_io[info.label]
        log_message(f"  X={tuple(X_all.shape)}, O={tuple(O_all.shape)}")

        # Save original weights before distillation (needed for auto-revert).
        if gelu_threshold is not None:
            if info.kind == "conv_stem":
                _orig_block_state = {
                    k: v.clone() for k, v in info.mlp.state_dict().items()
                }
                _orig_fc1_state = _orig_fc2_state = None
            else:
                _orig_block_state = None
                _orig_fc1_state = {
                    k: v.clone() for k, v in info.fc1.state_dict().items()
                }
                _orig_fc2_state = {
                    k: v.clone() for k, v in info.fc2.state_dict().items()
                }
        else:
            _orig_block_state = _orig_fc1_state = _orig_fc2_state = None

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
            gelu_ub=gelu_ub,
            device=device,
            log_every=args.log_every,
            aph_mode=args.aph_mode,
            log_fn=log_message,
            early_stop_patience=args.early_stop_patience if args.early_stop else 0,
            early_stop_delta=args.early_stop_delta,
        )
        elapsed_s = time.time() - block_start
        cos_sim = verify_reconstruction(info, X_all, O_all, device)

        gelu_cos_sim = float("nan")
        gelu_loss = float("nan")

        if gelu_threshold is not None and cos_sim < gelu_threshold:
            # Revert: restore original weights and GELU activation.
            if info.kind == "conv_stem":
                info.mlp.load_state_dict(_orig_block_state)
            else:
                info.fc1.load_state_dict(_orig_fc1_state)
                info.fc2.load_state_dict(_orig_fc2_state)
            restore_gelu(info)
            status = "REVERTED"
            log_message(
                f"  [REVERTED] cos_sim={cos_sim:.4f} < threshold={gelu_threshold:.4f},"
                f" block kept as GELU"
            )
            if not args.greedy:
                gelu_cos_sim, gelu_loss = verify_gelu_drift(
                    model, info, img_batches, txt_batches, O_all, device
                )
                log_message(
                    f"  [GELU drift] cos_sim={gelu_cos_sim:.4f}  loss={gelu_loss:.6f}"
                )
        else:
            relu_labels.append(info.label)
            done_set.add(info.label)
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
            "gelu_cos_sim": round(gelu_cos_sim, 4) if not math.isnan(gelu_cos_sim) else float("nan"),
            "gelu_loss": round(gelu_loss, 6) if not math.isnan(gelu_loss) else float("nan"),
        }
        metrics_history.append(row)
        append_metrics_row(metrics_csv, row)
        append_metrics_jsonl(metrics_jsonl, row)
        plot_reconstruction_metrics(metrics_history, output_path.parent)

        _save_checkpoint(
            model,
            args.model_name,
            relu_labels,
            args.output + ".tmp",
            args.checkpoint_path,
        )

    _save_checkpoint(
        model, args.model_name, relu_labels, args.output, args.checkpoint_path
    )
    total_elapsed = time.time() - run_start
    warn_blocks = [r["block"] for r in metrics_history if r["status"] == "WARN"]
    reverted_blocks = [r["block"] for r in metrics_history if r["status"] == "REVERTED"]
    skipped_blocks = [r["block"] for r in metrics_history if r["status"] == "SKIPPED"]
    summary = (
        f"Done. {len(relu_labels)}/{len(blocks)} blocks reconstructed"
        f"  total={total_elapsed:.0f}s"
    )
    if warn_blocks:
        summary += f"  WARN: {warn_blocks}"
    if reverted_blocks:
        summary += f"  REVERTED: {reverted_blocks}"
    if skipped_blocks:
        summary += f"  SKIPPED(GELU): {skipped_blocks}"
    log_message(summary)
    log_message(f"Saved: {args.output}")

    plot_reconstruction_metrics(metrics_history, output_path.parent)

    if not args.skip_eval:
        from pipeline.eval_local import run_clip_retrieval_eval

        log_message("=== Auto eval after training ===")
        metrics = run_clip_retrieval_eval(
            root_dir=args.eval_root_dir,
            image_to_text_csv=args.eval_image_to_text_csv,
            textnums_to_texts_csv=args.eval_textnums_to_texts_csv,
            model_name=args.model_name,
            batch_size=args.eval_batch_size,
            k=args.eval_k,
            device=str(device),
            checkpoint_path=args.output,
        )
        for name, value in metrics.items():
            log_message(f"  {name}: {value:.4f}")
            print(f"{name}: {value:.4f}")
        append_metrics_jsonl(metrics_jsonl, {"block": "__eval__", **metrics})

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import open_clip
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.clip_utils import _load_clip
from train.data import ContrastiveRecordDataset, create_collate_fn
from train.distributed import (
    broadcast_run_timestamp,
    cleanup_distributed,
    init_distributed_context,
    seed_worker,
    set_seed,
)
from train.loss import SigLipLoss
from train.metrics import append_metrics_jsonl, append_metrics_row, plot_training_curves
from train.optim import build_scheduler
from train.train_step import train_one_epoch
from train.train_utils import (
    MetricsRow,
    build_run_name,
    current_timestamp,
    log_message,
    make_run_timestamp,
    save_checkpoint,
    save_run_config,
    set_log_path,
    unwrap_model,
    write_tensorboard_scalars,
)


def _log_model_structure(model, label: str, run_name: str) -> None:
    """Print model structure and parameter counts to the training log."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [
        f"=== Model structure [{label}] ===",
        str(model),
        f"--- Params: total={total:,}  trainable={trainable:,}  frozen={total - trainable:,} ---",
    ]
    log_message("\n".join(lines), run_name=run_name)


def _export_onnx_checkpoint(
    model,
    model_name: str,
    output_dir: Path,
    label: str,
    run_name: str,
) -> None:
    """Export image + text ONNX from a QAT-trained (or regular) model.

    QAT path: uses AIMET sim.export() to produce a proper ONNX with
    QuantizeLinear / DequantizeLinear nodes, then splits into per-encoder files.
    Non-QAT path: reparameterizes and exports fp32.
    """
    from timm.utils import reparameterize_model
    from pipeline.export_onnx import export_encoders_to_onnx, export_quantized_encoders_to_onnx

    raw_model = unwrap_model(model)
    qat_sim = getattr(raw_model, "_qat_sim", None)

    onnx_dir = str(output_dir / f"onnx_{label}")
    log_message(f"Exporting ONNX to {onnx_dir}", run_name=run_name)

    if qat_sim is not None:
        export_quantized_encoders_to_onnx(qat_sim, onnx_dir)
    else:
        base_model = reparameterize_model(raw_model).eval()
        export_encoders_to_onnx(base_model, onnx_dir)


def run_training(args) -> None:
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

    previous_log_path_value = None
    import train.train_utils as _tu
    previous_log_path_value = _tu.TRAIN_LOG_PATH
    set_log_path(train_log_path if is_main_process else None)
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
            _amp_reason = (
                "disabled (--no-amp)" if args.no_amp
                else "disabled (QAT requires fp32)" if getattr(args, "qat_enabled", False)
                else "enabled (fp16 autocast + GradScaler)"
            )
            log_message(f"AMP: {_amp_reason}", run_name=run_name)
            qat_enabled = getattr(args, "qat_enabled", False)
            if qat_enabled:
                _excl_parts = []
                if getattr(args, "qat_exclude_group_conv", False):
                    _excl_parts.append("group-conv")
                if getattr(args, "qat_exclude_names", None):
                    _excl_parts.append(f"names={args.qat_exclude_names}")
                _excl_str = f", exclude=[{', '.join(_excl_parts)}]" if _excl_parts else ""
                log_message(
                    f"QAT: enabled (W{args.qat_weight_bw}A{args.qat_act_bw}, "
                    f"scheme={args.qat_quant_scheme}, calib_samples={args.qat_calib_samples}"
                    f"{_excl_str})",
                    run_name=run_name,
                )
            log_message(f"Run outputs will be saved to {output_dir}", run_name=run_name)
            log_message(f"Train log saved to {train_log_path}", run_name=run_name)
            log_message(
                f"TensorBoard logs saved to {tensorboard_dir}",
                run_name=run_name,
            )
            log_message(f"Run config saved to {config_json_path}", run_name=run_name)

        # --- Build QAT config ---
        qat_config = None
        qat_enabled = getattr(args, "qat_enabled", False)
        if qat_enabled:
            from utils.qat_utils import QATConfig, GroupConvRule, NamePatternRule

            exclusion_rules = []
            if getattr(args, "qat_exclude_group_conv", False):
                exclusion_rules.append(GroupConvRule())
            raw_patterns = getattr(args, "qat_exclude_names", None)
            if raw_patterns:
                patterns = [p.strip() for p in raw_patterns.split(",") if p.strip()]
                if patterns:
                    exclusion_rules.append(NamePatternRule(patterns))

            qat_config = QATConfig(
                enabled=True,
                weight_bw=args.qat_weight_bw,
                act_bw=args.qat_act_bw,
                quant_scheme=args.qat_quant_scheme,
                calib_samples=args.qat_calib_samples,
                exclusion_rules=exclusion_rules,
            )

        # --- Model loading (QAT wrap happens inside _load_clip when qat_config set) ---
        model, _, tokenizer = _load_clip(
            args.model_name,
            device,
            checkpoint_path=getattr(args, "resume", None),
            pretrained=args.pretrained,
            qat_config=qat_config,
            relu_image=getattr(args, "relu_image", False),
            relu_text=getattr(args, "relu_text", False),
        )
        relu_labels = getattr(model, "_relu_blocks", None)

        if args.grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing()

        if is_main_process:
            _log_model_structure(model, "after _load_clip", run_name)

        # --- Freeze modules by name ---
        freeze_names_raw = getattr(args, "freeze_modules", None)
        if freeze_names_raw:
            freeze_names = [n.strip() for n in freeze_names_raw.split(",") if n.strip()]
            top_level_names = {name for name, _ in model.named_children()}
            all_module_names = dict(model.named_modules())
            frozen_params = 0
            for name in freeze_names:
                first_segment = name.split(".")[0]
                if first_segment not in top_level_names:
                    raise ValueError(
                        f"[freeze] '{name}' does not start with a top-level module of the model. "
                        f"Top-level modules: {sorted(top_level_names)}"
                    )
                submodule = all_module_names.get(name)
                if submodule is None:
                    raise ValueError(
                        f"[freeze] Module '{name}' not found in model. "
                        f"Check the full dotted path (e.g. visual.head, not just head)."
                    )
                for param in submodule.parameters():
                    if param.requires_grad:
                        param.requires_grad_(False)
                        frozen_params += param.numel()
                if is_main_process:
                    log_message(f"[freeze] Froze module: {name}", run_name=run_name)
            if is_main_process:
                _log_model_structure(model, "after freeze", run_name)

        # --- Dataset and DataLoader (built before DDP so calibration can use it) ---
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
        run_config["effective_global_batch_size"] = (
            args.batch_size * max(args.accum_freq, 1) * world_size
        )
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

        # --- QAT calibration (must happen before DDP wrap) ---
        if qat_config is not None and getattr(model, "_qat_needs_calibration", False):
            if is_main_process:
                log_message(
                    f"Running QAT calibration ({args.qat_calib_samples} samples)...",
                    run_name=run_name,
                )
            from utils.qat_utils import calibrate_quantsim
            calibrate_quantsim(model._qat_sim, train_dataloader, args.qat_calib_samples, device)
            if is_main_process:
                log_message("QAT calibration complete", run_name=run_name)

        # --- DDP wrap (after calibration) ---
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
                static_graph=True,
            )
            if is_main_process:
                _log_model_structure(model, "after DDP wrap", run_name)

        if args.loss_type == "siglip":
            loss_fn = SigLipLoss(rank=rank, world_size=world_size).to(device)
            trainable_params = (
                [p for p in model.parameters() if p.requires_grad]
                + list(loss_fn.parameters())
            )
        else:
            loss_fn = open_clip.ClipLoss(
                cache_labels=True,
                rank=rank,
                world_size=world_size,
            )
            trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = optim.AdamW(
            trainable_params,
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
        # QAT uses fp32 fake quantization; AMP fp16 casts conflict and double memory.
        amp_enabled = not args.no_amp and device.type == "cuda" and not (qat_config is not None)
        scaler = GradScaler(device=device.type, enabled=amp_enabled)

        metrics_history: List[MetricsRow] = []
        global_step = 0
        epoch_digits = max(len(str(args.epochs)), 2)
        previous_latest_path: Optional[Path] = None

        # Resolved once for convenience
        export_onnx = getattr(args, "export_onnx", False)

        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_start = time.time()
            epoch_metrics, global_step = train_one_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                loss_fn=loss_fn,
                loss_type=args.loss_type,
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
                amp_enabled=amp_enabled,
                run_name=run_name,
                writer=writer,
            )

            epoch_duration_s = time.time() - epoch_start

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
                "epoch_duration_s": epoch_duration_s,
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
                    {"global_step": global_step, "epoch_duration_s": epoch_duration_s},
                    epoch,
                )
                writer.flush()

                log_message(
                    f"Epoch {epoch} summary: "
                    f"total={epoch_metrics['train_total_loss']:.4f} "
                    f"clip={epoch_metrics['train_loss']:.4f} "
                    f"hardneg={epoch_metrics['train_hard_negative_loss']:.4f} "
                    f"time={epoch_duration_s:.1f}s",
                    run_name=run_name,
                )

                sim = getattr(unwrap_model(model), "_qat_sim", None)

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
                    sim=sim,
                    relu_labels=relu_labels,
                )
                previous_latest_path = latest_path

                save_n = args.save_every_n_epochs
                if save_n and save_n > 0 and epoch % save_n == 0:
                    epoch_path = (
                        output_dir / f"checkpoint_epoch_{epoch:0{epoch_digits}d}.pt"
                    )
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
                        sim=sim,
                        relu_labels=relu_labels,
                    )
                    if export_onnx:
                        _export_onnx_checkpoint(
                            model, args.model_name, output_dir,
                            label=f"epoch_{epoch:0{epoch_digits}d}",
                            run_name=run_name,
                        )

        final_weights_path = output_dir / f"{args.model_name}_finetuned.pt"
        if is_main_process:
            sim = getattr(unwrap_model(model), "_qat_sim", None)
            # Save final checkpoint (includes QAT state when applicable).
            save_checkpoint(
                save_path=final_weights_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=args.epochs,
                global_step=global_step,
                args=args,
                metrics_history=metrics_history,
                sim=sim,
                relu_labels=relu_labels,
            )
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
            if export_onnx:
                _export_onnx_checkpoint(
                    model, args.model_name, output_dir,
                    label="final",
                    run_name=run_name,
                )
    finally:
        if writer is not None:
            writer.close()
        set_log_path(previous_log_path_value)
        cleanup_distributed()

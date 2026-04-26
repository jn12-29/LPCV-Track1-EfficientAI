from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train.loss import compute_hard_negative_loss
from train.train_utils import StepTimeAccum, StepTimer, log_message, write_tensorboard_scalars


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    loss_fn,
    loss_type: str,
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
    amp_enabled: bool = True,
    enable_step_timing: bool = False,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[Dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = amp_enabled and device.type == "cuda"
    start_time = time.time()
    num_batches = len(dataloader)
    _cuda_sync = enable_step_timing and device.type == "cuda"
    _timers = StepTimeAccum() if enable_step_timing else None

    running_total_loss = 0.0
    running_clip_loss = 0.0
    running_hard_negative_loss = 0.0

    for batch_idx, batch in enumerate(dataloader, start=1):
        # --- Data transfer to GPU ---
        if enable_step_timing and _cuda_sync:
            torch.cuda.synchronize()
        _t_data = time.perf_counter() if enable_step_timing else 0.0
        images = batch["images"].to(device, non_blocking=True)
        positive_tokens = batch["positive_tokens"].to(device, non_blocking=True)
        negative_tokens = batch["negative_tokens"].to(device, non_blocking=True)
        negative_mask = batch["negative_mask"].to(device, non_blocking=True)
        if enable_step_timing:
            if _cuda_sync:
                torch.cuda.synchronize()
            _timers.update("data", time.perf_counter() - _t_data)

        with autocast(device_type=device.type, enabled=amp_enabled):
            # --- Forward pass (model inference, including negative encoding) ---
            if enable_step_timing and _cuda_sync:
                torch.cuda.synchronize()
            _t_fwd = time.perf_counter() if enable_step_timing else 0.0

            image_features, positive_features, logit_scale = model(
                images, positive_tokens
            )

            if loss_type == "siglip":
                negative_features_3d: Optional[torch.Tensor] = None
                if negative_tokens.numel() > 0:
                    flat_negative_tokens = negative_tokens.view(
                        -1, negative_tokens.shape[-1]
                    )
                    _, flat_negative_features, _ = model(text=flat_negative_tokens)
                    negative_features_3d = flat_negative_features.view(
                        negative_tokens.shape[0],
                        negative_tokens.shape[1],
                        -1,
                    )
                negative_features_clip: Optional[torch.Tensor] = None
            else:
                negative_features_3d = None
                if negative_tokens.numel() > 0:
                    flat_negative_tokens = negative_tokens.view(
                        -1, negative_tokens.shape[-1]
                    )
                    _, flat_negative_features, _ = model(text=flat_negative_tokens)
                    flat_negative_features = F.normalize(flat_negative_features, dim=-1)
                    negative_features_clip = flat_negative_features.view(
                        negative_tokens.shape[0],
                        negative_tokens.shape[1],
                        -1,
                    )
                else:
                    negative_features_clip = None

            if enable_step_timing:
                if _cuda_sync:
                    torch.cuda.synchronize()
                _timers.update("forward", time.perf_counter() - _t_fwd)

            # --- Loss computation ---
            if enable_step_timing and _cuda_sync:
                torch.cuda.synchronize()
            _t_loss = time.perf_counter() if enable_step_timing else 0.0

            if loss_type == "siglip":
                main_loss = loss_fn(
                    image_features,
                    positive_features,
                    logit_scale,
                    extra_text_features=negative_features_3d,
                    extra_text_mask=(
                        negative_mask if negative_features_3d is not None else None
                    ),
                )
                clip_loss = main_loss
                hard_negative_loss = main_loss.new_zeros(())
                total_loss = main_loss
            else:
                clip_loss = loss_fn(image_features, positive_features, logit_scale)

                image_features = F.normalize(image_features, dim=-1)
                positive_features = F.normalize(positive_features, dim=-1)
                positive_scores = (image_features * positive_features).sum(dim=-1)

                if negative_features_clip is not None:
                    hard_negative_loss = compute_hard_negative_loss(
                        image_features=image_features,
                        negative_features=negative_features_clip,
                        negative_mask=negative_mask,
                        positive_scores=positive_scores,
                        margin=hard_negative_margin,
                        strategy=hard_negative_loss_type,
                    )
                else:
                    hard_negative_loss = clip_loss.new_zeros(())

                total_loss = clip_loss + hard_negative_weight * hard_negative_loss

            loss_for_backward = total_loss / accum_freq

            if enable_step_timing:
                if _cuda_sync:
                    torch.cuda.synchronize()
                _timers.update("loss", time.perf_counter() - _t_loss)

        # --- Backward pass ---
        if enable_step_timing and _cuda_sync:
            torch.cuda.synchronize()
        _t_bwd = time.perf_counter() if enable_step_timing else 0.0
        scaler.scale(loss_for_backward).backward()
        if enable_step_timing:
            if _cuda_sync:
                torch.cuda.synchronize()
            _timers.update("backward", time.perf_counter() - _t_bwd)

        # --- Optimizer step ---
        if batch_idx % accum_freq == 0 or batch_idx == num_batches:
            if enable_step_timing and _cuda_sync:
                torch.cuda.synchronize()
            _t_opt = time.perf_counter() if enable_step_timing else 0.0
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
            if enable_step_timing:
                if _cuda_sync:
                    torch.cuda.synchronize()
                _timers.update("optim", time.perf_counter() - _t_opt)
            global_step += 1

        running_total_loss += total_loss.item()
        running_clip_loss += clip_loss.item()
        running_hard_negative_loss += hard_negative_loss.item()

        if is_main_process and (
            batch_idx % log_every_n_steps == 0 or batch_idx == num_batches
        ):
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
            if enable_step_timing:
                write_tensorboard_scalars(
                    writer,
                    "step_time_ms",
                    {k: v * 1000 for k, v in _timers.means().items()},
                    tb_step,
                )
            timing_str = _timers.summary_str() if enable_step_timing else ""
            log_message(
                f"Epoch {epoch} Step {batch_idx}/{num_batches} "
                f"Total {total_loss.item():.4f} "
                f"CLIP {clip_loss.item():.4f} "
                f"HardNeg {hard_negative_loss.item():.4f} "
                f"LR {current_lr:.6e} Throughput {throughput:.2f} samples/s"
                + timing_str,
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

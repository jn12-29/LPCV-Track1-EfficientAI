from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from train.loss import compute_hard_negative_loss
from train.train_utils import StepTimeAccum, log_message, write_tensorboard_scalars


@dataclass
class CachedBatch:
    # Raw inputs (already on device)
    images: torch.Tensor            # (B, ...)
    positive_tokens: torch.Tensor   # (B, L)
    negative_tokens: torch.Tensor   # (B, K, L)
    negative_mask: torch.Tensor     # (B, K)

    # Phase 1 outputs — detached, no grad
    img_features: Optional[torch.Tensor] = field(default=None)   # (B, D)
    pos_features: Optional[torch.Tensor] = field(default=None)   # (B, D)
    neg_features: Optional[torch.Tensor] = field(default=None)   # (B, K, D) or None

    # Phase 2 outputs — pre-computed feature gradients (GradScaler-scaled)
    img_grad: Optional[torch.Tensor] = field(default=None)       # (B, D)
    pos_grad: Optional[torch.Tensor] = field(default=None)       # (B, D)
    neg_grad: Optional[torch.Tensor] = field(default=None)       # (B, K, D) or None


def phase1_forward_no_grad(
    model,
    batches: List[CachedBatch],
    device: torch.device,
    amp_enabled: bool,
    loss_type: str,
) -> None:
    """No-grad forward for every batch. Populates features in-place.

    For CLIP, negative features are L2-normalised here (matching current train_step
    behaviour). For SigLIP, negatives are kept unnormalised (SigLipLoss normalises
    internally).
    """
    with torch.no_grad():
        for cb in batches:
            with autocast(device_type=device.type, enabled=amp_enabled):
                img_f, pos_f, _ = model(cb.images, cb.positive_tokens)
                cb.img_features = img_f.detach()
                cb.pos_features = pos_f.detach()

                if cb.negative_tokens.numel() > 0:
                    B, K, L = cb.negative_tokens.shape
                    flat_tok = cb.negative_tokens.view(B * K, L)
                    _, flat_neg, _ = model(text=flat_tok)
                    if loss_type != "siglip":
                        flat_neg = F.normalize(flat_neg, dim=-1)
                    cb.neg_features = flat_neg.view(B, K, -1).detach()
                else:
                    cb.neg_features = None


def phase2_feature_backward(
    batches: List[CachedBatch],
    loss_fn,
    loss_type: str,
    model,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    hard_negative_weight: float,
    hard_negative_margin: float,
    hard_negative_loss_type: str,
    distributed: bool,
    world_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute full-batch contrastive loss on leaf tensors; populate feature grads.

    logit_scale (and logit_bias for SigLIP) are accessed directly from the raw model,
    bypassing DDP, so their gradients are all-reduced manually at the end.

    Returns (total_loss, clip_loss, hard_negative_loss) tensors for logging.
    """
    # --- Build leaf tensors ---
    img_leaf = torch.cat([cb.img_features for cb in batches], dim=0).detach().requires_grad_(True)
    pos_leaf = torch.cat([cb.pos_features for cb in batches], dim=0).detach().requires_grad_(True)

    has_neg = all(cb.neg_features is not None for cb in batches)
    if has_neg:
        neg_leaf = torch.cat([cb.neg_features for cb in batches], dim=0).detach().requires_grad_(True)
        all_mask = torch.cat([cb.negative_mask for cb in batches], dim=0)
    else:
        neg_leaf = None
        all_mask = None

    # logit_scale with grad (not detached) so its gradient is updated here
    raw_model = model.module if hasattr(model, "module") else model
    logit_scale = raw_model.logit_scale.exp()

    # --- Compute loss ---
    with autocast(device_type=device.type, enabled=amp_enabled):
        if loss_type == "siglip":
            # neg_leaf: (N, K, D) — SigLipLoss normalises internally
            clip_loss = loss_fn(
                img_leaf, pos_leaf, logit_scale,
                extra_text_features=neg_leaf,
                extra_text_mask=all_mask,
            )
            hard_negative_loss = clip_loss.new_zeros(())
            total_loss = clip_loss
        else:
            clip_loss = loss_fn(img_leaf, pos_leaf, logit_scale)

            img_norm = F.normalize(img_leaf, dim=-1)
            pos_norm = F.normalize(pos_leaf, dim=-1)
            positive_scores = (img_norm * pos_norm).sum(dim=-1)

            if neg_leaf is not None:
                hard_negative_loss = compute_hard_negative_loss(
                    image_features=img_norm,
                    negative_features=neg_leaf,   # already normalised in Phase 1
                    negative_mask=all_mask,
                    positive_scores=positive_scores,
                    margin=hard_negative_margin,
                    strategy=hard_negative_loss_type,
                )
            else:
                hard_negative_loss = clip_loss.new_zeros(())
            total_loss = clip_loss + hard_negative_weight * hard_negative_loss

    scaler.scale(total_loss).backward()

    # --- Scatter feature gradients back to individual batches ---
    B = batches[0].img_features.shape[0]
    for i, cb in enumerate(batches):
        s, e = i * B, (i + 1) * B
        cb.img_grad = img_leaf.grad[s:e].detach().clone()
        cb.pos_grad = pos_leaf.grad[s:e].detach().clone()
        if neg_leaf is not None and neg_leaf.grad is not None:
            cb.neg_grad = neg_leaf.grad[s:e].detach().clone()

    # --- All-reduce params that were updated outside DDP ---
    if distributed:
        params_to_sync: List[torch.Tensor] = []
        if raw_model.logit_scale.grad is not None:
            params_to_sync.append(raw_model.logit_scale.grad)
        if loss_type == "siglip" and hasattr(loss_fn, "logit_bias"):
            if loss_fn.logit_bias.grad is not None:
                params_to_sync.append(loss_fn.logit_bias.grad)
        for g in params_to_sync:
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g /= world_size

    return total_loss.detach(), clip_loss.detach(), hard_negative_loss.detach()


def phase3_model_backward(
    model,
    batches: List[CachedBatch],
    loss_type: str,
    distributed: bool,
    amp_enabled: bool,
    device: torch.device,
) -> None:
    """Re-forward each mini-batch and backward with pre-computed feature gradients.

    Uses model.no_sync() on all but the last batch to suppress premature DDP
    all-reduce. The last batch's backward triggers the single collective sync.
    """
    n = len(batches)
    for i, cb in enumerate(batches):
        is_last = (i == n - 1)
        sync_ctx = contextlib.nullcontext() if (not distributed or is_last) else model.no_sync()

        with sync_ctx:
            # Image + positive text
            with autocast(device_type=device.type, enabled=amp_enabled):
                img_f, pos_f, _ = model(cb.images, cb.positive_tokens)
            img_f.backward(gradient=cb.img_grad)
            pos_f.backward(gradient=cb.pos_grad)

            # Negative text (only if we have gradients for them)
            if cb.neg_grad is not None:
                B, K, L = cb.negative_tokens.shape
                with autocast(device_type=device.type, enabled=amp_enabled):
                    _, flat_neg_f, _ = model(text=cb.negative_tokens.view(B * K, L))
                    if loss_type != "siglip":
                        # Phase 1 normalised CLIP negatives; mirror that here so
                        # cb.neg_grad is w.r.t. the same normalised tensor.
                        flat_neg_f = F.normalize(flat_neg_f, dim=-1)
                    neg_f = flat_neg_f.view(B, K, -1)
                neg_f.backward(gradient=cb.neg_grad)


def train_one_epoch_accum(
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
    world_size: int = 1,
    run_name: Optional[str] = None,
    amp_enabled: bool = True,
    enable_step_timing: bool = False,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[Dict[str, float], int]:
    """Cache-based gradient accumulation epoch loop.

    Contrastive pool per optimizer step = accum_freq × world_size × batch_size.
    """
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
    last_step_lr: float = optimizer.param_groups[0]["lr"]

    total_consumed = 0
    optimizer_step_idx = 0
    num_optimizer_steps = -(-num_batches // accum_freq)  # ceil division
    batch_iter = iter(dataloader)

    while total_consumed < num_batches:
        n = min(accum_freq, num_batches - total_consumed)

        # ---- Data transfer ----
        if enable_step_timing and _cuda_sync:
            torch.cuda.synchronize()
        _t_data = time.perf_counter() if enable_step_timing else 0.0

        batches: List[CachedBatch] = []
        for _ in range(n):
            raw = next(batch_iter)
            batches.append(CachedBatch(
                images=raw["images"].to(device, non_blocking=True),
                positive_tokens=raw["positive_tokens"].to(device, non_blocking=True),
                negative_tokens=raw["negative_tokens"].to(device, non_blocking=True),
                negative_mask=raw["negative_mask"].to(device, non_blocking=True),
            ))

        if enable_step_timing:
            if _cuda_sync:
                torch.cuda.synchronize()
            _timers.update("data", time.perf_counter() - _t_data)

        # ---- Phase 1: no-grad forward ----
        if enable_step_timing and _cuda_sync:
            torch.cuda.synchronize()
        _t_fwd = time.perf_counter() if enable_step_timing else 0.0
        phase1_forward_no_grad(model, batches, device, amp_enabled, loss_type)
        if enable_step_timing:
            if _cuda_sync:
                torch.cuda.synchronize()
            _timers.update("forward_p1", time.perf_counter() - _t_fwd)

        # ---- Phase 2: feature-level loss + backward ----
        if enable_step_timing and _cuda_sync:
            torch.cuda.synchronize()
        _t_loss = time.perf_counter() if enable_step_timing else 0.0
        total_loss, clip_loss, hard_negative_loss = phase2_feature_backward(
            batches=batches,
            loss_fn=loss_fn,
            loss_type=loss_type,
            model=model,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            hard_negative_weight=hard_negative_weight,
            hard_negative_margin=hard_negative_margin,
            hard_negative_loss_type=hard_negative_loss_type,
            distributed=distributed,
            world_size=world_size,
        )
        if enable_step_timing:
            if _cuda_sync:
                torch.cuda.synchronize()
            _timers.update("loss_p2", time.perf_counter() - _t_loss)

        # ---- Phase 3: model re-forward + backward ----
        if enable_step_timing and _cuda_sync:
            torch.cuda.synchronize()
        _t_bwd = time.perf_counter() if enable_step_timing else 0.0
        phase3_model_backward(model, batches, loss_type, distributed, amp_enabled, device)
        if enable_step_timing:
            if _cuda_sync:
                torch.cuda.synchronize()
            _timers.update("backward_p3", time.perf_counter() - _t_bwd)

        # ---- Optimizer step ----
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
        last_step_lr = optimizer.param_groups[0]["lr"]
        current_scale = scaler.get_scale() if amp_enabled else 1.0
        if not amp_enabled or current_scale >= previous_scale:
            scheduler.step()
        if enable_step_timing:
            if _cuda_sync:
                torch.cuda.synchronize()
            _timers.update("optim", time.perf_counter() - _t_opt)

        total_consumed += n
        optimizer_step_idx += 1
        global_step += 1

        running_total_loss += total_loss.item()
        running_clip_loss += clip_loss.item()
        running_hard_negative_loss += hard_negative_loss.item()

        if is_main_process and (
            optimizer_step_idx % log_every_n_steps == 0
            or optimizer_step_idx == num_optimizer_steps
        ):
            elapsed = time.time() - start_time
            throughput = (total_consumed * batches[0].images.shape[0]) / max(elapsed, 1e-6)
            tb_step = (epoch - 1) * num_optimizer_steps + optimizer_step_idx
            write_tensorboard_scalars(
                writer,
                "train_step",
                {
                    "total_loss": total_loss.item(),
                    "clip_loss": clip_loss.item(),
                    "hard_negative_loss": hard_negative_loss.item(),
                    "lr": last_step_lr,
                    "throughput": throughput,
                },
                tb_step,
            )
            if enable_step_timing and _timers:
                write_tensorboard_scalars(
                    writer,
                    "step_time_ms",
                    {k: v * 1000 for k, v in _timers.means().items()},
                    tb_step,
                )
            timing_str = _timers.summary_str() if enable_step_timing else ""
            log_message(
                f"Epoch {epoch} Step {optimizer_step_idx}/{num_optimizer_steps} "
                f"[accum={n}×{batches[0].images.shape[0]}] "
                f"Total {total_loss.item():.4f} "
                f"CLIP {clip_loss.item():.4f} "
                f"HardNeg {hard_negative_loss.item():.4f} "
                f"LR {last_step_lr:.6e} Throughput {throughput:.2f} samples/s"
                + timing_str,
                run_name=run_name,
            )

    # ---- Epoch-end: reduce stats across ranks ----
    reduced_stats = torch.tensor(
        [running_total_loss, running_clip_loss, running_hard_negative_loss,
         float(optimizer_step_idx)],
        device=device, dtype=torch.float64,
    )
    if distributed:
        dist.all_reduce(reduced_stats, op=dist.ReduceOp.SUM)

    total_steps = max(reduced_stats[3].item(), 1.0)
    return (
        {
            "train_total_loss": reduced_stats[0].item() / total_steps,
            "train_loss": reduced_stats[1].item() / total_steps,
            "train_hard_negative_loss": reduced_stats[2].item() / total_steps,
            "lr": last_step_lr,
        },
        global_step,
    )

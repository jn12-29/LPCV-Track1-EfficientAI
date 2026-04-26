from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp_reconstruction.mlp_blocks import MLPBlockInfo, replace_gelu_with_relu
from mlp_reconstruction.aph import compute_aph_weights


def _get_hbar_view(H_bar: torch.Tensor, O: torch.Tensor) -> torch.Tensor:
    if O.ndim == 4:  # (B, C, H, W) — FastVit
        return H_bar.view(1, -1, 1, 1)
    return H_bar.view(*([1] * (O.ndim - 1)), -1)  # (B, L, D) or (B, D)


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, H_bar_view: torch.Tensor) -> torch.Tensor:
    return ((target - pred) ** 2 * H_bar_view).mean()


def distill_mlp(
    model: nn.Module,
    info: MLPBlockInfo,
    X_all: torch.Tensor,
    O_all: torch.Tensor,
    lr: float = 1e-3,
    batch_size: int = 32,
    n_iters: int = 20000,
    alpha: float = 2.0,
    gelu_ub: float | None = None,
    device: torch.device = torch.device('cuda'),
    log_every: int = 500,
    aph_mode: str = 'uniform',
    log_fn=print,
    early_stop_patience: int = 5,
    early_stop_delta: float = 5e-4,
) -> float:
    """Distill a single MLP block in-place (GELU→ReLU + fit weights). Returns final loss.

    Early stopping: checks every log_every steps. If EMA loss does not improve by
    more than early_stop_delta (relative) for early_stop_patience consecutive checks,
    training stops. Set early_stop_patience=0 to disable.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    if info.kind == 'conv_stem':
        # Optimize all ConvStem params jointly; ConvStem[2] (no activation) acts as the
        # compensating projection for the GELU→ReLU change in layers 0 and 1.
        for p in info.mlp.parameters():
            p.requires_grad_(True)
    else:
        for p in info.fc1.parameters():
            p.requires_grad_(True)
        for p in info.fc2.parameters():
            p.requires_grad_(True)

    replace_gelu_with_relu(info)

    H_bar = compute_aph_weights(O_all, mode=aph_mode).to(device)
    X_gpu = X_all.to(device)
    O_gpu = O_all.to(device)
    N = X_gpu.shape[0]

    if info.kind == 'conv_stem':
        params = list(info.mlp.parameters())
    else:
        params = list(info.fc1.parameters()) + list(info.fc2.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=0.)

    model.eval()
    final_loss = 0.0
    ema_loss: float | None = None
    prev_check_ema: float | None = None
    patience_counter = 0
    steps_run = n_iters
    start_time = time.time()

    for step in range(n_iters):
        idx = torch.randint(0, N, (batch_size,), device=device)
        X_b = X_gpu[idx]
        O_b = O_gpu[idx]

        H_view = _get_hbar_view(H_bar, O_b)

        if info.kind == 'conv_stem':
            # Full ConvStem forward (ReLU already installed in layers 0 and 1)
            O_direct = info.mlp(X_b)
            loss = _weighted_mse(O_direct, O_b, H_view)
        else:
            h = info.fc1(X_b)
            act = F.relu(h)
            O_direct = info.fc2(act)

            L_direct = _weighted_mse(O_direct, O_b, H_view)

            if alpha > 0:
                if gelu_ub is not None:
                    threshold = act.new_tensor(gelu_ub)
                else:
                    pos_vals = act[act > 0]
                    threshold = torch.quantile(pos_vals, 0.99) if pos_vals.numel() > 0 else act.new_tensor(1.0)
                A_clamped = act.clamp(max=threshold)
                O_clamp = info.fc2(A_clamped)
                loss = L_direct + alpha * _weighted_mse(O_clamp, O_b, H_view)
            else:
                loss = L_direct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss = loss.item()
        ema_loss = final_loss if ema_loss is None else 0.98 * ema_loss + 0.02 * final_loss

        at_check = log_every > 0 and step % log_every == 0
        if at_check or step == n_iters - 1:
            elapsed = time.time() - start_time
            throughput = batch_size * (step + 1) / max(elapsed, 1e-6)
            current_lr = scheduler.get_last_lr()[0]
            parts = [
                f"  [{info.label}] {step:>6}/{n_iters}",
                f"loss={final_loss:.6f}",
                f"ema={ema_loss:.6f}",
                f"lr={current_lr:.2e}",
            ]
            if prev_check_ema is not None:
                rel_impr = (prev_check_ema - ema_loss) / (prev_check_ema + 1e-10)
                parts.append(f"rel={rel_impr:+.2%}")
                if early_stop_patience > 0:
                    parts.append(f"p={patience_counter}/{early_stop_patience}")
            parts.append(f"{elapsed:.0f}s  {throughput:.0f} samples/s")
            log_fn("  ".join(parts))

        if at_check and early_stop_patience > 0 and prev_check_ema is not None:
            rel_impr = (prev_check_ema - ema_loss) / (prev_check_ema + 1e-10)
            if rel_impr < early_stop_delta:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    elapsed = time.time() - start_time
                    log_fn(f"  [{info.label}] Early stop at step {step}"
                           f"  p={patience_counter}/{early_stop_patience}"
                           f"  rel={rel_impr:.2%}  elapsed={elapsed:.0f}s")
                    steps_run = step + 1
                    break
            else:
                patience_counter = 0
        if at_check:
            prev_check_ema = ema_loss

    for p in model.parameters():
        p.requires_grad_(False)

    return final_loss, steps_run


@torch.no_grad()
def verify_reconstruction(
    info: MLPBlockInfo,
    X_all: torch.Tensor,
    O_GELU: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Compute mean cosine similarity between O_GELU and ReLU-MLP output on calibration data."""
    cos_sims: list[float] = []
    N = X_all.shape[0]
    for i in range(0, N, batch_size):
        X_b = X_all[i : i + batch_size].to(device)
        O_b = O_GELU[i : i + batch_size].to(device)
        if info.kind == 'conv_stem':
            O_pred = info.mlp(X_b)
        else:
            h = info.fc1(X_b)
            O_pred = info.fc2(F.relu(h))
        O_flat = O_b.reshape(O_b.shape[0], -1)
        P_flat = O_pred.reshape(O_pred.shape[0], -1)
        cos = F.cosine_similarity(O_flat, P_flat, dim=1).mean().item()
        cos_sims.append(cos)
    return sum(cos_sims) / len(cos_sims)

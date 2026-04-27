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
    keep_activation: bool = False,
) -> tuple[float, int]:
    """Distill a single MLP block in-place. Returns (final_loss, steps_run).

    keep_activation=True: skip GELU→ReLU replacement, optimize fc1/fc2 keeping the
    current activation (used for GELU drift distillation after a revert). L_Clamp is
    disabled in this mode (alpha is ignored).

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

    if not keep_activation:
        replace_gelu_with_relu(info)
    act_fn = None if info.kind == 'conv_stem' else getattr(info.mlp, info.act_attr)

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

        if info.kind == 'conv_stem' or keep_activation:
            if info.kind == 'conv_stem':
                O_direct = info.mlp(X_b)
            else:
                O_direct = info.fc2(act_fn(info.fc1(X_b)))
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
    """Compute mean cosine similarity between O_GELU and MLP output on calibration data.

    Uses the module's actual activation function (whatever is currently installed),
    so this is correct for both ReLU-reconstructed and GELU-distilled blocks.
    """
    cos_sims: list[float] = []
    N = X_all.shape[0]
    for i in range(0, N, batch_size):
        X_b = X_all[i : i + batch_size].to(device)
        O_b = O_GELU[i : i + batch_size].to(device)
        if info.kind == 'conv_stem':
            O_pred = info.mlp(X_b)
        else:
            act_fn = getattr(info.mlp, info.act_attr)
            h = info.fc1(X_b)
            O_pred = info.fc2(act_fn(h))
        O_flat = O_b.reshape(O_b.shape[0], -1)
        P_flat = O_pred.reshape(O_pred.shape[0], -1)
        cos = F.cosine_similarity(O_flat, P_flat, dim=1).mean().item()
        cos_sims.append(cos)
    return sum(cos_sims) / len(cos_sims)


@torch.no_grad()
def verify_gelu_drift(
    model: nn.Module,
    info: MLPBlockInfo,
    img_batches: list[torch.Tensor] | None,
    txt_batches: list[torch.Tensor] | None,
    O_orig: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[float, float, torch.Tensor]:
    """Measure output drift of a GELU-restored block due to upstream ReLU substitutions.

    Called after restore_gelu(info) in non-greedy mode. Re-collects the block's actual
    output using the current model (upstream blocks already ReLU) and compares it to the
    original GELU output O_orig captured before any distillation.
    Returns (cos_sim, mse_loss, X_cur).

    X_cur is the block's actual input under the current (partially-replaced) model,
    returned so the caller can reuse it for GELU distillation without a second inference pass.
    """
    from mlp_reconstruction.calibrate import collect_mlp_io

    X_cur, O_cur, _ = collect_mlp_io(model, info, img_batches, txt_batches, device)

    cos_sims: list[float] = []
    mse_losses: list[float] = []
    N = O_cur.shape[0]
    for i in range(0, N, batch_size):
        O_b = O_orig[i : i + batch_size].to(device)
        P_b = O_cur[i : i + batch_size].to(device)
        O_flat = O_b.reshape(O_b.shape[0], -1)
        P_flat = P_b.reshape(P_b.shape[0], -1)
        cos_sims.append(F.cosine_similarity(O_flat, P_flat, dim=1).mean().item())
        mse_losses.append(F.mse_loss(P_flat, O_flat).item())
    return sum(cos_sims) / len(cos_sims), sum(mse_losses) / len(mse_losses), X_cur

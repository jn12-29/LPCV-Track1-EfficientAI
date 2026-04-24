from __future__ import annotations

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
    device: torch.device = torch.device('cuda'),
    log_every: int = 500,
    aph_mode: str = 'uniform',
) -> float:
    """Distill a single MLP block in-place (GELU→ReLU + fit weights). Returns final loss."""
    for p in model.parameters():
        p.requires_grad_(False)

    for p in info.fc1.parameters():
        p.requires_grad_(True)
    for p in info.fc2.parameters():
        p.requires_grad_(True)

    replace_gelu_with_relu(info)

    H_bar = compute_aph_weights(O_all, mode=aph_mode).to(device)
    X_gpu = X_all.to(device)
    O_gpu = O_all.to(device)
    N = X_gpu.shape[0]

    params = list(info.fc1.parameters()) + list(info.fc2.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    model.eval()
    final_loss = 0.0
    for step in range(n_iters):
        idx = torch.randint(0, N, (batch_size,), device=device)
        X_b = X_gpu[idx]
        O_b = O_gpu[idx]

        h = info.fc1(X_b)
        act = F.relu(h)
        O_direct = info.fc2(act)

        H_view = _get_hbar_view(H_bar, O_b)
        L_direct = _weighted_mse(O_direct, O_b, H_view)

        pos_vals = act[act > 0]
        threshold = torch.quantile(pos_vals, 0.99) if pos_vals.numel() > 0 else act.new_tensor(1.0)
        A_clamped = act.clamp(max=threshold)
        O_clamp = info.fc2(A_clamped)
        L_clamp = _weighted_mse(O_clamp, O_b, H_view)

        loss = L_direct + alpha * L_clamp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if log_every > 0 and (step % log_every == 0 or step == n_iters - 1):
            print(f"  [{info.label}] {step:>6}/{n_iters}: loss={final_loss:.6f}")

    for p in model.parameters():
        p.requires_grad_(False)

    return final_loss


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
        h = info.fc1(X_b)
        O_pred = info.fc2(F.relu(h))
        O_flat = O_b.reshape(O_b.shape[0], -1)
        P_flat = O_pred.reshape(O_pred.shape[0], -1)
        cos = F.cosine_similarity(O_flat, P_flat, dim=1).mean().item()
        cos_sims.append(cos)
    return sum(cos_sims) / len(cos_sims)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp_reconstruction.mlp_blocks import MLPBlockInfo


@torch.no_grad()
def verify_reconstruction(
    info: MLPBlockInfo,
    X_all: torch.Tensor,
    O_GELU: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Mean cosine similarity between O_GELU and current MLP output on calibration data."""
    cos_sims: list[float] = []
    N = X_all.shape[0]
    for i in range(0, N, batch_size):
        X_b = X_all[i : i + batch_size].to(device)
        O_b = O_GELU[i : i + batch_size].to(device)
        O_pred = info.forward(X_b)
        cos = F.cosine_similarity(
            O_b.reshape(O_b.shape[0], -1),
            O_pred.reshape(O_pred.shape[0], -1),
            dim=1,
        ).mean().item()
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
    """Measure output drift of a GELU block due to upstream ReLU substitutions.

    Re-collects the block's actual output under the current (partially-replaced)
    model and compares it to O_orig (original all-GELU output).
    Returns (cos_sim, mse_loss, X_cur).

    X_cur is returned so the caller can reuse it for GELU distillation without
    a second inference pass.
    """
    from mlp_reconstruction.collect import collect_mlp_io

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

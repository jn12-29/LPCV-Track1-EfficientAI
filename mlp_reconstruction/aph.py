from __future__ import annotations

import torch


def compute_aph_weights(O_GELU: torch.Tensor, mode: str = 'uniform') -> torch.Tensor:
    """Compute H_bar importance weights for MLP output dimensions.

    Args:
        O_GELU: collected GELU outputs; shape (N,L,D) for text/ViT or (N,C,H,W) for FastVit
        mode: 'uniform' → all ones; 'magnitude' → mean absolute value per channel

    Returns:
        H_bar: shape (D,) for text/ViT or (C,) for FastVit
    """
    if O_GELU.ndim == 4:  # FastVit: (N, C, H, W)
        if mode == 'uniform':
            return torch.ones(O_GELU.shape[1], dtype=O_GELU.dtype)
        return O_GELU.abs().mean(dim=(0, 2, 3))  # (C,)
    else:  # text/ViT: (N, L, D)
        if mode == 'uniform':
            return torch.ones(O_GELU.shape[-1], dtype=O_GELU.dtype)
        return O_GELU.abs().mean(dim=(0, 1))  # (D,)

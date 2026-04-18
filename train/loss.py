from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def compute_hard_negative_loss(
    image_features: torch.Tensor,
    negative_features: torch.Tensor,
    negative_mask: torch.Tensor,
    positive_scores: torch.Tensor,
    margin: float,
    strategy: str,
) -> torch.Tensor:
    if negative_features.numel() == 0 or negative_mask.numel() == 0:
        return image_features.new_zeros(())

    negative_scores = torch.einsum("bd,bnd->bn", image_features, negative_features)
    active_mask = negative_mask > 0
    if not active_mask.any():
        return image_features.new_zeros(())

    if strategy == "hinge":
        margins = margin + negative_scores - positive_scores.unsqueeze(1)
        losses = F.relu(margins) * negative_mask
        return losses.sum() / negative_mask.sum().clamp_min(1.0)

    if strategy == "logsigmoid":
        diffs = positive_scores.unsqueeze(1) - negative_scores
        losses = -F.logsigmoid(diffs / max(margin, 1e-6)) * negative_mask
        return losses.sum() / negative_mask.sum().clamp_min(1.0)

    raise ValueError(f"Unsupported hard negative loss strategy: {strategy}")


def _siglip_chunk_loss(
    img_chunk: torch.Tensor,
    all_txt: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    pos_col_start: int,
    n_local: int,
    extra_mask_chunk: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute SigLIP loss for a chunk of image rows against all text columns.

    pos_col_start: column index where this chunk's positive pairs begin (rank_offset).
    n_local: total number of positive text columns (= global batch of positives).
    extra_mask_chunk: (chunk, n_local * K) mask for hard negatives, or None.
    """
    logits = logit_scale * img_chunk @ all_txt.T + logit_bias  # (chunk, N_txt_total)
    chunk = img_chunk.shape[0]

    # Build labels: -1 everywhere, +1 on the diagonal within the positive block
    labels = torch.full_like(logits, -1.0)
    for i in range(chunk):
        col = pos_col_start + i
        if col < n_local:
            labels[i, col] = 1.0

    if extra_mask_chunk is not None:
        # Hard-negative columns start after n_local positives
        loss_mask = torch.ones_like(logits)
        loss_mask[:, n_local:] = extra_mask_chunk
        loss = -F.logsigmoid(labels * logits)
        return (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

    return -F.logsigmoid(labels * logits).mean()


class SigLipLoss(nn.Module):
    """Sigmoid pairwise loss (SigLIP — Zhai et al. 2023).

    Key techniques implemented:
    - All-gather across GPUs so every rank sees the full global batch of negatives.
    - Chunked bidirectional loss (img→txt and txt→img) for memory efficiency.
    - Hard negatives appended as extra text columns (absorbed into main loss).
    - logit_bias learned parameter initialized to -10.
    """

    def __init__(self, rank: int = 0, world_size: int = 1, chunk_size: int = 64):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.chunk_size = chunk_size
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    @staticmethod
    def _all_gather(x: torch.Tensor, world_size: int) -> torch.Tensor:
        """Gather tensor from all ranks; keeps gradient on the local slice."""
        if world_size == 1:
            return x
        gathered = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x)
        return x

    @staticmethod
    def _all_gather_with_grad(x: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
        """Gather across all ranks while preserving gradients via autograd."""
        if world_size == 1:
            return x
        gathered = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x.detach())
        gathered[rank] = x  # re-inject local tensor to keep grad
        return torch.cat(gathered, dim=0)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        extra_text_features: Optional[torch.Tensor] = None,
        extra_text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        img_local = F.normalize(image_features, dim=-1)
        txt_local = F.normalize(text_features, dim=-1)
        n_local = img_local.shape[0]

        # All-gather positive features across GPUs
        img_global = self._all_gather_with_grad(img_local, self.world_size, self.rank)
        txt_global = self._all_gather_with_grad(txt_local, self.world_size, self.rank)
        n_global = img_global.shape[0]  # world_size * n_local
        rank_offset = self.rank * n_local  # column of this rank's positives

        # Append hard negatives to text columns (local, not gathered)
        if extra_text_features is not None and extra_text_mask is not None:
            extra = F.normalize(extra_text_features, dim=-1)  # (n_local, K, D)
            K = extra.shape[1]
            flat_extra = extra.view(-1, extra.shape[-1])  # (n_local*K, D)
            all_txt = torch.cat([txt_global, flat_extra], dim=0)  # (n_global + n_local*K, D)
            flat_mask_local = extra_text_mask.view(1, -1).expand(n_local, -1)  # (n_local, n_local*K)
        else:
            all_txt = txt_global
            flat_mask_local = None
            K = 0

        total_loss = img_local.new_zeros(())
        n_chunks = 0

        # --- image → text direction (local images vs all texts) ---
        for start in range(0, n_local, self.chunk_size):
            end = min(start, n_local - 1) + self.chunk_size
            end = min(end, n_local)
            chunk_img = img_local[start:end]
            chunk_mask = flat_mask_local[start:end] if flat_mask_local is not None else None
            total_loss = total_loss + _siglip_chunk_loss(
                chunk_img, all_txt, logit_scale, self.logit_bias,
                pos_col_start=rank_offset + start,
                n_local=n_global,
                extra_mask_chunk=chunk_mask,
            )
            n_chunks += 1

        # --- text → image direction (local texts vs all images, no hard negatives) ---
        for start in range(0, n_local, self.chunk_size):
            end = min(start, n_local - 1) + self.chunk_size
            end = min(end, n_local)
            chunk_txt = txt_local[start:end]
            total_loss = total_loss + _siglip_chunk_loss(
                chunk_txt, img_global, logit_scale, self.logit_bias,
                pos_col_start=rank_offset + start,
                n_local=n_global,
                extra_mask_chunk=None,
            )
            n_chunks += 1

        return total_loss / max(n_chunks, 1)

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import torch
from torch import nn
from transformers.models.gemma.modeling_gemma import GemmaMLP


def apply_rope_direct(
    x: torch.Tensor,
    rope_emb_sin: torch.Tensor,
    rope_emb_cos: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE using precomputed sin/cos. Simplified version for direct application.

    Parameters
    ----------
    x
        [B, L, H, D] (D even)
    rope_emb_sin
        [B, L, 1, D/2] (float32)
    rope_emb_cos
        [B, L, 1, D/2] (float32)

    Returns
    -------
    rotated_x : torch.Tensor
        [B, L, H, D] (same dtype as input)
    """
    d_half = x.shape[-1] // 2
    x1, x2 = x.split(d_half, dim=-1)
    part1 = x1 * rope_emb_cos - x2 * rope_emb_sin
    part2 = x2 * rope_emb_cos + x1 * rope_emb_sin
    return torch.cat([part1, part2], dim=-1)


def adapt_model() -> None:
    """
    Entry method to apply all necessary adaptations for on-device
    deployment.
    """


class GemmaMLPSplitLinear(torch.nn.Module):
    """
    Wrap a GemmaMLP and replace its large Linear layers with multiple
    smaller Linear layers, each with out_features or in_features at most
    max_mlp_dim. This helps on-device ML by avoiding a single large
    projection (e.g., 16384).

    Splitting strategy:
      - gate_proj, up_proj: split along out_features and process
        per-chunk.
      - down_proj: split along in_features and process per-chunk, then
        sum partial outputs.

    Memory-friendly forward:
      - Delay concatenation and keep tensors small. For each chunk, run
        gate/up projections, apply activation on the small gate chunk,
        do elementwise product with the small up chunk, pass through the
        matching down chunk, and accumulate the result. This avoids
        materializing the full 16384-wide mid activation.
    """

    def __init__(self, model: GemmaMLP, max_mlp_dim: int = 2048) -> None:
        super().__init__()
        self.hidden_size: int = model.hidden_size
        self.intermediate_size: int = model.intermediate_size
        self.act_fn = model.act_fn

        # Preserve config reference if present.
        self.config = getattr(model, "config", None)

        # Source weights / metadata.
        device = model.gate_proj.weight.device
        dtype = model.gate_proj.weight.dtype

        in_f: int = self.hidden_size
        mid_f: int = self.intermediate_size
        out_f: int = self.hidden_size

        def _make_chunks(total_n: int, max_n: int) -> list[int]:
            sizes: list[int] = []
            rem = total_n
            while rem > 0:
                step = min(max_n, rem)
                sizes.append(step)
                rem -= step
            return sizes

        # Build chunks for the intermediate dim.
        mid_chunks: list[int] = _make_chunks(mid_f, max_mlp_dim)

        # -------- gate_proj and up_proj (in -> mid; split by out) -----
        self.gate_proj_chunks = nn.ModuleList()
        self.up_proj_chunks = nn.ModuleList()

        start = 0
        for sz in mid_chunks:
            gate = nn.Linear(in_f, sz, bias=False, device=device, dtype=dtype)
            up = nn.Linear(in_f, sz, bias=False, device=device, dtype=dtype)

            with torch.no_grad():
                g_w = model.gate_proj.weight[start : start + sz, :]
                u_w = model.up_proj.weight[start : start + sz, :]
                gate.weight.copy_(g_w)
                up.weight.copy_(u_w)

            self.gate_proj_chunks.append(gate)
            self.up_proj_chunks.append(up)
            start += sz

        # -------- down_proj (mid -> out; split by in, then sum) --------
        self.down_proj_chunks = nn.ModuleList()

        start = 0
        for sz in mid_chunks:
            down = nn.Linear(sz, out_f, bias=False, device=device, dtype=dtype)
            with torch.no_grad():
                # Original W shape: [out_f, mid_f]. Slice cols.
                w_slice = model.down_proj.weight[:, start : start + sz]
                down.weight.copy_(w_slice)
            self.down_proj_chunks.append(down)
            start += sz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute: down_proj(act(gate_proj(x)) * up_proj(x)) using chunked
        projections. Keeps tensors small by processing one chunk at a
        time and accumulating the output, avoiding large concatenations.

        Parameters
        ----------
        x
            Input tensor [..., hidden_size].

        Returns
        -------
        result : torch.Tensor
            Output tensor [..., hidden_size].
        """
        y_accum: torch.Tensor | None = None

        # Process each (gate, up, down) chunk trio independently to
        # avoid building a full 16384-wide activation.
        for gate, up, down in zip(
            self.gate_proj_chunks,
            self.up_proj_chunks,
            self.down_proj_chunks,
            strict=False,
        ):
            g = gate(x)  # [..., sz]
            u = up(x)  # [..., sz]
            mid = self.act_fn(g)  # apply activation while tensor is small
            mid = mid * u  # GLU-like elementwise product

            out_chunk = down(mid)  # [..., out_f]
            y_accum = out_chunk if y_accum is None else y_accum + out_chunk

        # y_accum is guaranteed to be set since there is at least one
        # chunk (intermediate_size > 0 in valid configs).
        assert y_accum is not None
        return y_accum

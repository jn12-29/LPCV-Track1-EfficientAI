# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import sys
import types
from typing import Any, cast

import torch
import torch.nn.functional as F


def _convex_upsample(
    self: Any, flow: torch.Tensor, mask: torch.Tensor, rate: int = 4
) -> torch.Tensor:
    """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination.

    Replaces the original implementation's 7D/6D-reshape approach with a
    conv2d-based gather.

    Parameters
    ----------
    self
        Module instance.
    flow
        Flow field of shape ``[N, 2, H, W]``.
    mask
        Raw (un-normalised) convex weights of shape ``[N, rate*rate*9, H, W]``.
    rate
        Upsample factor (default 4 -> 1/4 resolution -> full resolution).

    Returns
    -------
    torch.Tensor
        Upsampled flow of shape ``[N, 2, rate*H, rate*W]``.
    """
    N, _, H, W = flow.shape
    # Reshape mask to [N, H*W, rate*rate, 9] then softmax over the 9 neighbours.
    mask = (
        mask.permute(0, 2, 3, 1).view(N, H * W, 9, rate * rate).permute(0, 1, 3, 2)
    )  # [N, H*W, rate*rate, 9]
    mask = torch.softmax(mask, dim=-1)
    # Gather the 9-neighbourhood for every flow pixel via conv2d: [N*2, H*W, 1, 9]
    up_flow = (
        F.conv2d(
            (rate * flow).view(N * 2, 1, H, W),
            self._unfold_kernel,
            padding=1,
        )
        .permute(0, 2, 3, 1)
        .view(N * 2, H * W, 1, 9)
    )  # [N*2, H*W, 1, 9]
    # Weighted sum over neighbours → [N*2, H*W, rate*rate]
    up_flow = (mask * up_flow).sum(dim=-1)  # broadcast N*2 over N
    # Fold back to spatial layout: [N*2, H, W, rate, rate] → [N*2, H, rate, W, rate]
    up_flow = up_flow.reshape(N * 2, H, W, rate, rate).permute(0, 1, 3, 2, 4)
    return up_flow.reshape(N, 2, rate * H, rate * W)


def _linear_attention_forward(
    self: Any,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    q_mask: torch.Tensor | None = None,
    kv_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-Head linear attention proposed in "Transformers are RNNs".

    Replaces the original einsum-based implementation with explicit matmul
    ops.

    Parameters
    ----------
    self
        Module instance.
    queries
        Query tensor of shape ``[N, L, H, D]``.
    keys
        Key tensor of shape ``[N, S, H, D]``.
    values
        Value tensor of shape ``[N, S, H, D]``.
    q_mask
        Optional query padding mask of shape ``[N, L]``.
    kv_mask
        Optional key/value padding mask of shape ``[N, S]``.

    Returns
    -------
    torch.Tensor
        Attended output of shape ``[N, L, H, D]``.
    """
    Q = self.feature_map(queries)
    K = self.feature_map(keys)

    # Zero out padded positions.
    if q_mask is not None:
        Q = Q * q_mask[:, :, None, None]
    if kv_mask is not None:
        K = K * kv_mask[:, :, None, None]
        values = values * kv_mask[:, :, None, None]

    v_length = values.size(1)
    v_length_inv = 1.0 / v_length  # avoid fp16 overflow
    values = values * v_length_inv

    # KV = K^T @ V  per head: [N, H, D, V]
    # K: [N, S, H, D] -> [N, H, D, S]
    # values: [N, S, H, V] -> [N, H, S, V]
    KV = torch.matmul(K.permute(0, 2, 3, 1), values.permute(0, 2, 1, 3))

    # Normalisation denominator Z: [N, L, H]
    # K_reduced = sum_S K: [N, H, D]
    # Q: [N, L, H, D] -> [N, H, L, D]
    N, L, H, D = Q.shape
    Q_reshaped = Q.permute(0, 2, 1, 3)  # [N, H, L, D]
    K_reduced = K.sum(dim=1).reshape(N, H, D, 1)  # [N, H, D, 1]
    # [N, H, L, D] @ [N, H, D, 1] -> [N, H, L, 1] -> [N, L, H]
    denom = torch.matmul(Q_reshaped, K_reduced).reshape(N, H, L).permute(0, 2, 1)
    Z = 1.0 / torch.clamp(denom, min=self.eps)  # [N, L, H]

    # Attended output: Q @ KV per head, then normalise.
    # [N, H, L, D] @ [N, H, D, V] -> [N, H, L, V] -> [N, L, H, V]
    out = torch.matmul(Q_reshaped, KV).permute(0, 2, 1, 3)  # [N, L, H, V]
    queried_values = out * Z.reshape(N, L, H, 1) * v_length

    return queried_values.contiguous()


def _bilinear_sampler(
    img: torch.Tensor,
    coords: torch.Tensor,
    mode: str = "bilinear",
    mask: bool = False,
) -> torch.Tensor:
    """Wrapper for F.grid_sample using pixel coordinates.

    Replaces the original ``bilinear_grid_sample`` call with ``F.grid_sample``
    which is directly supported by QNN.

    Parameters
    ----------
    img
        Input feature map of shape ``(N, C, H, W)``.
    coords
        Pixel coordinates of shape ``(N, Hg, Wg, 2)``.
    mode
        Interpolation mode (default: ``"bilinear"``).
    mask
        If ``True``, also return a validity mask.

    Returns
    -------
    torch.Tensor
        Sampled tensor.
    """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, align_corners=True)


def _corr_att_offset(
    self: Any,
    left_feature: torch.Tensor,
    right_feature: torch.Tensor,
    flow: torch.Tensor,
    extra_offset: torch.Tensor,
    small_patch: bool = False,
) -> torch.Tensor:
    """Compute attention-guided offset correlation.

    Replaces the original implementation to avoid export-unfriendly ops:

    - ``repeat_interleave`` on offsets (runtime-dependent count).
    - 5D tensor reshape ``[N, C, -1, H, W]`` unsupported by QNN.
    - ``repeat_interleave`` on ``left_feature``.

    Instead, ``right_feature`` is reshaped to ``[N*C, -1, H, W]``,
    ``left_feature`` is transposed to align, and ``mean`` is reduced
    over ``dim=0``.

    Parameters
    ----------
    self
        Module instance.
    left_feature
        Left feature map of shape ``[N, C, H, W]``.
    right_feature
        Right feature map of shape ``[N, C, H, W]``.
    flow
        Current flow estimate of shape ``[N, 2, H, W]``.
    extra_offset
        Learned offset of shape ``[N, search_num*2, H, W]``.
    small_patch
        If ``True``, use 3x3 patches; otherwise use 1x9 patches.

    Returns
    -------
    torch.Tensor
        Correlation volume of shape ``[N, 4*search_num, H, W]``.
    """
    N, C, H, W = left_feature.shape

    if self.att is not None:
        left_feature = left_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)
        right_feature = right_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)
        left_feature, right_feature = self.att(left_feature, right_feature)
        left_feature, right_feature = [
            x.reshape(N, H, W, C).permute(0, 3, 1, 2)
            for x in [left_feature, right_feature]
        ]

    lefts = torch.split(left_feature, left_feature.shape[1] // 4, dim=1)
    rights = torch.split(right_feature, right_feature.shape[1] // 4, dim=1)
    C = C // 4

    psize_list = [(3, 3)] * 4 if small_patch else [(1, 9)] * 4
    dilate_list = [(1, 1)] * 4

    search_num = 9
    extra_offset = extra_offset.reshape(N, search_num, 2, H, W).permute(0, 1, 3, 4, 2)

    corrs = []
    for i in range(len(psize_list)):
        left_feature, right_feature = lefts[i], rights[i]
        psize, dilate = psize_list[i], dilate_list[i]
        psizey, psizex = psize
        dilatey, dilatex = dilate
        ry = psizey // 2 * dilatey
        rx = psizex // 2 * dilatex

        x_grid, y_grid = torch.meshgrid(
            torch.arange(-rx, rx + 1, dilatex, device=self.fmap1.device),
            torch.arange(-ry, ry + 1, dilatey, device=self.fmap1.device),
            indexing="xy",
        )
        offsets = torch.stack((x_grid, y_grid))
        offsets = offsets.reshape(2, -1).permute(1, 0)
        for d in sorted((0, 2, 3)):
            offsets = offsets.unsqueeze(d)
        offsets = offsets + extra_offset

        coords = self.coords + flow
        coords = coords.permute(0, 2, 3, 1)
        coords = torch.unsqueeze(coords, 1) + offsets
        coords = coords.reshape(N, -1, W, 2)

        right_feature = _bilinear_sampler(right_feature, coords)
        # Reshape to [N*C, search_num, H, W] to avoid the 5D tensor.
        right_feature = right_feature.reshape(N * C, -1, H, W)
        # Transpose left_feature to [C*N, H, W] then unsqueeze to align.
        left_feature = left_feature.transpose(1, 0)
        corr = torch.mean(left_feature * right_feature, dim=0, keepdim=True)
        corrs.append(corr)

    return torch.cat(corrs, dim=1)


def patch_crestereo(net: torch.nn.Module) -> None:
    """Apply all export-friendly patches to the CREStereo nn.Module.

    Patches applied:

    1. **convex_upsample** - Replaces the 6D-reshape approach with a
       conv2d-based gather. Registers a 3x3 identity-gather kernel as a
       buffer and binds the new method to the instance.
    2. **LinearAttention.forward** - Replaces the einsum-based implementation
       with explicit matmul ops for every ``LinearAttention`` sub-module.
    3. **bilinear_sampler** - Replaces the custom ``bilinear_grid_sample``
       with ``F.grid_sample`` which is directly supported by QNN.
    4. **AGCL.corr_att_offset** - Replaces ``repeat_interleave`` and 5D
       tensor ops with export-friendly equivalents.

    Must be called after ``load_state_dict`` so buffer registration does not
    interfere with weight loading.
    """
    # Patch 1: convex_upsample
    _kernel = torch.eye(9).view(9, 1, 3, 3)
    net.register_buffer("_unfold_kernel", _kernel, persistent=False)
    object.__setattr__(net, "convex_upsample", types.MethodType(_convex_upsample, net))

    # Patch 2: LinearAttention.forward
    for module in net.modules():
        if type(module).__name__ == "LinearAttention":
            object.__setattr__(
                module, "forward", types.MethodType(_linear_attention_forward, module)
            )

    # Patch 3: bilinear_sampler
    corr_mod = sys.modules.get("nets.corr")
    utils_mod = sys.modules.get("nets.utils.utils")
    if utils_mod is not None:
        cast(Any, utils_mod).bilinear_sampler = _bilinear_sampler
    else:
        raise RuntimeError("nets.corr or nets.utils.utils not found in sys.modules")

    # Patch 4: AGCL.corr_att_offset
    agcl_cls = getattr(corr_mod, "AGCL", None) if corr_mod is not None else None
    if agcl_cls is not None:
        agcl_cls.corr_att_offset = _corr_att_offset
    else:
        raise RuntimeError("AGCL class not found in nets.corr")

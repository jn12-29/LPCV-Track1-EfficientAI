# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from stereonet.model import CostVolume
from torch import nn


class Conv3dOptimized(nn.Module):
    """Conv3d replacement implemented as (unfold -> 1x1 Conv2d)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        expected_depth: int = 35,
        use_norm: bool = True,
        use_activation: bool = True,
        activation_negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.out_channels = out_channels

        # Normalization and activation layers.
        if use_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        if use_activation:
            self.leaky_relu = nn.LeakyReLU(negative_slope=activation_negative_slope)
        self.use_norm = use_norm
        self.use_activation = use_activation

        padded_depth = expected_depth + 2 * padding
        out_depth = padded_depth - kernel_size + 1
        total_out_channels = out_depth * (kernel_size**3)
        self.unfold_conv = nn.Conv2d(
            in_channels=padded_depth,
            out_channels=total_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=False,
        )
        with torch.no_grad():
            self.unfold_conv.weight.copy_(
                self._create_unfold_weights(padded_depth, kernel_size, torch.float32)
            )
        self.unfold_conv.weight.requires_grad = False

        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels * (kernel_size**3),
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )

    @staticmethod
    def _create_unfold_weights(
        in_depth: int, kernel_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Create weights that emulate a depth-aware unfold via Conv2d."""
        k = kernel_size
        out_depth = in_depth - k + 1

        k_d, k_h, k_w = torch.meshgrid(
            torch.arange(k),
            torch.arange(k),
            torch.arange(k),
            indexing="ij",
        )  # each [K, K, K]
        k_d = k_d.reshape(-1)
        k_h = k_h.reshape(-1)
        k_w = k_w.reshape(-1)
        k_idx = torch.arange(k**3)
        d_out_idx = torch.arange(out_depth)

        out_ch = (k_idx[:, None] * out_depth + d_out_idx[None, :]).reshape(-1)
        in_ch = (d_out_idx[None, :] + k_d[:, None]).reshape(-1)
        kh_idx = k_h.repeat_interleave(out_depth)
        kw_idx = k_w.repeat_interleave(out_depth)

        weights = torch.zeros(k**3 * out_depth, in_depth, k, k, dtype=dtype)
        weights[out_ch, in_ch, kh_idx, kw_idx] = 1.0
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution.

        Parameters
        ----------
        x
            Input tensor in BCHWD order.

        Returns
        -------
        torch.Tensor
            Output tensor in BCHWD order.
        """
        batch_size, in_channels, height, width, _depth = x.shape

        x = x.view(batch_size * in_channels, height, width, _depth)
        if self.padding > 0:
            x = F.pad(
                x,
                (self.padding,) * 6,
                mode="constant",
                value=0,
            )

        x = x.permute(0, 3, 1, 2)
        padded_depth = x.shape[1]

        output_conv = self.unfold_conv(x)
        _, _, h_out, w_out = output_conv.shape
        d_out = padded_depth - self.kernel_size + 1
        k3 = self.kernel_size**3

        output_conv = (
            output_conv.permute(0, 2, 3, 1)
            .contiguous()
            .view(batch_size, in_channels, h_out, w_out, d_out * k3)
        )

        output_conv = torch.cat(torch.split(output_conv, d_out, dim=-1), dim=1)
        output_conv = output_conv.contiguous().view(
            batch_size, k3 * in_channels, h_out, w_out * d_out
        )

        out_conv = self.pointwise_conv(output_conv)
        if self.use_norm:
            out_conv = self.batch_norm(out_conv)
        if self.use_activation:
            out_conv = self.leaky_relu(out_conv)

        return out_conv.contiguous().view(
            batch_size, self.out_channels, h_out, w_out, d_out
        )


class CostVolumeOptimized(nn.Module):
    """Optimized compute_volume and fix the memory allocation issue in QNN"""

    def __init__(self, source: CostVolume) -> None:
        super().__init__()
        self._max_downsampled_disps = source._max_downsampled_disps
        in_channels: int = source.in_channels
        out_channels: int = source.out_channels

        net: OrderedDict[str, nn.Module] = OrderedDict()

        for block_idx in range(4):
            expected_depth = (
                self._max_downsampled_disps if block_idx == 0 else out_channels
            )
            net[f"segment_0_conv_{block_idx}"] = Conv3dOptimized(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                expected_depth=expected_depth,
                use_norm=True,
                use_activation=True,
                activation_negative_slope=0.2,
            )
            in_channels = out_channels

        net["segment_1_conv_0"] = Conv3dOptimized(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=3,
            padding=1,
            expected_depth=out_channels,
            use_norm=False,
            use_activation=False,
        )

        self.net = nn.Sequential(net)

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor], side: str = "left"
    ) -> torch.Tensor:
        """Compute the filtered cost volume.

        Parameters
        ----------
        x
            Tuple of (reference_embedding, target_embedding) in BCHW.
        side
            If "left", sweep the target image to the left (standard). If "right",
            sweep to the right.

        Returns
        -------
        torch.Tensor
            Filtered cost volume of shape ``[B, D, H, W]``.
        """
        reference_embedding, target_embedding = x

        cost = compute_volume(
            reference_embedding,
            target_embedding,
            max_downsampled_disps=self._max_downsampled_disps,
            side=side,
        )

        return self.net(cost).squeeze(1).permute(0, 3, 1, 2)


def compute_volume(
    reference_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    max_downsampled_disps: int,
    side: str = "left",
) -> torch.Tensor:
    """Compute a difference-based cost volume using a single gather operation.

    Replaces the original implementation's multiple scatter operations with a
    single ``torch.gather`` call, which shifts target embeddings by each
    disparity level in one step and is more efficient on device.

    Parameters
    ----------
    reference_embedding
        Feature map from the reference (left or right) image in BCHW order.
    target_embedding
        Feature map from the target image in BCHW order.
    max_downsampled_disps
        Number of disparity levels ``D`` at the current (downsampled) scale.
    side
        Sweep direction. ``"left"`` shifts the target leftward (standard
        left-to-right stereo); ``"right"`` shifts it rightward.

    Returns
    -------
    torch.Tensor
        Difference-based cost volume of shape ``[B, C, H, W, D]`` (BCHWD).
        Out-of-bounds positions are zeroed out via a validity mask.
    """
    batch, channel, height, width = reference_embedding.size()
    device = reference_embedding.device

    ref_expanded = (
        reference_embedding.reshape(batch * channel, height, width)
        .unsqueeze(-1)
        .expand(-1, -1, -1, max_downsampled_disps)
    )

    disp_indices = torch.arange(max_downsampled_disps, device=device)
    width_grid = (
        torch.arange(width, device=device)
        .unsqueeze(1)
        .expand(-1, max_downsampled_disps)
    )

    if side == "left":
        shifted_idx = torch.clamp(width_grid - disp_indices, min=0)
        valid_mask = width_grid >= disp_indices
    else:  # side == 'right'
        shifted_idx = torch.clamp(width_grid + disp_indices, max=width - 1)
        valid_mask = (width_grid + disp_indices) < width

    tgt_bc_hw = target_embedding.reshape(batch * channel, height, width)
    shifted_idx_expanded = shifted_idx.view(1, 1, width, max_downsampled_disps).expand(
        batch * channel, height, width, max_downsampled_disps
    )
    tgt_expanded = tgt_bc_hw.unsqueeze(-1).expand(
        batch * channel, height, width, max_downsampled_disps
    )
    tgt_gathered = torch.gather(tgt_expanded, dim=2, index=shifted_idx_expanded)

    cost = (ref_expanded - tgt_gathered) * valid_mask.view(
        1, 1, width, max_downsampled_disps
    )
    return cost.reshape(batch, channel, height, width, max_downsampled_disps)

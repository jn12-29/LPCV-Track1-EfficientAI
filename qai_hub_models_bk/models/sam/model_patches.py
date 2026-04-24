# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from segment_anything.modeling.image_encoder import Attention as SAMEncoderAttention
from segment_anything.modeling.image_encoder import get_rel_pos
from torch import nn

from qai_hub_models.models._shared.sam.model_patches import Conv2DInplaceLinear


class SplitHeadSAMEncoderAttention(nn.Module):
    """
    SAM Attention block with the following modifications necessary to run on QNN:
        * Heads are split into separate ops, rather than all heads running in a single op.
        * QKV is unpacked from 1 tensor into 3 tensors.
    """

    def __init__(self, attention_block: SAMEncoderAttention) -> None:
        super().__init__()
        self.out_feature, self.in_feature = (
            attention_block.qkv.weight.shape[0] // 3 // attention_block.num_heads,
            attention_block.qkv.weight.shape[1],
        )
        chunk_size = attention_block.qkv.weight.shape[0] // 3

        bias = attention_block.qkv.bias[: self.out_feature] is not None
        self.q = torch.nn.ModuleList()
        self.k = torch.nn.ModuleList()
        self.v = torch.nn.ModuleList()
        self.proj = Conv2DInplaceLinear.from_linear(attention_block.proj)
        self.use_rel_pos = attention_block.use_rel_pos
        self.scale = attention_block.scale
        self.num_heads = attention_block.num_heads
        self.rel_pos_h = attention_block.rel_pos_h
        self.rel_pos_w = attention_block.rel_pos_w

        for i in range(attention_block.num_heads):
            for chunk, projList in enumerate([self.q, self.k, self.v]):
                split_layer = Conv2DInplaceLinear(
                    self.in_feature, self.out_feature, has_bias=bias
                )
                split_layer.conv2d.weight.data.copy_(
                    attention_block.qkv.weight[
                        i * self.out_feature + (chunk * chunk_size) : (i + 1)
                        * self.out_feature
                        + (chunk * chunk_size),
                        :,
                        None,
                        None,
                    ]
                )

                assert split_layer.conv2d.bias is not None
                split_layer.conv2d.bias.data.copy_(
                    attention_block.qkv.bias[
                        i * self.out_feature + (chunk * chunk_size) : (i + 1)
                        * self.out_feature
                        + (chunk * chunk_size)
                    ]
                )

                projList.append(split_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        """
        # Original code (replaced by split-head implementation below):
        # qkv with shape (3, B, nHead, H * W, C)
        # qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        # q0, k0, v0 = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        """
        x_list: list[torch.Tensor] = []
        for i in range(self.num_heads):
            q_i = self.q[i](x).reshape(B, H * W, 1, -1).permute(0, 2, 1, 3)
            k_i = self.k[i](x).reshape(B, H * W, 1, -1).permute(0, 2, 1, 3)
            v_i = self.v[i](x).reshape(B, H * W, 1, -1).permute(0, 2, 1, 3)
            attn_i = (q_i * self.scale) @ k_i.transpose(-2, -1)

            if self.use_rel_pos:
                attn_i = SplitHeadSAMEncoderAttention.add_decomposed_rel_pos_unpack(
                    attn_i,
                    q_i,
                    self.rel_pos_h,
                    self.rel_pos_w,
                    (H, W),
                    (H, W),
                )

            attn_i = attn_i.softmax(dim=-1)
            x_i = (attn_i @ v_i).reshape(B, 1, H * W, -1)
            x_list.append(x_i)
        x = (
            torch.concat(x_list, dim=1)
            .reshape(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        return self.proj(x)

    @staticmethod
    def einsum_to_matmul_bhwc_hkc_bhwk(
        r_q: torch.Tensor, Rh: torch.Tensor
    ) -> torch.Tensor:
        Rh = torch.transpose(Rh, 2, 1)
        return torch.matmul(r_q, Rh)

    @staticmethod
    def einsum_to_matmul_bhwc_wkc_bhwk(
        r_q: torch.Tensor, Rw: torch.Tensor
    ) -> torch.Tensor:
        r_q = torch.transpose(r_q, 2, 1)
        Rw = torch.transpose(Rw, 2, 1)
        test_result_second = torch.matmul(r_q, Rw)
        return torch.transpose(test_result_second, 2, 1)

    @staticmethod
    def add_decomposed_rel_pos_unpack(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: tuple[int, int],
        k_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.

        Lifted from segment_anything.modeling.image_encoder.add_decomposed_rel_pos
        Modifications by Qualcomm:
         * Enable compatibility of Q shape with other changes that unpack attention QKV
         * Replace Einsum with equivalent ops (einsum is not supported by QNN)

        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Parameters
        ----------
        attn
            Attention map.
        q
            Query q in the attention layer with shape (B, q_h, q_w, C).
        rel_pos_h
            Relative position embeddings (Lh, C) for height axis.
        rel_pos_w
            Relative position embeddings (Lw, C) for width axis.
        q_size
            Spatial sequence size of query q with (q_h, q_w).
        k_size
            Spatial sequence size of key k with (k_h, k_w).

        Returns
        -------
        attn : torch.Tensor
            Attention map with added relative positional embeddings.
        """
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)

        # -- Begin Qualcomm Change
        B, _, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = SplitHeadSAMEncoderAttention.einsum_to_matmul_bhwc_hkc_bhwk(r_q, Rh)
        rel_w = SplitHeadSAMEncoderAttention.einsum_to_matmul_bhwc_wkc_bhwk(r_q, Rw)
        # -- End Qualcomm Change

        return (
            attn.view(B, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None]
            + rel_w[:, :, :, None, :]
        ).view(B, 1, q_h * q_w, k_h * k_w)

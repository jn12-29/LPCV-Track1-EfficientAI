# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

import torch
from typing_extensions import Self

from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, IoType, TensorSpec

SOURCE_REPO = "https://github.com/nikitakaraevv/pointnet"
COMMIT_HASH = "256437e9ab27b197347464cecff87121c5c824ff"
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "pretrained/save.pth",
)
PATCHES = [
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "patches", "pointnet_changes.diff")
    )
]


class Pointnet(BaseModel):
    @classmethod
    def from_pretrained(cls, weights_name: str = DEFAULT_WEIGHTS) -> Self:
        weights = load_torch(weights_name)
        with SourceAsRoot(
            SOURCE_REPO,
            COMMIT_HASH,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            source_repo_patches=PATCHES,
        ):
            from source.model import PointNet

            model = PointNet()
            model.load_state_dict(weights)
            return cls(model)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the PointNet model for point cloud classification.

        Parameters
        ----------
        image
            A tensor of shape (B, 3, N), where:
                - B is the batch size,
                - 3 represents the x, y, z coordinates of each point,
                - N is the number of points in the cloud (e.g., 1024).
            Channel Layout: XYZ coordinates.
            Range: floating point values, typically normalized between 0 and 1.

        Returns
        -------
        x : torch.Tensor
            Tensor of classification logits with shape (B, num_classes).
        crit_idxs : torch.Tensor
            Critical point indices used internally by PointNet.
        A_feat : torch.Tensor
            Feature matrix used internally by PointNet.
        """
        return self.model(image)

    @staticmethod
    def get_input_spec(
        num_points: int = 1024,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(1, 3, num_points),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["x", "crit_idxs", "A_feat"]

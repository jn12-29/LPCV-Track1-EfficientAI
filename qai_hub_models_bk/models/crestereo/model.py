# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models import TargetRuntime
from qai_hub_models.models.common import Precision
from qai_hub_models.models.crestereo.model_patches import patch_crestereo
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

CRESTEREO_SOURCE_REPOSITORY = "https://github.com/ibaiGorordo/CREStereo-Pytorch.git"
CRESTEREO_SOURCE_REPO_COMMIT = "b6c7a9fe8dc2e9e56ba7b96f4677312309282d15"

# downloaded from https://drive.google.com/file/d/1D2s1v4VhJlNz98FQpFxf_kBAKQVN_7xo/view
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "crestereo_eth3d.pth"
)


class CREStereo(BaseModel):
    """CREStereo stereo depth estimation model."""

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> Self:
        with SourceAsRoot(
            CRESTEREO_SOURCE_REPOSITORY,
            CRESTEREO_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ):
            from nets import Model

            ckpt_path = weights_path or str(DEFAULT_WEIGHTS.fetch())
            net = Model(max_disp=256, mixed_precision=False, test_mode=True)
            state_dict = torch.load(
                ckpt_path, map_location=torch.device("cpu"), weights_only=False
            )
            net.load_state_dict(state_dict, strict=True)

        patch_crestereo(net)
        return cls(net)

    def forward(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
        left_image_dw2: torch.Tensor,
        right_image_dw2: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full two-pass CREStereo pipeline.

        Parameters
        ----------
        left_image
            Float32 BGR [0, 1] tensor of shape ``[B, 3, H, W]``.
        right_image
            Float32 BGR [0, 1] tensor of shape ``[B, 3, H, W]``.
        left_image_dw2
            Half-resolution left image of shape ``[B, 3, H/2, W/2]``, BGR [0, 1].
        right_image_dw2
            Half-resolution right image of shape ``[B, 3, H/2, W/2]``, BGR [0, 1].

        Returns
        -------
        torch.Tensor
            Disparity map of shape ``[B, 1, H, W]``.
        """
        left_image = left_image * 255.0
        right_image = right_image * 255.0
        left_image_dw2 = left_image_dw2 * 255.0
        right_image_dw2 = right_image_dw2 * 255.0

        # Pass 1 - coarse inference at half resolution
        flow_init = self.model(
            left_image_dw2, right_image_dw2, iters=20, flow_init=None
        )
        # Pass 2 - full-resolution refinement
        flow = self.model(left_image, right_image, iters=20, flow_init=flow_init)
        return flow[:, 0:1, :, :]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
        if target_runtime != TargetRuntime.ONNX:
            return " --truncate_64bit_tensors " + compile_options
        return compile_options

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> str:
        # NPU has accuracy issues; force CPU execution.
        return " --compute_unit cpu " + super().get_hub_profile_options(
            target_runtime, other_profile_options
        )

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 240,
        width: int = 320,
    ) -> InputSpec:
        shape = (batch_size, 3, height, width)
        shape_dw2 = (batch_size, 3, height // 2, width // 2)
        bgr_metadata = ImageMetadata(
            color_format=ColorFormat.BGR,
            value_range=(0.0, 1.0),
        )
        return {
            "left_image": TensorSpec(
                shape=shape,
                dtype="float32",
                io_type=IoType.IMAGE,
                image_metadata=bgr_metadata,
            ),
            "right_image": TensorSpec(
                shape=shape,
                dtype="float32",
                io_type=IoType.IMAGE,
                image_metadata=bgr_metadata,
            ),
            "left_image_dw2": TensorSpec(
                shape=shape_dw2,
                dtype="float32",
                io_type=IoType.IMAGE,
                image_metadata=bgr_metadata,
            ),
            "right_image_dw2": TensorSpec(
                shape=shape_dw2,
                dtype="float32",
                io_type=IoType.IMAGE,
                image_metadata=bgr_metadata,
            ),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["disparity"]

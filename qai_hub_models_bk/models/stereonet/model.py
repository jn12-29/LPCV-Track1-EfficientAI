# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn.functional as F
from qai_hub.client import Device
from stereonet.model import CostVolume, soft_argmin
from stereonet.model import StereoNet as StereoNetModel
from torch import nn
from typing_extensions import Self

from qai_hub_models.models.stereonet.model_patch import CostVolumeOptimized
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec, IoType, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

DEFAULT_CKPT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "epoch=21-step=696366.ckpt"
)


class StereoNetWrapper(nn.Module):
    """
    Wrapper to avoid any PyTorch Lightning dependencies,
    which causes error in export.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        base_model = StereoNetModel(in_channels=in_channels)

        self.feature_extractor = base_model.feature_extractor
        self.cost_volumizer = base_model.cost_volumizer
        self.refiners = base_model.refiners

        self.in_channels = base_model.in_channels
        self.k_refinement_layers = base_model.k_refinement_layers
        self.candidate_disparities = base_model.candidate_disparities

    def forward(self, batch: torch.Tensor, side: str = "left") -> torch.Tensor:
        """Run inference.

        Parameters
        ----------
        batch
            Input tensor of shape `[B, 2, H, W]` containing left/right grayscale
            images stacked along the channel dimension.
        side
            Which image is the reference view. ``"left"`` (default) or ``"right"``.

        Returns
        -------
        torch.Tensor
            Disparity prediction of shape `[B, 1, H, W]`.
        """
        if side == "left":
            reference = batch[:, : self.in_channels]
            shifting = batch[:, self.in_channels : self.in_channels * 2]
        else:
            reference = batch[:, self.in_channels : self.in_channels * 2]
            shifting = batch[:, : self.in_channels]

        reference_embedding = self.feature_extractor(reference)
        shifting_embedding = self.feature_extractor(shifting)

        cost = self.cost_volumizer((reference_embedding, shifting_embedding), side=side)

        disparity = soft_argmin(cost, self.candidate_disparities)

        for idx, refiner in enumerate(self.refiners, start=1):
            scale = (2**self.k_refinement_layers) / (2**idx)
            new_h = int(reference.shape[2] // scale)
            new_w = int(reference.shape[3] // scale)
            size = [new_h, new_w]
            disparity_rescaled = F.interpolate(
                disparity, size, mode="bilinear", align_corners=True
            )
            reference_rescaled = F.interpolate(
                reference, size, mode="bilinear", align_corners=True
            )
            disparity = F.relu(
                refiner(torch.cat((reference_rescaled, disparity_rescaled), dim=1))
                + disparity_rescaled
            )

        return disparity


class StereoNet(BaseModel):
    """StereoNet disparity estimation model."""

    @classmethod
    def from_pretrained(cls, ckpt: str | None = None) -> Self:
        checkpoint_path = ckpt or str(DEFAULT_CKPT.fetch())
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=False
        )

        model = StereoNetWrapper(in_channels=1)

        # Replaced conv3d  to fix the below error
        # QNN_COMMON_ERROR_MEM_ALLOC: Memory allocation related error
        for name, module in model.named_children():
            if isinstance(module, CostVolume):
                setattr(model, name, CostVolumeOptimized(module))

        state_dict = checkpoint["state_dict"]
        remapped_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            key = key.removeprefix("model.")

            prefix = "cost_volumizer.net."
            if key.startswith(prefix):
                if "conv" in key and key.endswith(".weight"):
                    out_c = value.shape[0]
                    value = (
                        value.permute(0, 2, 3, 4, 1).contiguous().view(out_c, -1, 1, 1)
                    )
                    key = key[: -len(".weight")] + ".pointwise_conv.weight"
                elif "conv" in key and key.endswith(".bias"):
                    key = key[: -len(".bias")] + ".pointwise_conv.bias"
                elif "bn" in key:
                    segment, suffix = key[len(prefix) :].split(".")
                    segment, i = segment.split("_bn_")
                    key = f"{prefix + segment}_conv_{i}.batch_norm.{suffix}"

            remapped_state_dict[key] = value

        # strict=False because conv3d weights were remapped to CostVolumeOptimized
        # pointwise_conv format above; extra/missing keys are expected.
        model.load_state_dict(remapped_state_dict, strict=False)

        return cls(model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Run StereoNet on a stereo pair and return a disparity map.

        Parameters
        ----------
        image
            Input tensor of shape `[B, 2, H, W]`.
            - Channel 0: left grayscale image
            - Channel 1: right grayscale image
            Expected range: `float32` in `[0, 1]`.

        Returns
        -------
        torch.Tensor
            Disparity tensor of shape `[B, 1, H, W]`.
        """
        mean = (torch.tensor([111.5684, 113.6528]) / 255).view(1, 2, 1, 1)
        std = (torch.tensor([61.9625, 62.0313]) / 255).view(1, 2, 1, 1)
        return self.model((image - mean) / std)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 786,
        width: int = 490,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 2, height, width),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["disparity"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--truncate_64bit_tensors" not in other_compile_options
        ):
            other_compile_options += " --truncate_64bit_tensors"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["disparity"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

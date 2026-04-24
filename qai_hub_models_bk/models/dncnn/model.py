# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.denoising_evaluator import DenoisingEvaluator
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
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
MODEL_ASSET_VERSION = 1
SOURCE_REPO = "https://github.com/cszn/KAIR"
COMMIT_HASH = "fc1732f4a4514e42ce15e5b3a1e18c828af47a1e"
WEIGHTS_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_25.pth"
DEFAULT_INPUT_HEIGHT = 256
DEFAULT_INPUT_WIDTH = 256
NUM_CHANNELS = 64
NUM_LAYERS = 17


class DnCNN(BaseModel):
    """
    DnCNN: Denoising Convolutional Neural Network.

    A 17-layer CNN that uses residual learning for Gaussian noise removal.
    The network predicts the noise residual, and the denoised output is
    computed as: denoised = input - predicted_noise.

    Source: https://github.com/cszn/KAIR
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls) -> Self:
        """Load DnCNN with pretrained weights for sigma=25 Gaussian denoising."""
        with SourceAsRoot(SOURCE_REPO, COMMIT_HASH, MODEL_ID, MODEL_ASSET_VERSION):
            from models.network_dncnn import DnCNN as KairDnCNN

            kair_model = KairDnCNN(
                in_nc=1, out_nc=1, nc=NUM_CHANNELS, nb=NUM_LAYERS, act_mode="R"
            )

        checkpoint_asset = CachedWebModelAsset(
            WEIGHTS_URL, MODEL_ID, MODEL_ASSET_VERSION, "dncnn_25.pth"
        )
        state_dict = load_torch(checkpoint_asset)
        kair_model.load_state_dict(state_dict)
        return cls(kair_model).eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Denoise an input image.

        Parameters
        ----------
        image
            Noisy grayscale image of shape [N, 1, H, W].
            Pixel values in [0, 1].

        Returns
        -------
        torch.Tensor
            Denoised image of shape [N, 1, H, W].
            Pixel values in [0, 1].
        """
        denoised = self.model(image)
        return torch.clamp(denoised, 0.0, 1.0)

    @staticmethod
    def get_input_spec(
        height: int = DEFAULT_INPUT_HEIGHT,
        width: int = DEFAULT_INPUT_WIDTH,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(1, 1, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.GRAYSCALE,
                    value_range=(0.0, 1.0),
                ),
            ),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["denoised_image"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["denoised_image"]

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["bsd300_denoising"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "bsd300_denoising"

    def get_evaluator(self) -> BaseEvaluator:
        return DenoisingEvaluator()

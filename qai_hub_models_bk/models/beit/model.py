# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from qai_hub.client import Device
from transformers import BeitForImageClassification
from typing_extensions import Self

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.base_model import Precision, TargetRuntime

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "microsoft/beit-base-patch16-224"
MODEL_ASSET_VERSION = 1


class Beit(ImagenetClassifier):
    """Exportable Beit model, end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS) -> Self:
        return cls(BeitForImageClassification.from_pretrained(ckpt_name))

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

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(image_tensor, return_dict=False)[0]

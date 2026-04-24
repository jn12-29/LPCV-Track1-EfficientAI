# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

from typing_extensions import Self
from ultralytics import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models.models._shared.ultralytics.segmentation_model import (
    UltralyticsMulticlassSegmentor,
)
from qai_hub_models.models._shared.yolo.model import YoloSegEvalMixin

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]


SUPPORTED_WEIGHTS = [
    "yolo26n-seg.pt",
    "yolo26s-seg.pt",
    "yolo26m-seg.pt",
    "yolo26l-seg.pt",
    "yolo26x-seg.pt",
]
DEFAULT_WEIGHTS = "yolo26n-seg.pt"


class Yolo26Segmentor(UltralyticsMulticlassSegmentor, YoloSegEvalMixin):
    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS) -> Self:
        if ckpt_name not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported checkpoint name provided {ckpt_name}.\n"
                f"Supported checkpoints are {list(SUPPORTED_WEIGHTS)}."
            )
        model = cast(SegmentationModel, ultralytics_YOLO(ckpt_name).model)
        return cls(model)

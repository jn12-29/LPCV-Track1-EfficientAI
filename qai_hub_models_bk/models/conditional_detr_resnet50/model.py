# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub.client import Device
from transformers import ConditionalDetrForObjectDetection

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models._shared.detr.model import DETR
from qai_hub_models.utils.base_model import Precision, TargetRuntime

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class ConditionalDETRResNet50(DETR):
    """Exportable DETR model, end-to-end."""

    DEFAULT_WEIGHTS = "microsoft/conditional-detr-resnet-50"
    HF_DETR_CLS = ConditionalDetrForObjectDetection

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

    def get_evaluator(self) -> BaseEvaluator:
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(image_height, image_width, score_threshold=0.4)

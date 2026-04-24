# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models._shared.yolo.app import YoloObjectDetectionApp


class Yolo26DetectionApp(YoloObjectDetectionApp):
    def check_image_size(self, pixel_values: torch.Tensor) -> None:
        """YOLO26 does not enforce strict spatial dimension constraints for input images."""

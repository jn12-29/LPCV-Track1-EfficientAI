# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.metrics import PSNR, MetricMetadata
from qai_hub_models.utils.compare import compute_psnr


class DenoisingEvaluator(BaseEvaluator):
    """Evaluator for image denoising models using PSNR."""

    def __init__(self) -> None:
        self.reset()

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor) -> None:
        """
        Add a batch of denoised outputs and clean ground truth images.

        Parameters
        ----------
        output
            Denoised images of shape [N, C, H, W] in [0, 1].
        gt
            Clean ground truth images of shape [N, C, H, W] in [0, 1].
        """
        assert gt.shape == output.shape

        output = output.detach()
        gt = gt.detach()

        batch_size = gt.shape[0]
        for i in range(batch_size):
            pred = output[i].numpy()
            truth = gt[i].numpy()
            # PSNR on [0, 1] data range
            psnr = compute_psnr(pred, truth, data_range=1.0)
            self.psnr_list.append(psnr)

    def reset(self) -> None:
        self.psnr_list: list[float] = []

    def get_accuracy_score(self) -> float:
        if not self.psnr_list:
            return 0.0
        return float(np.mean(np.array(self.psnr_list)))

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.2f} dB PSNR"

    def get_metric_metadata(self) -> MetricMetadata:
        return PSNR.with_description(
            "A measure of how similar the denoised image is to the clean original."
        )

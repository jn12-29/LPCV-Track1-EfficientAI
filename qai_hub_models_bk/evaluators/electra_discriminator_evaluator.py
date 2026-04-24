# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.metrics import ACCURACY_TOP1, MetricMetadata


class ElectraDiscriminatorEvaluator(BaseEvaluator):
    """Evaluator for the Electra discriminator on the WikiTextMasked dataset."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset accumulated counts."""
        self._num_correct: int = 0
        self._num_total: int = 0

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor) -> None:
        """
        Parameters
        ----------
        output
            Binary predictions for all token positions, shape [seq_len] or
            [batch_size, seq_len]. Values are 0 (real) or 1 (fake).
        gt
            Mask position index per sample, shape [batch_size] or [batch_size, 1].
        """
        output = output.detach().cpu()
        gt = gt.detach().cpu().view(-1).to(torch.int64)

        num_samples = gt.shape[0]

        if output.dim() == 1:
            seq_len = output.shape[0] // num_samples
            output = output.view(num_samples, seq_len)

        for i in range(num_samples):
            mask_pos = int(gt[i].item())
            assert 0 <= mask_pos < output.shape[1], (
                f"mask_pos {mask_pos} out of bounds for seq_len {output.shape[1]}"
            )
            pred_at_mask = output[i, mask_pos].item()
            # The [MASK] token is always fake (1); correct if model predicts 1
            if pred_at_mask == 1:
                self._num_correct += 1
            self._num_total += 1

    def get_accuracy_score(self) -> float:
        if self._num_total == 0:
            return 0.0
        return 100.0 * float(self._num_correct) / float(self._num_total)

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.2f}%"

    def get_metric_metadata(self) -> MetricMetadata:
        return ACCURACY_TOP1.with_description(
            "Percentage of [MASK] positions correctly predicted as fake/replaced "
            "by the Electra discriminator."
        )

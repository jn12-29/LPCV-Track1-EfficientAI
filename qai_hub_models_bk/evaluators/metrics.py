# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Centralized metric definitions used by evaluators and info.yaml validation.

Each metric is registered via ``_register_metric``, which appends to
a module-level list and collects ``(name, unit)`` into VALID_METRIC_PAIRS.
After all metrics are defined, VALID_METRIC_PAIRS is frozen to prevent
accidental mutation.
"""

from __future__ import annotations

from typing import NamedTuple


class MetricMetadata(NamedTuple):
    name: str
    unit: str
    description: str
    range: tuple[float | None, float | None]
    float_vs_device_threshold: float | None = None

    def with_description(self, description: str) -> MetricMetadata:
        """Return a copy with a different description."""
        return self._replace(description=description)


# Set of valid (metric_name, unit) pairs, built during module init
# and frozen at the bottom of this file.
_VALID_METRIC_PAIRS: set[tuple[str, str]] = set()


def _register_metric(metric: MetricMetadata) -> MetricMetadata:
    """Register a MetricMetadata and return it."""
    key = (metric.name, metric.unit)
    assert key not in _VALID_METRIC_PAIRS, (
        f"Duplicate metric key {key!r} — use a distinct unit to differentiate"
    )
    _VALID_METRIC_PAIRS.add(key)
    return metric


ACCURACY = _register_metric(
    MetricMetadata(
        name="Accuracy",
        unit="%",
        description="Percentage of model predictions that are correct.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

ACCURACY_TOP1 = _register_metric(
    MetricMetadata(
        name="Top-1 Accuracy",
        unit="%",
        description="Percentage of top 1 model predictions that are correct.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

AVERAGE_PRECISION = _register_metric(
    MetricMetadata(
        name="Average Precision",
        unit="AP",
        description="Area under the precision-recall curve for a single class.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

COLORFULNESS = _register_metric(
    MetricMetadata(
        name="Colorfulness",
        unit="",
        description="A measure of how varied the colors are in the image.",
        range=(0.0, None),
        float_vs_device_threshold=10.0,
    )
)

DELTA_THRESHOLD_ACCURACY = _register_metric(
    MetricMetadata(
        name="Delta Threshold Accuracy",
        unit="δ1",
        description="The percentage of pixels where the predicted depth is within 25% of the expected.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

KL_DIVERGENCE = _register_metric(
    MetricMetadata(
        name="KL divergence",
        unit="kldiv",
        description="A distance metric between two probability distributions. Lower is better.",
        range=(0.0, None),
    )
)

MEAN_ANGULAR_ERROR = _register_metric(
    MetricMetadata(
        name="Mean Angular Error",
        unit="MAE (Degrees)",
        description="Mean angular error between predicted and ground truth gaze directions. Lower is better.",
        range=(0.0, None),
        float_vs_device_threshold=5.0,
    )
)

MEAN_AVERAGE_PRECISION = _register_metric(
    MetricMetadata(
        name="Mean Average Precision",
        unit="mAP",
        description="Mean Average Precision (across predicted classes).",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

MEAN_AVERAGE_PRECISION_IOU_5_95 = _register_metric(
    MetricMetadata(
        name="Mean Average Precision",
        unit="mAP@0.5:0.95",
        description="Mean Average Precision averaged over IOU thresholds 0.5 to 0.95 in 0.05 increments.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

MEAN_IOU = _register_metric(
    MetricMetadata(
        name="Mean Intersection Over Union",
        unit="mIOU",
        description="Overlap of predicted and expected segmentation divided by the union size.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

MMLU = _register_metric(
    MetricMetadata(
        name="Massive Multitask Language Understanding",
        unit="MMLU",
        description="Set of multiple choice questions that the model answered correctly.",
        range=(0.0, 1.0),
        float_vs_device_threshold=0.1,
    )
)

NORMALIZED_MEAN_ERROR = _register_metric(
    MetricMetadata(
        name="Normalized Mean Error",
        unit="NME",
        description="Average error between predictions and ground truth, normalized by a reference scale. Lower is better.",
        range=(0.0, None),
        float_vs_device_threshold=0.01,
    )
)

PANOPTIC_QUALITY = _register_metric(
    MetricMetadata(
        name="Panoptic Quality",
        unit="PQ",
        description="A measure of how well all objects in the image were correctly identified and segmented.",
        range=(0.0, 1.0),
        float_vs_device_threshold=0.1,
    )
)

PERCENTAGE_CORRECT_KEYPOINTS = _register_metric(
    MetricMetadata(
        name="Percentage Correct Keypoints (head-normalized)",
        unit="PCKh",
        description="Percentage of keypoints within a certain distance of expected.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

PERPLEXITY = _register_metric(
    MetricMetadata(
        name="Perplexity",
        unit="PPL",
        description="A measure of how likely the model is to predict a given sequence of words. Lower is better.",
        range=(0.0, None),
    )
)

PSNR = _register_metric(
    MetricMetadata(
        name="Peak Signal-to-Noise Ratio (PSNR)",
        unit="dB",
        description="A measure of how similar two images are.",
        range=(0.0, None),
        float_vs_device_threshold=10.0,
    )
)

SSIM = _register_metric(
    MetricMetadata(
        name="Structural Similarity",
        unit="SSIM",
        description="A measure of the perceived quality difference between two images.",
        range=(0.0, 1.0),
        float_vs_device_threshold=0.1,
    )
)

WORD_ERROR_RATE = _register_metric(
    MetricMetadata(
        name="Word Error Rate",
        unit="WER",
        description="The percentage of words incorrectly predicted. Lower is better.",
        range=(0.0, 100.0),
        float_vs_device_threshold=10.0,
    )
)

# Freeze the valid pairs set now that all metrics are registered.
VALID_METRIC_PAIRS: frozenset[tuple[str, str]] = frozenset(_VALID_METRIC_PAIRS)

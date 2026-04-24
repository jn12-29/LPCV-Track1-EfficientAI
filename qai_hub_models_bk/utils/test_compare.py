# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.utils.compare import (
    compute_mae,
    compute_max_abs_diff,
    compute_mse,
    compute_top_k_accuracy,
)


def test_compute_mse() -> None:
    # Identical arrays should have MSE of 0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert compute_mse(a, b) == 0.0

    # MSE of [1,2,3] vs [2,3,4] = mean([1,1,1]) = 1.0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(compute_mse(a, b), 1.0)


def test_compute_mae() -> None:
    # Identical arrays should have MAE of 0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert compute_mae(a, b) == 0.0

    # MAE of [1,2,3] vs [2,3,4] = mean([1,1,1]) = 1.0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(compute_mae(a, b), 1.0)


def test_compute_max_abs_diff() -> None:
    # Identical arrays should have max abs diff of 0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert compute_max_abs_diff(a, b) == 0.0

    # Max abs diff of [1,2,3] vs [1,2,6] = 3.0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 6.0])
    np.testing.assert_allclose(compute_max_abs_diff(a, b), 3.0)


def test_compute_top_k_accuracy() -> None:
    expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    actual = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

    k = 3
    result = compute_top_k_accuracy(expected, actual, k)
    np.testing.assert_allclose(1 / 3, result)

    actual = np.array([0.1, 0.2, 0.3, 0.5, 0.4])
    result = compute_top_k_accuracy(expected, actual, k)
    np.testing.assert_allclose(result, 1, atol=1e-3)

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest

from qai_hub_models.utils.testing_export_eval import _pad_and_concatenate


def test_pad_and_concatenate_no_padding() -> None:
    """Arrays with matching shapes (except axis 0) need no padding."""
    a = np.ones((2, 3, 4))
    b = np.full((3, 3, 4), 2.0)
    result = _pad_and_concatenate([a, b])
    assert result.shape == (5, 3, 4)
    np.testing.assert_array_equal(result[:2], a)
    np.testing.assert_array_equal(result[2:], b)


def test_pad_and_concatenate_with_padding() -> None:
    """Arrays with different non-batch dims get zero-padded."""
    a = np.ones((2, 3))
    b = np.full((3, 5), 2.0)
    result = _pad_and_concatenate([a, b])
    assert result.shape == (5, 5)
    # a's data in first 3 cols, zero-padded in cols 3-4
    np.testing.assert_array_equal(result[:2, :3], 1.0)
    np.testing.assert_array_equal(result[:2, 3:], 0.0)
    # b's data fills all 5 cols
    np.testing.assert_array_equal(result[2:, :5], 2.0)


def test_pad_and_concatenate_3d_padding() -> None:
    """Padding works across multiple non-batch dimensions."""
    a = np.ones((1, 2, 3))
    b = np.full((1, 4, 5), 2.0)
    result = _pad_and_concatenate([a, b])
    assert result.shape == (2, 4, 5)
    # a is padded in both dim 1 and dim 2
    np.testing.assert_array_equal(result[0, :2, :3], 1.0)
    np.testing.assert_array_equal(result[0, 2:, :], 0.0)
    np.testing.assert_array_equal(result[0, :2, 3:], 0.0)
    # b fills the full shape
    np.testing.assert_array_equal(result[1], 2.0)


def test_pad_and_concatenate_single_array() -> None:
    """A single array is returned as-is (via concatenate fast path)."""
    a = np.arange(12).reshape(3, 4)
    result = _pad_and_concatenate([a])
    assert result.shape == (3, 4)
    np.testing.assert_array_equal(result, a)


def test_pad_and_concatenate_preserves_dtype() -> None:
    """Output dtype matches input dtype."""
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((1, 5), dtype=np.float32)
    result = _pad_and_concatenate([a, b])
    assert result.dtype == np.float32


def test_pad_and_concatenate_rank_mismatch_raises() -> None:
    """Arrays with different ndim raise ValueError."""
    a = np.ones((2, 3))
    b = np.ones((2, 3, 4))
    with pytest.raises(ValueError, match="same number of dimensions"):
        _pad_and_concatenate([a, b])


def test_pad_and_concatenate_multiple_arrays() -> None:
    """Works correctly with more than two arrays."""
    a = np.ones((1, 2))
    b = np.full((2, 4), 2.0)
    c = np.full((1, 3), 3.0)
    result = _pad_and_concatenate([a, b, c])
    assert result.shape == (4, 4)
    # a: row 0, cols 0-1 = 1, cols 2-3 = 0
    np.testing.assert_array_equal(result[0, :2], 1.0)
    np.testing.assert_array_equal(result[0, 2:], 0.0)
    # b: rows 1-2, all cols = 2
    np.testing.assert_array_equal(result[1:3, :4], 2.0)
    # c: row 3, cols 0-2 = 3, col 3 = 0
    np.testing.assert_array_equal(result[3, :3], 3.0)
    np.testing.assert_array_equal(result[3, 3:], 0.0)

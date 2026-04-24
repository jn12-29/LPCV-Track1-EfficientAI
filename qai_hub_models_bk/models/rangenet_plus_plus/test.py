# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import pytest

from qai_hub_models.models.rangenet_plus_plus.app import (
    RangeNetApp,
    project_points_to_range_image,
)
from qai_hub_models.models.rangenet_plus_plus.demo import main as demo_main
from qai_hub_models.models.rangenet_plus_plus.model import (
    OUTPUT_MASK_ADDRESS,
    SAMPLE_POINT_CLOUD_ADDRESS,
    RangeNetPlusPlus,
)
from qai_hub_models.utils.testing import assert_most_same, skip_clone_repo_check


@skip_clone_repo_check
def test_task() -> None:
    model = RangeNetPlusPlus.from_pretrained()
    app = RangeNetApp(model)

    # Load the same sample point cloud used by the demo
    points = np.fromfile(
        str(SAMPLE_POINT_CLOUD_ADDRESS.fetch()), dtype=np.float32
    ).reshape(-1, 4)
    range_image, _, _ = project_points_to_range_image(points)

    # Run inference
    mask = app.segment_range_image(range_image, raw_output=True)

    expected_mask = np.load(str(OUTPUT_MASK_ADDRESS.fetch()))
    assert_most_same(mask, expected_mask, diff_tol=0.0)


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    model = RangeNetPlusPlus.from_pretrained().convert_to_torchscript()
    app = RangeNetApp(model)

    points = np.fromfile(
        str(SAMPLE_POINT_CLOUD_ADDRESS.fetch()), dtype=np.float32
    ).reshape(-1, 4)
    range_image, _, _ = project_points_to_range_image(points)

    mask = app.segment_range_image(range_image, raw_output=True)
    expected_mask = np.load(str(OUTPUT_MASK_ADDRESS.fetch()))
    assert_most_same(mask, expected_mask, diff_tol=0.0)


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)

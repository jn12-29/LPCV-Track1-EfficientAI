# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np

from qai_hub_models.models.pointnet import App, Model
from qai_hub_models.models.pointnet.demo import DATASET_ADDR
from qai_hub_models.models.pointnet.demo import main as demo_main
from qai_hub_models.models.pointnet.model import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.testing import assert_most_same, skip_clone_repo_check

OUTPUT_TEST_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "tested_output/output.pt"
)


@skip_clone_repo_check
def test_task() -> None:
    model = Model.from_pretrained()
    app = App(model)
    test_loader = app.load_cloud_data(DATASET_ADDR)
    pred_output = app.predict(test_loader=test_loader)
    expected_output = load_torch(OUTPUT_TEST_ADDRESS)
    assert_most_same(np.asarray(pred_output), np.asarray(expected_output), diff_tol=0.0)


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import numpy as np
import pytest

from qai_hub_models.models.resnet34_ssd1200 import App, Model
from qai_hub_models.models.resnet34_ssd1200.demo import main as demo_main
from qai_hub_models.models.resnet34_ssd1200.model import (
    INPUT_IMAGE_ADDRESS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
)
from qai_hub_models.utils.testing import assert_most_same, skip_clone_repo_check

EXP_IMG = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "r34_ssd_demo_output.png"
)


@skip_clone_repo_check
def test_task() -> None:
    image = load_image(INPUT_IMAGE_ADDRESS)
    exp_img = load_image(EXP_IMG)
    app = App(Model.from_pretrained())
    pred_img = app.predict_boxes_from_image(image)[0]
    assert_most_same(np.asarray(pred_img), np.asarray(exp_img), diff_tol=0.02)


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    image = load_image(INPUT_IMAGE_ADDRESS)
    exp_img = load_image(EXP_IMG)
    model = Model.from_pretrained()
    input_spec = model.get_input_spec()
    traced_model = model.convert_to_torchscript(input_spec)
    app = App(traced_model)
    pred_img = app.predict_boxes_from_image(image)[0]
    assert_most_same(np.asarray(pred_img), np.asarray(exp_img), diff_tol=0.02)

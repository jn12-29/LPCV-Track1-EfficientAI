# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import numpy as np

from qai_hub_models.models.yolo26_det.app import Yolo26DetectionApp
from qai_hub_models.models.yolo26_det.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolo26_det.demo import main as demo_main
from qai_hub_models.models.yolo26_det.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Yolo26Detector,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
)
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/output_image.png"
)
WEIGHTS = "yolo26n.pt"


@skip_clone_repo_check
def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    app = Yolo26DetectionApp(
        Yolo26Detector.from_pretrained(WEIGHTS), nms_score_threshold=0.25
    )
    assert np.allclose(app.predict_boxes_from_image(image)[0], np.asarray(output_image))


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)

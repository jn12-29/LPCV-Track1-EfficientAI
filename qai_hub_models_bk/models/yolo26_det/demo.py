# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolo26_det.app import Yolo26DetectionApp
from qai_hub_models.models.yolo26_det.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Yolo26Detector,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/input_image.jpg"
)


def main(is_test: bool = False) -> None:
    yolo_detection_demo(
        Yolo26Detector,
        MODEL_ID,
        Yolo26DetectionApp,
        IMAGE_ADDRESS,
        is_test=is_test,
        default_score_threshold=0.25,
    )


if __name__ == "__main__":
    main()

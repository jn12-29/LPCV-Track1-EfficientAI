# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.yolov9_det.app import YoloV9DetectionApp
from qai_hub_models.models.yolov9_det.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV9Detector,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/input_image.jpg"
)


def main(is_test: bool = False) -> None:
    yolo_detection_demo(
        YoloV9Detector,
        MODEL_ID,
        YoloV9DetectionApp,
        IMAGE_ADDRESS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()

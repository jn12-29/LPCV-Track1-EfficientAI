# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.yolo.demo import yolo_detection_demo
from qai_hub_models.models.resnet34_ssd1200.app import ResNet34SSDApp
from qai_hub_models.models.resnet34_ssd1200.model import (
    INPUT_IMAGE_ADDRESS,
    MODEL_ID,
    Resnet34SSD,
)


def main(is_test: bool = False) -> None:
    yolo_detection_demo(
        Resnet34SSD,
        MODEL_ID,
        ResNet34SSDApp,
        INPUT_IMAGE_ADDRESS,
        is_test=is_test,
    )


if __name__ == "__main__":
    main()

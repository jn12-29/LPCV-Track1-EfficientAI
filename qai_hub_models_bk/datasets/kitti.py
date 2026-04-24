# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetSplit,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebDatasetAsset,
    load_image,
)
from qai_hub_models.utils.image_processing import pre_process_with_affine
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

KITTI_FOLDER_NAME = "kitti"
KITTI_VERSION = 2
KITTI_IMAGES_DIR_NAME = "data_object_image_2"
KITTI_LABELS_DIR_NAME = "data_object_label_2"
KITTI_CALIBS_DIR_NAME = "data_object_calib"

# https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt
VAL_TXT = CachedWebDatasetAsset.from_asset_store(
    KITTI_FOLDER_NAME,
    KITTI_VERSION,
    "val.txt",
)
# https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt
TRAIN_TXT = CachedWebDatasetAsset.from_asset_store(
    KITTI_FOLDER_NAME,
    KITTI_VERSION,
    "train.txt",
)

KITTI_INSTALLATION_STEPS = [
    "Download images from https://www.cvlibs.net/download.php?file=data_object_image_2.zip",
    "Download annotations from https://www.cvlibs.net/download.php?file=data_object_label_2.zip",
    "Download calibrations from https://www.cvlibs.net/download.php?file=data_object_calib.zip",
    "Run `python -m qai_hub_models.datasets.configure_dataset --dataset kitti --files /path/to/data_object_image_2.zip /path/to/data_object_label_2.zip /path/to/data_object_calib.zip`",
]

KITTI_IMAGES_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/kitti/v{KITTI_VERSION}/data_object_image_2.zip",
    KITTI_FOLDER_NAME,
    KITTI_VERSION,
    f"{KITTI_IMAGES_DIR_NAME}.zip",
    installation_steps=KITTI_INSTALLATION_STEPS,
)

KITTI_LABELS_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/kitti/v{KITTI_VERSION}/data_object_label_2.zip",
    KITTI_FOLDER_NAME,
    KITTI_VERSION,
    f"{KITTI_LABELS_DIR_NAME}.zip",
    installation_steps=KITTI_INSTALLATION_STEPS,
)

KITTI_CALIBS_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/kitti/v{KITTI_VERSION}/data_object_calib.zip",
    KITTI_FOLDER_NAME,
    KITTI_VERSION,
    f"{KITTI_CALIBS_DIR_NAME}.zip",
    installation_steps=KITTI_INSTALLATION_STEPS,
)


class KittiDataset(BaseDataset):
    def __init__(
        self,
        input_images_zip: str | None = None,
        input_labels_zip: str | None = None,
        input_calibs_zip: str | None = None,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.input_images_zip = input_images_zip
        self.input_labels_zip = input_labels_zip
        self.input_calibs_zip = input_calibs_zip
        self.calib_data_path = (
            KITTI_CALIBS_ASSET.extracted_path
            / ("training" if split == DatasetSplit.TRAIN else "testing")
            / "calib"
        )
        self.image2_data_path = (
            KITTI_IMAGES_ASSET.extracted_path
            / ("training" if split == DatasetSplit.TRAIN else "testing")
            / "image_2"
        )
        BaseDataset.__init__(
            self, KITTI_CALIBS_ASSET.extracted_path.parent, split=split
        )

        input_spec = input_spec or {"image": ((1, 3, 384, 1280), "")}
        self.input_width = input_spec["image"][0][3]
        self.input_height = input_spec["image"][0][2]
        with open(
            VAL_TXT.fetch() if split == DatasetSplit.VAL else TRAIN_TXT.fetch()
        ) as image_set_f:
            image_set = image_set_f.readlines()

        self.sample: list[dict[str, Any]] = []

        for line in image_set:
            if line[-1] == "\n":
                line = line[:-1]
            image_id = int(line)

            self.sample.append(
                {
                    "img_id": image_id,
                    "img_path": self.image2_data_path / f"{line}.png",
                    "calib_path": self.calib_data_path / f"{line}.txt",
                }
            )

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image_tensor : torch.Tensor
             Normalized image tensor [C, H, W], RGB [0-1]

        gt_data : tuple[int, np.ndarray, np.ndarray, np.ndarray]
            img_id
                image id
            center
                center of the image with shape (2,)
            scale
                scale of the image with shape (2,)
            calib
                camera calibration matrix with shape (3, 4)
        """
        image_path: str = self.sample[index]["img_path"]
        img_id: int = self.sample[index]["img_id"]

        calib_path: str = self.sample[index]["calib_path"]
        with open(calib_path) as calib_f:
            calib_str = calib_f.readlines()[2][:-1]
        calib = np.array(calib_str.split(" ")[1:], dtype=np.float32)
        calib = calib.reshape(3, 4)

        image = np.array(load_image(image_path))
        height, width = image.shape[0:2]
        c = np.array([width / 2, height / 2])
        s = np.array([width, height])

        image_tensor = pre_process_with_affine(
            image, c, s, 0, (self.input_height, self.input_width)
        ).squeeze(0)

        return image_tensor, (img_id, c, s, calib)

    def __len__(self) -> int:
        return len(self.sample)

    def _download_data(self) -> None:
        KITTI_IMAGES_ASSET.fetch(extract=True, local_path=self.input_images_zip)
        KITTI_LABELS_ASSET.fetch(extract=True, local_path=self.input_labels_zip)
        KITTI_CALIBS_ASSET.fetch(extract=True, local_path=self.input_calibs_zip)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 100

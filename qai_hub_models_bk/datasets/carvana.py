# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

CARVANA_VERSION = 2
CARVANA_DATASET_ID = "carvana"
IMAGES_DIR_NAME = "train"
GT_DIR_NAME = "train_masks"

CARVANA_INSTALLATION_STEPS = [
    "Go to https://www.kaggle.com/c/carvana-image-masking-challenge and make an account",
    "Go to https://www.kaggle.com/c/carvana-image-masking-challenge/data and download `train.zip` and `train_masks.zip`",
    "Run `python -m qai_hub_models.datasets.configure_dataset --dataset carvana --files /path/to/train.zip /path/to/train_masks.zip",
]

CARVANA_IMAGES_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/carvana/v{CARVANA_VERSION}/train.zip",
    CARVANA_DATASET_ID,
    CARVANA_VERSION,
    f"data/{IMAGES_DIR_NAME}.zip",
    installation_steps=CARVANA_INSTALLATION_STEPS,
)

CARVANA_GT_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/carvana/v{CARVANA_VERSION}/train_masks.zip",
    CARVANA_DATASET_ID,
    CARVANA_VERSION,
    f"data/{GT_DIR_NAME}.zip",
    installation_steps=CARVANA_INSTALLATION_STEPS,
)


class CarvanaDataset(BaseDataset):
    """Wrapper class around carvana dataset"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_images_zip: str | None = None,
        input_gt_zip: str | None = None,
    ) -> None:
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            CARVANA_DATASET_ID, CARVANA_VERSION, "data"
        )
        self.images_path = self.data_path / IMAGES_DIR_NAME
        self.gt_path = self.data_path / GT_DIR_NAME
        self.input_images_zip = input_images_zip
        self.input_gt_zip = input_gt_zip

        BaseDataset.__init__(self, self.data_path, split=split)

        self.input_height = 640
        self.input_width = 1280

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get dataset item.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image_tensor : torch.Tensor
            Normalized image tensor [C, H, W]
        mask_tensor : torch.Tensor
            Binary mask tensor [H, W] (0=background, 1=car)
        """
        orig_image = Image.open(self.images[index]).convert("RGB")
        image = orig_image.resize((self.input_width, self.input_height), Image.BILINEAR)

        _, img_tensor = app_to_net_image_inputs(image)
        img_tensor = img_tensor.squeeze(0)

        # Load and process mask
        orig_mask = Image.open(self.masks[index])
        mask = orig_mask.resize((self.input_width, self.input_height), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask)).float()

        return img_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.images)

    def _validate_data(self) -> bool:
        if not self.images_path.exists() or not self.gt_path.exists():
            return False
        self.im_ids = []
        self.images = []
        self.masks = []
        # Match images with their corresponding masks
        for image_path in sorted(self.images_path.glob("*.jpg")):
            im_id = image_path.stem
            mask_path = self.gt_path / f"{im_id}_mask.gif"
            if mask_path.exists():
                self.im_ids.append(im_id)
                self.images.append(image_path)
                self.masks.append(mask_path)

        if not self.images:
            raise ValueError(
                f"No valid image-mask pairs found in {self.images_path} and {self.gt_path}"
            )

        return True

    def _download_data(self) -> None:
        CARVANA_IMAGES_ASSET.fetch(extract=True, local_path=self.input_images_zip)
        CARVANA_GT_ASSET.fetch(extract=True, local_path=self.input_gt_zip)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.kaggle.com/competitions/carvana-image-masking-challenge",
            split_description="train split",
        )

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch
from torchvision.datasets import ImageFolder

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.image_processing import app_to_net_image_inputs
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

DATASET_VERSION = 2
DATASET_ID = "human_faces_dataset"

HUMAN_FACES_PRIVATE_ASSET = CachedPrivateDatasetAsset(
    "qai-hub-models/datasets/human_faces/faces.zip",
    DATASET_ID,
    DATASET_VERSION,
    "data.zip",
    installation_steps=[
        "Download the dataset from https://www.kaggle.com/datasets/ashwingupta3012/human-face",
        "Run `python -m qai_hub_models.datasets.configure_dataset --dataset human_faces --files /path/to/zip`",
    ],
    local_cache_extracted_path="data/Humans",
)


class HumanFacesDataset(BaseDataset):
    """
    Wrapper class for human faces dataset

    https://www.kaggle.com/datasets/ashwingupta3012/human-faces
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        width: int = 256,
        height: int = 256,
    ) -> None:
        self.images_path = HUMAN_FACES_PRIVATE_ASSET.extracted_path
        self.data_path = self.images_path.parent
        self.input_data_zip = input_data_zip

        self.img_width = width
        self.img_height = height
        self.scale_width = 1.0 / self.img_width
        self.scale_height = 1.0 / self.img_height
        BaseDataset.__init__(self, self.data_path, split=split)
        self.dataset = ImageFolder(str(self.data_path))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, _ = self.dataset[index]
        image = image.resize((self.img_width, self.img_height))
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return image_tensor, 0

    def __len__(self) -> int:
        return len(self.dataset)

    def _validate_data(self) -> bool:
        return self.images_path.exists() and len(os.listdir(self.images_path)) >= 100

    def _download_data(self) -> None:
        HUMAN_FACES_PRIVATE_ASSET.fetch(extract=True, local_path=self.input_data_zip)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 1000


class HumanFaces192Dataset(HumanFacesDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
    ) -> None:
        super().__init__(split, input_data_zip, 192, 192)

    @classmethod
    def dataset_name(cls) -> str:
        """
        Name for the dataset,
            which by default is set to the filename where the class is defined.
        """
        return "human_faces_192"

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.kaggle.com/datasets/ashwingupta3012/human-faces",
            split_description="validation split",
        )

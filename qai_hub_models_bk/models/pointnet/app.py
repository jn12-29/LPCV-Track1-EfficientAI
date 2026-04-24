# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from qai_hub_models.models.pointnet.model import (
    COMMIT_HASH,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    PATCHES,
    SOURCE_REPO,
)
from qai_hub_models.utils.asset_loaders import SourceAsRoot


class PointNetApp:
    """
    This class consists of lightweight 'app code' required to perform end-to-end inference for point cloud classification.

    The app uses the PointNet model for classifying 3D point cloud data.
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Initialize the PointNetApp with a pre-trained model.

        Parameters
        ----------
        model
            A callable model that takes a torch.Tensor of shape (B, 3, N)
            and returns classification logits.
        """
        self.model = model

    def load_cloud_data(self, path: str) -> DataLoader:
        """
        Loads and preprocesses point cloud data from the specified path.

        This method applies a series of transformations to the raw point cloud data:
            - Sampling 1024 points from each cloud
            - Normalizing the point cloud
            - Applying random rotation around the Z-axis
            - Adding random noise
            - Converting to PyTorch tensor

        Parameters
        ----------
        path
            Path to the directory containing point cloud data.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader that yields batches of transformed point cloud data.
            Each batch is a dictionary with a key "pointcloud" containing a tensor of shape (B, N, 3).
        """
        with SourceAsRoot(
            SOURCE_REPO,
            COMMIT_HASH,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            source_repo_patches=PATCHES,
        ):
            from source import dataset, utils

        test_transforms = transforms.Compose(
            [
                utils.PointSampler(1024),
                utils.Normalize(),
                utils.RandRotation_z(),
                utils.RandomNoise(),
                utils.ToTensor(),
            ]
        )

        test_ds = dataset.PointCloudData(
            path, valid=True, folder="test", transform=test_transforms
        )
        return DataLoader(dataset=test_ds, batch_size=1)

    def predict(self, test_loader: DataLoader) -> torch.Tensor:
        """
        Performs inference on the point cloud data using the PointNet model.

        Parameters
        ----------
        test_loader
            A PyTorch DataLoader that yields batches of point cloud data.
            Each batch should be a dictionary with a key "pointcloud" containing a tensor of shape (B, N, 3),
            where:
                - B is the batch size (typically 1),
                - N is the number of points per cloud (e.g., 1024),
                - 3 corresponds to the x, y, z coordinates of each point.

        Returns
        -------
        predicted_class_indices : torch.Tensor
            A tensor of predicted class indices for each input point cloud in the batch.
        """
        for data in test_loader:
            inputs = data["pointcloud"].to("cpu").float()
            with torch.no_grad():
                # Only classification logits needed; ignore critical indices and attention features
                outputs, _, _ = self.model(
                    inputs.transpose(1, 2)
                )  # Transpose to (B, 3, N)
                _, predicted = torch.max(outputs.data, 1)
        return predicted

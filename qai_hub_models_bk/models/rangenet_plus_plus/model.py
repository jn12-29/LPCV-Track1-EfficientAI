# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import importlib
import os
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML
from torch import nn
from typing_extensions import Self

from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec

RANGENET_SOURCE_REPOSITORY = "https://github.com/PRBonn/lidar-bonnetal.git"
RANGENET_SOURCE_REPO_COMMIT = "99b827f"

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

NUM_CLASSES = 20
INPUT_HEIGHT = 64
INPUT_WIDTH = 2048
INPUT_CHANNELS = 5

SAMPLE_POINT_CLOUD_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "000000.bin"
)

OUTPUT_MASK_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "rangenet_mask.npy"
)

# Source: https://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53.tar.gz
DARKNET53_MODEL_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "darknet53.tar.gz"
)


class RangeNetPlusPlus(BaseModel):
    """RangeNet++ LiDAR semantic segmentation model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path | None = None,
    ) -> Self:
        """
        Load pretrained RangeNet++.

        Parameters
        ----------
        model_dir
            Path to the darknet53 model folder. Downloaded automatically if None.

        Returns
        -------
        Self
            Loaded model instance.
        """
        if model_dir is None:
            tar_path = DARKNET53_MODEL_ASSET.fetch()
            model_dir = Path(tar_path.parent) / "darknet53"
            if not model_dir.exists():
                with tarfile.open(tar_path) as tar:
                    tar.extractall(tar_path.parent)

        model = _load_rangenet_source_model(Path(os.path.abspath(model_dir)))
        return cls(model)

    def forward(self, range_image: torch.Tensor) -> torch.Tensor:
        """
        Predict semantic labels for a LiDAR range image.

        Parameters
        ----------
        range_image
            float32 tensor of shape [1, 5, H, W] with channels ordered
            [depth, x, y, z, intensity]. Values must be pre-normalised
            per-channel using the SemanticKITTI mean/std statistics
            (see ``project_points_to_range_image`` in app.py). The
            expected input range after normalisation is approximately
            [-3, 3] for each channel.

        Returns
        -------
        torch.Tensor
            uint8 class-index mask of shape [1, H, W]. Each value is a
            class index in [0, NUM_CLASSES - 1] corresponding to the
            SemanticKITTI 20-class label set.
        """
        logits = self.model(range_image)
        return torch.argmax(logits, dim=1).byte()

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = INPUT_HEIGHT,
        width: int = INPUT_WIDTH,
    ) -> InputSpec:
        return {"range_image": ((batch_size, INPUT_CHANNELS, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["mask"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["range_image"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["mask"]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        from qai_hub_models.models.rangenet_plus_plus.app import (
            project_points_to_range_image,
        )

        points = np.fromfile(
            str(SAMPLE_POINT_CLOUD_ADDRESS.fetch()), dtype=np.float32
        ).reshape(-1, 4)
        arr, _, _ = project_points_to_range_image(points)
        return {"range_image": [arr]}


def _load_rangenet_source_model(model_dir: Path) -> nn.Module:
    config_path = model_dir / "arch_cfg.yaml"

    with SourceAsRoot(
        RANGENET_SOURCE_REPOSITORY,
        RANGENET_SOURCE_REPO_COMMIT,
        MODEL_ID,
        MODEL_ASSET_VERSION,
        source_root_subdir="train/tasks/semantic",
        imported_but_unused_modules=["vispy", "open3d"],
    ):
        if "imp" not in sys.modules:
            sys.modules["imp"] = types.ModuleType("imp")

        train_path = os.path.normpath(os.path.join(os.getcwd(), "../../"))
        if train_path not in sys.path:
            sys.path.insert(0, train_path)

        booger_stub = types.ModuleType("rangenet_bonnetal_init")
        booger_stub.TRAIN_PATH = train_path  # type: ignore[attr-defined]
        booger_stub.DEPLOY_PATH = os.path.join(train_path, "../deploy")  # type: ignore[attr-defined]
        sys.modules["rangenet_bonnetal_init"] = booger_stub
        sys.modules["__init__"] = booger_stub

        def _load_source(name: str, path: str) -> types.ModuleType:
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        sys.modules["imp"].load_source = _load_source  # type: ignore[attr-defined]

        from modules.segmentator import Segmentator

        with open(config_path) as f:
            arch_cfg = YAML(typ="safe", pure=True).load(f)

        model = Segmentator(arch_cfg, NUM_CLASSES, path=str(model_dir))
        model.eval()

    return model

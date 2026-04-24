# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, runtime_checkable

import torch
from qai_hub.client import DatasetEntries

from qai_hub_models.utils.base_model import PretrainedCollectionModel
from qai_hub_models.utils.input_spec import InputSpec

RUN_MODEL_RETURN_TYPE = list[torch.Tensor] | torch.Tensor
CollectionAppTypeVar = TypeVar("CollectionAppTypeVar", bound="CollectionAppProtocol")


class BaseCollectionApp(ABC):
    @abstractmethod
    def run_model(
        self, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> tuple[RUN_MODEL_RETURN_TYPE, ...] | RUN_MODEL_RETURN_TYPE:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model: PretrainedCollectionModel) -> BaseCollectionApp:
        pass


@runtime_checkable
class CollectionAppProtocol(Protocol):
    """Protocol for apps that provide calibration data for CollectionModels."""

    @staticmethod
    def calibration_dataset_name() -> str:
        """Name of the dataset used for calibration across all components."""
        ...

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: PretrainedCollectionModel,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        """
        Produces a numpy dataset to be used for calibration data of a quantize job.

        Parameters
        ----------
        collection_model
            The parent collection model.
        component_name
            The name of the component being calibrated.
        input_specs
            Per-component input specs. If None, uses each component's defaults.
        num_samples
            Number of data samples to use. If not specified, uses
            default specified on dataset.

        Returns
        -------
        DatasetEntries
            Dataset compatible with the format expected by AI Hub Workbench.
        """
        ...

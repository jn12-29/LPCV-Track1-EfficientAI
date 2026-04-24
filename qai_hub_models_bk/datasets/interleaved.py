# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit


class InterleavedDataset(BaseDataset):
    """Base class for datasets that interleave multiple source datasets.

    Items are selected in round-robin order: index ``i`` maps to dataset
    ``i % N`` and item ``i // N``. The total length is
    ``min_dataset_len * N`` so every dataset is exhausted evenly.

    Subclasses must set DATASET_NAME and implement load_datasets().
    """

    DATASET_NAME: str = ""

    @classmethod
    def dataset_name(cls) -> str:
        assert cls.DATASET_NAME, (
            "DATASET_NAME must be set in InterleavedDataset subclass."
        )
        return cls.DATASET_NAME

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        num_samples: int = 0,
        **kwargs: Any,
    ) -> None:
        self._datasets = self.load_datasets(split, **kwargs)
        assert len(self._datasets) > 0
        self._min_len = min(len(ds) for ds in self._datasets)
        self.num_samples = num_samples

    def load_datasets(self, split: DatasetSplit, **kwargs: Any) -> list[BaseDataset]:
        raise NotImplementedError

    @staticmethod
    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> tuple[Any, ...]:
        item = batch[0]
        if isinstance(item, dict) and "input_ids" in item and "attention_mask" in item:
            result: tuple[Any, ...] = (
                item["input_ids"],
                item["attention_mask"],
                item.get("label", item["input_ids"]),
            )
            for key in ("pixel_values", "image_grid_thw"):
                if key in item:
                    result = (*result, item[key])
            return result
        return tuple(batch)

    def __len__(self) -> int:
        total = self._min_len * len(self._datasets)
        if self.num_samples > 0:
            return min(self.num_samples, total)
        return total

    def __getitem__(self, idx: int) -> Any:
        ds_idx = idx % len(self._datasets)
        item_idx = idx // len(self._datasets)
        return self._datasets[ds_idx][item_idx]

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        return 1

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="",
            split_description="Interleaved from multiple source datasets",
        )


class InterleavedAOKVQAWikitext(InterleavedDataset):
    """Interleaves AOKVQA and Wikitext for mixed VLM calibration."""

    DATASET_NAME = "interleaved_aokvqa_wikitext"

    def load_datasets(self, split: DatasetSplit, **kwargs: Any) -> list[BaseDataset]:
        from qai_hub_models.datasets.aokvqa import AOKVQA
        from qai_hub_models.datasets.wikitext import WikiText

        return [
            AOKVQA(
                split=split,
                tokenizer=kwargs.get("tokenizer"),
                block_size=kwargs.get("block_size", 128),
                context_length=kwargs.get("context_length", 4096),
                num_samples=kwargs.get("num_samples", 0) // 2,
                processor=kwargs.get("processor"),
                image_size=kwargs.get("image_size"),
            ),
            WikiText(
                split=split,
                tokenizer=kwargs["tokenizer"],
                block_size=kwargs.get("block_size", 128),
                context_length=kwargs.get("context_length", 4096),
                num_samples=kwargs.get("num_samples", 0) // 2,
            ),
        ]

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

from datasets import load_dataset

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit

AOKVQA_HF_REPO = "HuggingFaceM4/A-OKVQA"
AOKVQA_URL = f"https://huggingface.co/datasets/{AOKVQA_HF_REPO}"


class AOKVQA(BaseDataset):
    """A-OKVQA dataset for multimodal backbone calibration.

    Each sample is an image + question + multiple-choice answers processed
    through a VLM processor. For the train split, the correct answer and
    first rationale are appended as an assistant turn, giving the quantizer
    longer, more representative sequences to calibrate on.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        block_size: int = 128,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TRAIN,
        num_samples: int = 0,
        seed: int = 42,
        processor: Any = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.block_size = block_size
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.processor = processor
        self.image_size = tuple(image_size) if image_size is not None else None
        self.include_answer = split == DatasetSplit.TRAIN

        if split == DatasetSplit.TRAIN:
            split_str = "train"
        elif split == DatasetSplit.VAL:
            split_str = "validation"
        elif split == DatasetSplit.TEST:
            split_str = "test"
        else:
            raise ValueError(f"AOKVQA does not support split: {split}")

        self.dataset = load_dataset(AOKVQA_HF_REPO, split=split_str)
        self.dataset = self.dataset.shuffle(seed)

    @staticmethod
    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> tuple[Any, ...]:
        item = batch[0]
        result: tuple[Any, ...] = (
            item["input_ids"],
            item["attention_mask"],
            item.get("label", item["input_ids"]),
        )
        for key in ("pixel_values", "image_grid_thw"):
            if key in item:
                result = (*result, item[key])
        return result

    def __len__(self) -> int:
        if self.num_samples != 0:
            return min(self.num_samples, len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]

        image = sample["image"]
        question = sample["question"]
        choices = sample["choices"]

        choices_text = "\n".join(
            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(choices)
        )

        content = [
            {"type": "image"},
            {"type": "text", "text": f"{question}\n{choices_text}\nAnswer:"},
        ]

        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

        if self.include_answer:
            answer_idx = sample["correct_choice_idx"]
            answer_letter = chr(ord("A") + answer_idx)
            answer_text = f"{answer_letter}. {choices[answer_idx]}"
            rationales = sample.get("rationales", [])
            if rationales:
                answer_text += f"\n\nReasoning: {rationales[0]}"
            messages.append({"role": "assistant", "content": answer_text})

        assert self.processor is not None, (
            "AOKVQA requires a VLM processor (not just a tokenizer)"
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not self.include_answer,
        )

        if self.image_size is not None:
            image = image.resize(self.image_size)

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        inputs["input_ids"] = inputs["input_ids"][:, -self.context_length :]
        inputs["attention_mask"] = inputs["attention_mask"][:, -self.context_length :]
        inputs.pop("mm_token_type_ids", None)

        return inputs

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        return 1

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link=AOKVQA_URL,
            split_description="train split (for calibration)",
        )

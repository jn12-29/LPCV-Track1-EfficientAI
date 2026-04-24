# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from transformers import ElectraForPreTraining, ElectraTokenizer
from typing_extensions import Self

from qai_hub_models.datasets import DATASET_NAME_MAP
from qai_hub_models.datasets.wikitext_masked import ElectraWikiTextMasked
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.electra_discriminator_evaluator import (
    ElectraDiscriminatorEvaluator,
)
from qai_hub_models.models._shared.bert_hf.model import BaseBertModel
from qai_hub_models.models._shared.bert_hf.model_patches import (
    patch_get_extended_attention_mask,
)
from qai_hub_models.utils.input_spec import InputSpec, IoType, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
WEIGHTS_NAME = "google/electra-base-discriminator"


class ElectraBertBaseDiscrimGoogle(BaseBertModel):
    """Exportable HuggingFace ElectraBertBaseDiscrimGoogle Model"""

    @classmethod
    def from_pretrained(cls, weights: str = WEIGHTS_NAME) -> Self:
        """Load HuggingFace Bert Model for Embeddings."""
        model = ElectraForPreTraining.from_pretrained(weights)
        tokenizer = ElectraTokenizer.from_pretrained(weights)
        model.electra.get_extended_attention_mask = patch_get_extended_attention_mask
        return cls(model, tokenizer)

    def get_evaluator(self) -> BaseEvaluator:
        return ElectraDiscriminatorEvaluator()

    @staticmethod
    def eval_datasets() -> list[str]:
        return ["electra_bert_wikitext_masked"]

    @staticmethod
    def calibration_dataset_name() -> str:
        return "electra_bert_wikitext_masked"

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_tokens
            Input token IDs with shape [batch_size, seq_len]
        attention_masks
            Attention masks with shape [batch_size, seq_len]

        Returns
        -------
        predictions : torch.Tensor
            Binary predictions for every token position, shape [batch_size, seq_len].
            Values: 1 = fake/replaced, 0 = real.
        """
        logits = self.model(input_tokens, attention_mask=attention_masks).logits
        return (logits > 0).float()

    @classmethod
    def get_dataset_class(cls, tokenizer_name: str) -> type:
        return ElectraWikiTextMasked

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        sample_length: int = 384,
    ) -> InputSpec:
        return {
            "input_tokens": TensorSpec(
                shape=(batch_size, sample_length),
                dtype="int32",
                io_type=IoType.TENSOR,
            ),
            "attention_masks": TensorSpec(
                shape=(batch_size, sample_length),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["predictions"]


DATASET_NAME_MAP["electra_bert_wikitext_masked"] = ElectraWikiTextMasked

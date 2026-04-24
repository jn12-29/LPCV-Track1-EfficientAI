# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import Gemma3nTextConfig  # type: ignore[attr-defined, unused-ignore]
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.gemma3n import Gemma3nForCausalLM, modeling_gemma3n
from typing_extensions import Self

from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    LLM_QNN,
    Embedding,
    LLM_AIMETOnnx,
    LLMBase,
    PositionProcessorBase,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

NUM_LAYERS = 30
NUM_SPLITS = 3
NUM_LAYERS_PER_SPLIT = 15
HIDDEN_SIZE = 2048
NUM_KEY_VALUE_HEADS = 2
NUM_ATTN_HEADS = 8

# Hugging face repo name and url
HF_REPO_NAME = "google/gemma-3n-E2B-it"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Minimum memory (RAM+swap) recommended for export.
MIN_MEMORY_RECOMMENDED = 30

END_TOKENS = {"<end_of_turn>", "<eos>"}

DEFAULT_PRECISION = Precision.float
SUPPORTED_PRECISIONS = [Precision.float]


class RopeEmbedding(Embedding):
    def __init__(
        self,
        head_dim: int | None = None,
        max_length: int = 2048,
        config: Gemma3nTextConfig | None = None,
    ) -> None:
        if config is None:
            config = Gemma3nTextConfig()
        head_dim = head_dim or (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // config.num_attention_heads
        )
        self.cos, self.sin = self.precompute(head_dim, max_length, config)

    def precompute(
        self, head_dim: int, max_length: int, config: Gemma3nTextConfig
    ) -> list[torch.Tensor]:
        if not hasattr(config, "rope_scaling"):
            config.rope_scaling = None

        rope = modeling_gemma3n.Gemma3nTextRotaryEmbedding(config=config)
        dummy_x = torch.Tensor([1.0])
        position_ids = torch.arange(max_length).view(1, -1)
        if hasattr(rope, "_original_forward"):
            embeddings = rope._original_forward(dummy_x, position_ids)  # type: ignore[operator, unused-ignore]
        else:
            embeddings = rope.forward(dummy_x, position_ids)

        # for adapted llama
        emb_size = embeddings[0].size(-1) // 2
        return [emb[:, :, :emb_size].unsqueeze(0) for emb in embeddings]

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_sim//2][2]
        """
        cos = self.cos[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        sin = self.sin[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


class GemmaPositionProcessor(PositionProcessorBase):
    """Prepares positions (RopeEmbedding and attention mask preparation); used by ORT GenAI."""

    def __init__(
        self,
        context_length: int,
        config: Gemma3nTextConfig,
    ) -> None:
        super().__init__(context_length, config=config)
        self.context_len = context_length
        self.rope_embedding = RopeEmbedding(max_length=self.context_len, config=config)

    def forward(
        self, attention_mask_before_processor: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position_ids_cos, position_ids_sin = self.rope_embedding.get_embedding(
            position_ids
        )
        attention_mask_converter = AttentionMaskConverter(True)
        attention_mask = attention_mask_converter.to_4d(
            attention_mask_before_processor,
            query_length=position_ids.shape[1],
            key_value_length=attention_mask_before_processor.shape[1],
            dtype=torch.float32,
        )
        attention_mask = attention_mask.clip(-50, 0)
        return attention_mask, position_ids_cos, position_ids_sin


class Gemma_3n_E2B(LLMBase):
    LMClass = Gemma3nForCausalLM
    EmbeddingClass = RopeEmbedding
    CacheClass = DynamicCache
    llm_io_type: LLMIOType = LLMIOType.huggingface_input_ids

    min_memory_recommended = MIN_MEMORY_RECOMMENDED

    # Default prompts for demos
    default_user_prompt = "What is gravity?"
    default_system_prompt = "You are a helpful assistant."

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path = HF_REPO_NAME,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            checkpoint,
            *args,
            **kwargs,
        )

    def _verify_ckpt(self) -> None:
        super()._verify_ckpt()
        """
        if not (
            self.llm_config.num_hidden_layers == NUM_LAYERS
            and self.llm_config.hidden_size == HIDDEN_SIZE
            and self.llm_config.num_attention_heads == NUM_ATTN_HEADS
            and self.llm_config.num_key_value_heads == NUM_KEY_VALUE_HEADS
        ):
            raise ValueError("Model config is not compatible with our implementation.")
        """

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = HF_REPO_NAME,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        load_pretrained: bool = True,
        _skip_optimizations: list[str] | None = None,
    ) -> Self:
        """
        Load a pre-trained Gemma-3n (2B) model from Google via HuggingFace.

        checkpoint:
            Local path or Hugging Face name of floating point checkpoint.
        sequence_length:
            Instantiate with this token sequence length input. A longer
            sequence length means the model is capable of processing more
            tokens at once. This can only be set to greater than one to process
            prompts, since responses are auto-regressive in nature and require
            this to be 1.
        context_length:
            Total context length of model. Longer context length means the
            model is more capable of making longer connections in the input
            prompt. However, it also hurts runtime performance (both time-to-
            first-token and tokens-per-second), so this is a tradeoff that may
            depend on the use case.
        """
        # Currently this model is not adapted and requires this:
        skip_optimizations = [*(_skip_optimizations or []), "sha_attention"]

        return cls(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            load_pretrained=load_pretrained,
            _skip_optimizations=skip_optimizations,
        )

    @staticmethod
    def get_output_names() -> list[str]:
        return LLMBase._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict[str, Any],
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return LLMBase._get_input_spec(
            num_hidden_layers=llm_config["num_hidden_layers"],
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config["hidden_size"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            num_attention_heads=llm_config["num_attention_heads"],
            llm_io_type=llm_io_type,
        )


class Gemma_3n_E2B_AIMETOnnx(LLM_AIMETOnnx):
    FPModel = Gemma_3n_E2B
    EmbeddingClass = RopeEmbedding
    llm_io_type: LLMIOType = Gemma_3n_E2B.llm_io_type

    def __init__(
        self, checkpoint: str | os.PathLike | Path | None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(
            checkpoint,
            *args,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path | None = "DEFAULT",
        host_device: torch.device | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        precision: Precision = DEFAULT_PRECISION,
        fp_model: LLMBase | None = None,
        _skip_quantsim_creation: bool = False,
        use_dynamic_shapes: bool = False,
    ) -> Self:
        """
        Load weight from Huggingface and create Aimet-ONNX QuantSim.
        Optionally load onnx model and AIMET encodings from a checkpoint.

        Parameters
        ----------
        checkpoint
            Path to previously calibrated AIMET encodings and ONNX
            models. Note that encodings are sensitive to AIMET ONNX versions.
            If passing None, initializes without encodings.
        host_device
            Device on which to load the model.
        sequence_length
            Instantiate with this token sequence length input.
        context_length
            Total context length of model.
        precision
            Precision to use for model weights.
        fp_model
            Optionally provide a floating point model to convert.
        _skip_quantsim_creation
            Internal parameter to skip quantsim creation. This helps export on platforms where aimet onnx is not available.
        use_dynamic_shapes
            Whether to use dynamic shapes for ONNX export.

        Returns
        -------
        model : Self
            Instantiated quantized model.
        """
        if host_device is None:
            host_device = torch.device("cpu")
        return super().from_pretrained(
            checkpoint=checkpoint,
            host_device=host_device,
            sequence_length=sequence_length,
            context_length=context_length,
            precision=precision,
            fp_model=fp_model,
            _skip_quantsim_creation=_skip_quantsim_creation,
            use_dynamic_shapes=use_dynamic_shapes,
        )

    @staticmethod
    def get_output_names() -> list[str]:
        return LLMBase._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict[str, Any],
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return LLMBase._get_input_spec(
            num_hidden_layers=NUM_LAYERS,
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=HIDDEN_SIZE,
            num_key_value_heads=NUM_KEY_VALUE_HEADS,
            num_attention_heads=NUM_ATTN_HEADS,
            llm_io_type=llm_io_type,
        )


class Gemma_3n_E2B_QNN(LLM_QNN):
    num_layers_per_split: int = NUM_LAYERS_PER_SPLIT

    @staticmethod
    def get_output_names() -> list[str]:
        return LLMBase._get_output_names(NUM_LAYERS)

    get_input_spec = staticmethod(Gemma_3n_E2B.get_input_spec)

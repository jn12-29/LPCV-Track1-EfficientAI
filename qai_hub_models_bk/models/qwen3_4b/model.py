# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from typing_extensions import Self

from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS as GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS as GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS,
)
from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    determine_precision_from_checkpoint,
)
from qai_hub_models.models._shared.qwen3.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    Qwen3Base,
    Qwen3Base_AIMETOnnx,
    Qwen3Base_QNN,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import InputSpec

DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Qwen3-4B model configuration
NUM_LAYERS = 36
NUM_SPLITS = 4
NUM_LAYERS_PER_SPLIT = 12
HIDDEN_SIZE = 2560
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32

# Hugging face repo name and url
HF_REPO_NAME = "Qwen/Qwen3-4B"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Minimum memory (RAM+swap) recommended for export.
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3
MIN_MEMORY_RECOMMENDED = 40
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT: dict[Precision, str] = {
    Precision.w4a16: "qwen34_w4a16_adascale",
}


class Qwen3_4B(Qwen3Base):
    min_memory_recommended = MIN_MEMORY_RECOMMENDED

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path = HF_REPO_NAME,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            checkpoint=checkpoint,  # type: ignore[misc]
            *args,  # noqa: B026
            **kwargs,
        )

    def _verify_ckpt(self) -> None:
        super()._verify_ckpt()
        if not (
            self.llm_config.num_hidden_layers == NUM_LAYERS
            and self.llm_config.hidden_size == HIDDEN_SIZE
            and self.llm_config.num_attention_heads == NUM_ATTN_HEADS
            and self.llm_config.num_key_value_heads == NUM_KEY_VALUE_HEADS
        ):
            raise ValueError("Model config is not compatible with our implementation.")

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
        Load a pre-trained Qwen3-4B model via HuggingFace.

        Parameters
        ----------
        checkpoint
            Local path or Hugging Face name of floating point checkpoint.
        sequence_length
            Instantiate with this token sequence length input. A longer
            sequence length means the model is capable of processing more
            tokens at once. This can only be set to greater than one to process
            prompts, since responses are auto-regressive in nature and require
            this to be 1.
        context_length
            Total context length of model. Longer context length means the
            model is more capable of making longer connections in the input
            prompt. However, it also hurts runtime performance (both time-to-
            first-token and tokens-per-second), so this is a tradeoff that may
            depend on the use case.
        host_device
            Device of the host computer.
        load_pretrained
            Whether to load pretrained weights.
        _skip_optimizations
            List of optimizations to skip.

        Returns
        -------
        model : Self
            The pre-trained Qwen3-4B model.
        """
        # Since we multiply the attention mask for Qwen3, the default value has
        # issues so we use the Genie value for the unquantized variant too.
        attention_mask_min_clip = -1000.0

        return cls(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            load_pretrained=load_pretrained,
            attention_mask_min_clip=attention_mask_min_clip,
            _skip_optimizations=_skip_optimizations,
        )

    @staticmethod
    def get_output_names() -> list[str]:
        return Qwen3Base._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return Qwen3Base._get_input_spec(
            num_hidden_layers=llm_config["num_hidden_layers"],
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config["hidden_size"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            num_attention_heads=llm_config["num_attention_heads"],
            head_dim=llm_config.get("head_dim"),
            llm_io_type=llm_io_type,
        )


class Qwen3_4B_AIMETOnnx(Qwen3Base_AIMETOnnx):
    ada_scale_num_rmsnorm_per_blk: int | None = NUM_ATTN_HEADS + NUM_KEY_VALUE_HEADS + 1

    @classmethod
    def attention_mask_min_clip_and_multiplier(
        cls,
        precision: Precision,
    ) -> tuple[float | None, float]:
        return (-100.0, 1.0)

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
            Device of the host computer.
        sequence_length
            Sequence length for the model.
        context_length
            Context length for the model.
        precision
            Target quantization precision of the model.
        fp_model
            Optional floating point model instance.
        _skip_quantsim_creation
            Internal parameter to skip quantsim creation. This helps export on platforms where aimet onnx is not available.
        use_dynamic_shapes
            Whether to use dynamic shapes for ONNX export.

        Returns
        -------
        model : Self
            The quantized Qwen3-4B model.
        """
        if host_device is None:
            host_device = torch.device("cpu")
        if isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT"):
            precision = determine_precision_from_checkpoint(checkpoint) or precision
            if precision not in SUPPORTED_PRECISIONS:
                available_precisions = [str(p) for p in SUPPORTED_PRECISIONS]
                raise ValueError(
                    f"This model is not supported for {precision!s} precision. "
                    f"Models are available in following precisions: {','.join(available_precisions)}."
                )
            if precision not in DEFAULT_CHECKPOINT:
                available_checkpoints = [str(p) for p in DEFAULT_CHECKPOINT]
                raise ValueError(
                    f"No checkpoint is available for this model in {precision!s} precision. If you would "
                    f"like to continue with this precision, please generate a local quantized checkpoint. "
                    f"Checkpoints are available in the following precisions: {','.join(available_checkpoints)}."
                )
            precision_checkpoint = DEFAULT_CHECKPOINT[precision]
            checkpoint = str(
                CachedWebModelAsset.from_asset_store(
                    MODEL_ID,
                    MODEL_ASSET_VERSION,
                    precision_checkpoint + ".zip",
                ).fetch(extract=True)
            )
            # Generate necessary ONNX models
            if fp_model is not None:
                cls.create_onnx_models(
                    checkpoint=checkpoint,
                    fp_model=fp_model,
                    context_length=context_length,
                    export_sequence_lengths=[sequence_length],
                    host_device=host_device,
                    llm_io_type=fp_model.llm_io_type,
                )

                cls.save_tokenizer_and_config(checkpoint=checkpoint, fp_model=fp_model)
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
        return Qwen3Base._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return Qwen3Base._get_input_spec(
            num_hidden_layers=llm_config["num_hidden_layers"],
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config["hidden_size"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            num_attention_heads=llm_config["num_attention_heads"],
            head_dim=llm_config.get("head_dim"),
            llm_io_type=llm_io_type,
        )


class Qwen3_4B_QNN(Qwen3Base_QNN):
    num_layers_per_split: int = NUM_LAYERS_PER_SPLIT

    @staticmethod
    def get_output_names() -> list[str]:
        return Qwen3Base._get_output_names(NUM_LAYERS)

    get_input_spec = staticmethod(Qwen3_4B.get_input_spec)

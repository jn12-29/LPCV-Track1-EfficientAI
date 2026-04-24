# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Llama 3.2 1B Instruct2 - PreSplit-Part architecture for LLM deployment.

Architecture:
- Llama3_2_1B_PreSplit (Singleton, FP): Manages full model + ONNX splitting
- Llama3_2_1B_QuantizablePreSplit (Singleton): Manages QuantSim + calibration
- Llama3_2_1B_PartBase -> Part1, Part2, Part3: Unified split inference
  (handles both FP and Quantizable modes based on precision)
- Collection class for deploying as 3 splits
"""

from __future__ import annotations

import contextlib
import logging

# isort: off
# This verifies aimet is installed, and this must be included first.
with contextlib.suppress(ImportError, ModuleNotFoundError):
    from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim
# isort: on
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import torch
from typing_extensions import Self

from qai_hub_models.models._shared.llama3.model import (
    Llama3Base,
    Llama3Base_AIMETOnnx,
    LlamaQuantizablePreSplitMixin,
)
from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    LLM_AIMETOnnx,
    PreSplitOnnxMixin,
    SingleSlotCacheMixin,
    SplitForwardMixin,
    get_onnx_model,
)
from qai_hub_models.models.common import (
    Precision,
    SampleInputsType,
    SourceModelFormat,
)
from qai_hub_models.utils.base_model import (
    CollectionModel,
    MultiGraphBaseModel,
    MultiGraphPretrainedCollectionModel,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle, mock_torch_onnx_inference
from qai_hub_models.utils.printing import print_with_box

logger = logging.getLogger(__name__)

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3

# Model architecture constants (from Llama 3.2 1B)
NUM_LAYERS = 16
NUM_SPLITS = 3
NUM_LAYERS_PER_SPLIT = 8
HIDDEN_SIZE = 2048
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32

# Hugging Face repo
HF_REPO_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 50

# Precision settings
DEFAULT_PRECISION = Precision.w4
SUPPORTED_PRECISIONS = [Precision.w4, Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4: "w4",
    Precision.w4a16: "w4a16",
}

# Name used for split ONNX file basenames (e.g. Llama3_2_1B_1_of_3.onnx)
SPLIT_MODEL_NAME = "Llama3_2_1B"

# ---------------------------------------------------------------------------
# Llama3_2_1B_PreSplit - FP PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Llama3_2_1B_PreSplit(SingleSlotCacheMixin, PreSplitOnnxMixin, Llama3Base):
    """
    FP PreSplit for Llama 3.2 1B.

    Manages the full torch model and ONNX splitting. Uses class-level cache
    keyed by checkpoint to reuse instances across calls with different
    sequence/context lengths (dynamic shapes). When a different checkpoint
    is requested, the old instance is evicted and freed.
    """

    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT

    def __init__(
        self,
        checkpoint: str | Path = HF_REPO_NAME,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, checkpoint=checkpoint, **kwargs)

    def _verify_ckpt(self) -> None:
        """Verify checkpoint compatibility."""
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
        checkpoint: str | Path = HF_REPO_NAME,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        _skip_optimizations: list[str] | None = None,
    ) -> Llama3_2_1B_PreSplit:
        """
        Load or return a cached FP PreSplit.

        If a cached instance exists for the same checkpoint, returns it
        with updated sequence/context lengths (dynamic shapes). If a
        different checkpoint is requested, the old instance is freed first.
        """
        cache_key = str(checkpoint)
        cached = cls.cache_lookup(cache_key)
        if cached is not None:
            cached.sequence_length = sequence_length
            cached.context_length = context_length
            return cached

        instance = cls(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            # Always load checkpoint weights. Subclass to disable.
            load_pretrained=True,
            _skip_optimizations=_skip_optimizations,
        )
        cls.cache_store(instance, cache_key)
        return instance

    @staticmethod
    def get_output_names() -> list[str]:
        """Get output names for the full model."""
        return Llama3Base._get_output_names(NUM_LAYERS)

    @staticmethod
    def get_input_spec(
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        """Get input spec for the model."""
        if llm_config is None:
            llm_config = {
                "num_hidden_layers": NUM_LAYERS,
                "hidden_size": HIDDEN_SIZE,
                "num_key_value_heads": NUM_KEY_VALUE_HEADS,
                "num_attention_heads": NUM_ATTN_HEADS,
            }
        return Llama3Base._get_input_spec(
            num_hidden_layers=llm_config.get("num_hidden_layers", NUM_LAYERS),
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config.get("hidden_size", HIDDEN_SIZE),
            num_key_value_heads=llm_config.get(
                "num_key_value_heads", NUM_KEY_VALUE_HEADS
            ),
            num_attention_heads=llm_config.get("num_attention_heads", NUM_ATTN_HEADS),
            llm_io_type=llm_io_type,
        )

    def get_full_onnx_bundle(self, temp_path: Path) -> ONNXBundle:
        """Export full ONNX from PyTorch with dynamic shapes."""
        print_with_box(
            [
                "Exporting ONNX model with dynamic shapes.",
                "This may take around 30 minutes.",
            ]
        )
        onnx_dir = temp_path / "full_dynamic"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "model.onnx"
        get_onnx_model(
            fp_model=self,
            context_length=self.context_length,
            sequence_length=self.sequence_length,
            path=str(onnx_path),
            return_model=False,
            llm_io_type=self.llm_io_type,
            use_dynamic_shapes=True,
            quiet=True,
        )
        return ONNXBundle.from_bundle_path(onnx_dir, "model")


# ---------------------------------------------------------------------------
# Llama3_2_1B_QuantizablePreSplit - Quantizable PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Llama3_2_1B_QuantizablePreSplit(  # type: ignore[misc]
    LlamaQuantizablePreSplitMixin[Llama3_2_1B_PreSplit], Llama3Base_AIMETOnnx
):
    """
    Quantizable PreSplit for Llama 3.2 1B.

    Manages QuantSim and calibration. Uses class-level cache keyed by
    checkpoint to reuse instances across calls with different
    sequence/context lengths (dynamic shapes). When a different checkpoint
    is requested, the old instance is evicted and freed.
    """

    FPModel = Llama3_2_1B_PreSplit  # type: ignore[assignment]
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT

    # QuantizablePreSplitMixin config
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    supported_precisions = SUPPORTED_PRECISIONS
    default_precision = DEFAULT_PRECISION

    @staticmethod
    def get_output_names() -> list[str]:
        """Get output names for the full model."""
        return Llama3Base._get_output_names(NUM_LAYERS)

    @classmethod
    def get_input_spec(
        cls,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        """Get input spec for the model."""
        return cls.FPModel.get_input_spec(
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=llm_io_type,
        )


# ---------------------------------------------------------------------------
# Unified Part Base & Concrete Parts
# ---------------------------------------------------------------------------


class Llama3_2_1B_PartBase(MultiGraphBaseModel):
    """
    Unified Part base: handles both FP and Quantizable modes based on precision.

    Each Part represents one split of the ONNX model for deployment.
    When precision is float, uses the FP PreSplit (ONNX ModelProto inference).
    When precision is quantized, uses the Quantizable PreSplit (ONNXBundle + encodings).
    """

    part_id: int = 0  # Override in subclasses (1-indexed)

    def __init__(
        self,
        presplit: Llama3_2_1B_PreSplit | Llama3_2_1B_QuantizablePreSplit,
        precision: Precision = DEFAULT_PRECISION,
    ) -> None:
        super().__init__()
        self._presplit = presplit
        self._precision = precision
        self._quant_sim: QuantizationSimModel | None = None
        self._fp_session: onnxruntime.InferenceSession | None = None

    @property
    def _is_quantized(self) -> bool:
        return self._precision != Precision.float

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        precision: Precision = DEFAULT_PRECISION,
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Create Part by getting or creating the appropriate PreSplit (cached)."""
        if precision == Precision.float:
            presplit: Llama3_2_1B_PreSplit | Llama3_2_1B_QuantizablePreSplit = (
                Llama3_2_1B_PreSplit.from_pretrained(
                    sequence_length=sequence_length,
                    context_length=context_length,
                    host_device=host_device,
                )
            )
        else:
            presplit = Llama3_2_1B_QuantizablePreSplit.from_pretrained(
                sequence_length=sequence_length,
                context_length=context_length,
                precision=precision,
                checkpoint=checkpoint,
                host_device=host_device,
                _skip_quantsim_creation=_skip_quantsim_creation,
            )
        return cls(presplit, precision=precision)

    @staticmethod
    def get_default_input_spec(
        llm_config: dict | None = None,
        sequence_length: int = 1,  # Default to token generator mode
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        """Get default input spec for the full model (class-level convenience)."""
        return Llama3_2_1B_PreSplit.get_input_spec(
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=llm_io_type,
        )

    def _get_input_spec_for_instance(self) -> InputSpec:
        """Get input spec for this specific Part instance.

        Part 1 (embedding): only input_ids.
        Part 2+: intermediate hidden state from previous part,
                 attention_mask, position embeddings, and only this
                 part's KV cache layers.

        Names are read from the actual split ONNX model at runtime.
        """
        if self.part_id == 1:
            # Embedding split: only input_ids
            return {"input_ids": ((1, self._presplit.sequence_length), "int32")}

        seq_len = self._presplit.sequence_length
        context_length = self._presplit.context_length
        head_dim = HIDDEN_SIZE // NUM_ATTN_HEADS
        embed_dim = head_dim // 2
        kv_seq_len = context_length - seq_len

        # Read actual input names from the split ONNX model
        onnx_input_names = self._get_onnx_input_names()

        spec: InputSpec = {}

        for name in onnx_input_names:
            if "past_key" in name:
                spec[name] = (
                    (NUM_KEY_VALUE_HEADS, 1, head_dim, kv_seq_len),
                    "float32",
                )
            elif "past_value" in name:
                spec[name] = (
                    (NUM_KEY_VALUE_HEADS, 1, kv_seq_len, head_dim),
                    "float32",
                )
            elif name == "attention_mask":
                spec[name] = (
                    (1, 1, seq_len, context_length),
                    "float32",
                )
            elif "position_ids_cos" in name or "position_ids_sin" in name:
                spec[name] = (
                    (1, 1, seq_len, embed_dim),
                    "float32",
                )
            else:
                # Intermediate hidden state from previous part
                # (found by process of elimination)
                spec[name] = (
                    (1, seq_len, HIDDEN_SIZE),
                    "float32",
                )

        return spec

    def get_output_names(self) -> list[str]:
        """Get output names for this specific Part instance.

        Names are read from the actual split ONNX model at runtime.
        """
        return [
            name.replace("/", "_").replace(".", "_")
            for name in self._get_onnx_output_names()
        ]

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """Source model format for AI Hub Workbench."""
        return SourceModelFormat.ONNX

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        """Get sample inputs for this specific part only.

        Uses actual ONNX input names read from the split model at runtime.
        """
        full_inputs = self._presplit._sample_inputs_impl()

        if self.part_id == 1:
            # Embedding split: only input_ids
            return {"input_ids": full_inputs["input_ids"]}

        # Parts 2+: read actual input names from ONNX and match them
        result: SampleInputsType = {}
        seq_len = self._presplit.sequence_length
        onnx_input_names = self._get_onnx_input_names()

        for name in onnx_input_names:
            if name in full_inputs:
                result[name] = full_inputs[name]
            else:
                # Intermediate hidden state (not in full model inputs)
                # found by process of elimination
                result[name] = [np.zeros((1, seq_len, HIDDEN_SIZE), dtype=np.float32)]

        return result

    # -------------------------------------------------------------------
    # Methods that branch on self._is_quantized
    # -------------------------------------------------------------------

    def _get_onnx_input_names(self) -> list[str]:
        """Read actual input names from split ONNX model."""
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [i.name for i in onnx_model.graph.input]

    def _get_onnx_output_names(self) -> list[str]:
        """Read actual output names from split ONNX model."""
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [o.name for o in onnx_model.graph.output]

    def _get_onnx_bundle(self) -> ONNXBundle:
        """Get ONNXBundle for this Part (works for both FP and quantized)."""
        return self._presplit.convert_to_onnx_and_split(part_id=self.part_id)

    def _get_quant_sim(self) -> QuantizationSimModel:
        """Get or create QuantSim for this specific part from its ONNXBundle."""
        if self._quant_sim is not None:
            return self._quant_sim

        onnx_bundle = self._get_onnx_bundle()

        # Load ONNX model
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=True
        )

        # Dynamo export (opset 18) produces IR version 11, but ORT 1.x
        # only supports up to 10.  Clamp to keep QuantSim compatible.
        onnx_model.ir_version = min(onnx_model.ir_version, 10)

        assert isinstance(self._presplit, Llama3_2_1B_QuantizablePreSplit)
        _hd = self._presplit.host_device
        host_device = _hd if isinstance(_hd, torch.device) else torch.device("cpu")
        providers = self._presplit.get_ort_providers(host_device)

        # Use shared construction + activation config. We skip the
        # full-model _configure_quant_sim (lm_head, KV tying) since
        # those heuristics misfire on split models.
        self._quant_sim = LLM_AIMETOnnx._build_quantsim(onnx_model, providers)
        LLM_AIMETOnnx._apply_precision_activations(self._quant_sim, self._precision)

        # Load encodings if available
        if onnx_bundle.aimet_encodings_path is not None:
            load_encodings_to_sim(
                self._quant_sim, str(onnx_bundle.aimet_encodings_path), strict=False
            )

        return self._quant_sim

    def _get_fp_session(self) -> onnxruntime.InferenceSession:
        """Get or create an ORT session for FP inference (cached)."""
        if self._fp_session is not None:
            return self._fp_session

        onnx_bundle = self._get_onnx_bundle()

        providers: list[str] = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        # Dynamo export (opset 18) produces IR version 11, but ORT 1.x
        # only supports up to 10.  Patch the file in-place (graph-only
        # load keeps external weight references intact).
        onnx_path = str(onnx_bundle.onnx_graph_path)
        onnx_model = onnx.load(onnx_path, load_external_data=False)
        if onnx_model.ir_version > 10:
            onnx_model.ir_version = 10
            onnx.save(onnx_model, onnx_path)

        self._fp_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        return self._fp_session

    def forward(
        self, *args: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor | Collection[torch.Tensor]:
        """Forward pass for this Part (FP or quantized based on precision)."""
        if self._is_quantized:
            quant_sim = self._get_quant_sim()
            return mock_torch_onnx_inference(quant_sim.session, *args, **kwargs)
        session = self._get_fp_session()
        return mock_torch_onnx_inference(session, *args, **kwargs)

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str:
        """Export ONNX model for this Part."""
        model_name = self.__class__.__name__

        ext = ".aimet" if self._is_quantized else ".onnx"
        out_dir = Path(output_path) / f"{model_name}{ext}"
        if (out_dir / f"{model_name}.onnx").exists():
            return str(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_bundle = self._get_onnx_bundle()
        onnx_bundle.move(
            dst_folder=str(out_dir),
            dst_model_name=model_name,
            copy=True,
        )
        return str(out_dir)

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> dict[str, str]:
        """Get profile options keyed by graph name.

        For quantized models, delegates to the PreSplit for extra options.
        """
        if self._is_quantized:
            out: dict[str, str] = {}
            for graph_name in self.get_input_spec():
                out[graph_name] = self._presplit.get_hub_profile_options(
                    target_runtime=target_runtime,
                    other_profile_options=other_profile_options,
                    context_graph_name=graph_name,
                )
            return out
        return MultiGraphBaseModel.get_hub_profile_options(
            self,
            target_runtime=target_runtime,
            other_profile_options=other_profile_options,
        )

    def get_input_spec(self) -> dict[str, InputSpec]:
        seq_len = self._presplit.sequence_length
        ctx_len = self._presplit.context_length
        inst = "token" if seq_len == 1 else "prompt"
        graph_name = f"{inst}_ar{seq_len}_cl{ctx_len}_{self.part_id}_of_{NUM_SPLITS}"
        return {graph_name: self._get_input_spec_for_instance()}


class Llama3_2_1B_Part1_Of_3(Llama3_2_1B_PartBase):
    """Part 1: Embedding + first layers."""

    part_id = 1


class Llama3_2_1B_Part2_Of_3(Llama3_2_1B_PartBase):
    """Part 2: Middle layers."""

    part_id = 2


class Llama3_2_1B_Part3_Of_3(Llama3_2_1B_PartBase):
    """Part 3: Final layers + LM head."""

    part_id = 3


class _Llama3SplitForwardMixin(SplitForwardMixin):
    """Llama-specific split-forward: returns the 3 concrete Part classes."""

    def get_split_part_classes(self) -> list[type]:
        return [
            Llama3_2_1B_Part1_Of_3,
            Llama3_2_1B_Part2_Of_3,
            Llama3_2_1B_Part3_Of_3,
        ]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    _Llama3SplitForwardMixin, Llama3_2_1B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""


class FPSplitModelWrapper(_Llama3SplitForwardMixin, Llama3_2_1B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""


# ---------------------------------------------------------------------------
# Collection Class
# ---------------------------------------------------------------------------


@CollectionModel.add_component(Llama3_2_1B_Part1_Of_3, "llama3_2_1b_part1_of_3")
@CollectionModel.add_component(Llama3_2_1B_Part2_Of_3, "llama3_2_1b_part2_of_3")
@CollectionModel.add_component(Llama3_2_1B_Part3_Of_3, "llama3_2_1b_part3_of_3")
class Llama3_2_1B_Collection(MultiGraphPretrainedCollectionModel):
    """
    Unified Collection with 3 Parts for Llama 3.2 1B.

    Supports both FP and Quantizable modes based on precision parameter.
    All Parts share the same PreSplit via class-level cache for memory efficiency.
    """

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        precision: Precision = DEFAULT_PRECISION,
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = True,
        **kwargs: Any,
    ) -> Self:
        """
        Create Collection with all 3 Parts.

        Parameters
        ----------
        checkpoint
            Path to checkpoint with ONNX + encodings, or ``"DEFAULT"``
            to create from HuggingFace.
        sequence_length
            Sequence length for the model.
        context_length
            Context length for the model.
        precision
            Precision mode. Use Precision.float for FP, or a quantized
            precision (e.g., Precision.w4a16) for quantized mode.
        host_device
            Device for computation.
        _skip_quantsim_creation
            Skip QuantSim creation (for testing).
        **kwargs
            Additional keyword arguments passed to parent.

        Returns
        -------
        Self
            The Collection with all 3 Parts.
        """
        part_kwargs = dict(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            precision=precision,
            host_device=host_device,
            _skip_quantsim_creation=_skip_quantsim_creation,
        )
        parts = [
            part_cls.from_pretrained(**part_kwargs)
            for part_cls in cls.component_classes.values()
        ]
        return cls(*parts)

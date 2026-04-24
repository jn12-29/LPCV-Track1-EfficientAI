# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

# isort: off
try:
    from qai_hub_models.utils.quantization_aimet_onnx import AIMETOnnxQuantizableMixin
    from aimet_onnx.common.defs import QuantizationDataType
    from aimet_onnx.common.utils import AimetLogger
except (ImportError, ModuleNotFoundError):
    print(
        "Some quantized models require the AIMET-ONNX package, which is only supported on Linux. "
        "Quantized model can be exported without this requirement."
    )
# isort: on

import contextlib
import functools
import gc
import glob
import json
import logging
import math
import os
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, NoReturn, TypeVar, cast

import numpy as np
import onnx
import qai_hub as hub
import torch
from onnx.external_data_helper import load_external_data_for_model
from packaging.version import Version
from qai_hub.client import DatasetEntries, Device
from torch.utils.data import DataLoader
from tqdm import tqdm

from qai_hub_models.configs.model_metadata import (
    GenieChatTemplate,
    GenieMetadata,
    GeniePipeline,
    GenieSampleInput,
    ModelFileMetadata,
    ModelMetadata,
)
from qai_hub_models.configs.tensor_spec import (
    QuantizationParameters,
    TensorSpec,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.onnx.torch_wrapper import (
    OnnxModelTorchWrapper,
    _verify_onnxruntime_qnn_installed,
)
from qai_hub_models.utils.runtime_torch_wrapper import ModelIODetails
from qai_hub_models.utils.version_helpers import ensure_supported_version

try:
    from transformers import AutoConfig, PretrainedConfig
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    from transformers.models.llama import LlamaConfig
except ImportError:

    class DynamicCache:  # type: ignore[no-redef, unused-ignore]
        pass


from typing_extensions import Self

from qai_hub_models.models._shared.llm.common import (
    TORCH_DYNAMIC_SHAPE_MIN_VERSION,
    LLMIOType,
    cleanup,
)
from qai_hub_models.models._shared.llm.sha_dynamic_kvcache import (
    SHADynamicCacheNewValueOnly,
)
from qai_hub_models.models._shared.llm.split_onnx_utils.utils import split_onnx
from qai_hub_models.models.common import SampleInputsType, SourceModelFormat
from qai_hub_models.utils.aimet.config_loader import get_aimet_config_path
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel, Precision, TargetRuntime
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.llm_helpers import (
    create_genie_config,
    save_htp_config_for_genie_bundle,
)
from qai_hub_models.utils.onnx.helpers import (
    ONNXBundle,
    generate_wrapper_onnx_file,
    safe_torch_onnx_export,
)
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries
from qai_hub_models.utils.system_info import has_recommended_memory

AIMET_ONNX_INSTALLED = False
try:
    import aimet_onnx.common.quantsim as qs
    from aimet_onnx import quantsim
    from aimet_onnx.quantsim import (
        QuantizationSimModel,
        QuantScheme,
        load_encodings_to_sim,
    )

    from qai_hub_models.models._shared.llm._utils import (
        _apply_int8_kv_cache_tying_and_lm_head,
        _get_kv_io_map,
        _get_lm_head_weights,
        _set_lm_head_to_8b,
    )
    from qai_hub_models.utils.quantization_aimet_onnx import (
        ensure_min_aimet_onnx_version,
    )

    AIMET_ONNX_INSTALLED = True
except (ImportError, ModuleNotFoundError):
    print(
        "Quantized models require the AIMET-ONNX package, which is only supported on Linux. "
        "Install qai-hub-models on a Linux machine to use quantized models."
    )

if TYPE_CHECKING:
    from qai_hub_models.evaluators.base_evaluators import BaseEvaluator

MIN_TRANSFORMER_VERSION = "4.45.0"
MIN_AIMET_ONNX_VERSION = "2.8.0"
# isort: off

DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_CONTEXT_LENGTH = 4096

DEFAULT_EXPORT_SEQUENCE_LENGTHS: list[int] = [DEFAULT_SEQUENCE_LENGTH, 1]
DEFAULT_EXPORT_CONTEXT_LENGTHS: list[int] = [DEFAULT_CONTEXT_LENGTH]

DEFAULT_CALIBRATION_SEQ_LEN = 2048

if AIMET_ONNX_INSTALLED:
    ensure_min_aimet_onnx_version(MIN_AIMET_ONNX_VERSION)


try:
    from transformers import (
        AutoTokenizer,
        PreTrainedTokenizerBase,
    )

    # TODO: 10761 remove transformer version check once AIMET
    # transformer restriction is uplifted.
    ensure_supported_version("transformers", min_version=MIN_TRANSFORMER_VERSION)
except ImportError:
    pass


def determine_precision_from_checkpoint(checkpoint: str) -> Precision | None:
    if checkpoint.startswith("DEFAULT_"):
        return Precision.parse(checkpoint[len("DEFAULT_") :].lower())
    return None


def sample_input(
    input_spec: InputSpec,
    input_prompt_processed: str,
    context_length: int,
    sequence_length: int,
    tokenizer: PreTrainedTokenizerBase,
    llm_config: PretrainedConfig,
    embedding: Embedding,
) -> dict[str, list[np.ndarray]]:
    input_tokens = tokenizer(
        input_prompt_processed,
        return_tensors="pt",
        padding="max_length",
        max_length=context_length,
    )
    num_tokens = int(
        min(
            torch.sum(input_tokens["attention_mask"]).item(),
            sequence_length,
        )
    )
    input_ids = input_tokens["input_ids"].type(torch.int32)[:, -sequence_length:]

    padding_size = sequence_length - num_tokens
    position_ids_list = [0] * (padding_size) + list(
        range(sequence_length - padding_size)
    )
    position_ids = (
        torch.Tensor(position_ids_list).type(torch.long).reshape(1, sequence_length)
    )
    position_ids_cos, position_ids_sin = embedding.get_embedding(position_ids)
    attention_mask = torch.zeros((1, context_length))
    attention_mask[:, -num_tokens:] = 1.0

    attention_mask_converter = AttentionMaskConverter(True)
    cm_attention_mask = attention_mask_converter.to_4d(
        attention_mask,
        query_length=sequence_length,
        key_value_length=context_length,
        dtype=torch.float32,
    )
    cm_attention_masks = cm_attention_mask.clip(-50, 0)

    if "input_ids" in input_spec:
        input_dict = {
            "input_ids": [input_ids.detach().cpu().numpy()],
        }
    else:
        inputs_embeds = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], llm_config.vocab_size)
        )
        input_dict = {"inputs_embeds": [inputs_embeds.detach().numpy()]}

    input_dict = input_dict | {
        "attention_mask": [cm_attention_masks.detach().numpy()],
        "position_ids_cos": [position_ids_cos.detach().numpy()],
        "position_ids_sin": [position_ids_sin.detach().numpy()],
    }

    # Populate the rest with zeros (KV cache input)
    for k, (shape, _) in input_spec.items():
        if k.startswith("past_"):
            input_dict[k] = [np.zeros(shape, dtype=np.float32)]

    return input_dict


def get_tokenizer(
    model_ckpt: str | os.PathLike | Path | None,
) -> PreTrainedTokenizerBase:
    """Tokenizer to use for LLMs"""
    assert model_ckpt is not None
    print()
    print(f"Loading tokenizer from {model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, is_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    return tokenizer


def get_llm_config(model_ckpt: str | os.PathLike | Path | None) -> LlamaConfig:
    """Construct and return a HuggingFace LLM config."""
    assert model_ckpt is not None
    print()
    print(f"Loading model config from {model_ckpt}")
    llm_config = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
    # If it's a multi-modal model, extract only the language model
    if hasattr(llm_config, "text_config"):
        llm_config = llm_config.text_config
    llm_config._attn_implementation = "eager"
    llm_config._attn_implementation_internal = "eager"

    # Force use_cache=true for all LLMs
    llm_config.use_cache = True

    return llm_config


def get_onnx_model(
    fp_model: LLMBase,
    context_length: int,
    sequence_length: int,
    path: str,
    return_model: bool = False,
    llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    use_dynamic_shapes: bool = False,
    quiet: bool = False,
) -> onnx.ModelProto | None:
    if use_dynamic_shapes:
        ensure_supported_version("torch", min_version=TORCH_DYNAMIC_SHAPE_MIN_VERSION)
    else:
        ensure_supported_version("torch", min_version="2.4.1", below_version="2.9")

    # Create the checkpoint directory if it does not exist.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # The GPU memory of the model passed into torch.onnx.export cannot
    # subsequently be released due to what looks like a PyTorch bug. We export
    # on the CPU as a workaround.
    assert fp_model.model is not None
    assert isinstance(fp_model.model, torch.nn.Module)
    # Note: torch.nn.Module.device does not exist according to PyTorch
    # documentation and mypy.
    old_device: torch.device = next(iter(fp_model.model.parameters())).device
    device = torch.device("cpu")
    fp_model.to(device)

    input_specs = fp_model.get_input_spec(
        llm_config=fp_model.llm_config.to_dict(),
        context_length=context_length,
        sequence_length=sequence_length,
        llm_io_type=llm_io_type,
    )
    if not quiet:
        print()
        if use_dynamic_shapes:
            print(
                "Exporting ONNX model with dynamic sequence length and dynamic context length."
            )
        else:
            print(
                f"Exporting ONNX model with sequence length {sequence_length} and context length {context_length}. This could take around 10 minutes."
            )

    example_input = [
        torch.zeros(
            input_specs[name][0], dtype=getattr(torch, input_specs[name][1])
        ).to(device)
        for name in input_specs
    ]

    data_full_path = os.path.join(os.path.dirname(path), "model.data")
    data_backup_full_path = os.path.join(os.path.dirname(path), ".model.data")

    # If the data file already exists, we need to preserve it. This may have
    # quantization-modified weights (from algorithms like AdaScale), so if we
    # re-export the model from PyTorch, we will revert the weights. To prevent
    # this, we rename the original data file so it is not overwritten and
    # restore it after the export.
    already_has_data = os.path.isfile(data_full_path)
    if already_has_data:
        os.rename(data_full_path, data_backup_full_path)

    # Build dynamic_shapes if requested (for dynamo export)
    dynamic_shapes = None
    if use_dynamic_shapes:
        from torch.export import Dim

        # Define dimension constraints
        # seq_len, kv_seq_len, and ctx_len are all dynamic.
        # ctx_len = seq_len + kv_seq_len, but we declare them independently
        # and let the tracer infer the relationship from the model computation.
        seq_len = Dim.DYNAMIC  # type: ignore[attr-defined, unused-ignore]
        kv_seq_len = Dim.DYNAMIC  # type: ignore[attr-defined, unused-ignore]
        ctx_len = Dim.DYNAMIC  # type: ignore[attr-defined, unused-ignore]

        # The model's forward signature is: forward(self, input_tokens, attention_mask, *args)
        # So torch.export sees the input structure as: (input_tokens, attention_mask, args_tuple)
        # We need dynamic_shapes to match this 3-element structure
        input_names = list(input_specs.keys())

        # Build dynamic shapes for the *args portion (everything after input_tokens and attention_mask)
        args_dynamic_shapes: list[dict[int, Any] | None] = []
        for name in input_names[2:]:  # Skip input_ids/input_embeds and attention_mask
            if name == "position_ids":
                # Shape: (1, seq_len)
                args_dynamic_shapes.append({1: seq_len})
            elif name in ("position_ids_cos", "position_ids_sin"):
                # Shape: (1, 1, seq_len, embed_dim)
                args_dynamic_shapes.append({2: seq_len})
            elif name.startswith("past_key_"):
                # past_key shape: (num_kv_heads, 1, embed_dim*2, kv_seq_len)
                args_dynamic_shapes.append({3: kv_seq_len})
            elif name.startswith("past_value_"):
                # past_value shape: (num_kv_heads, 1, kv_seq_len, embed_dim*2)
                args_dynamic_shapes.append({2: kv_seq_len})
            else:
                # No dynamic dims for other inputs
                args_dynamic_shapes.append(None)

        # Structure: (input_tokens_shape, attention_mask_shape, args_shapes_tuple)
        dynamic_shapes = (
            {1: seq_len},  # input_tokens (input_ids or input_embeds): (1, seq_len, ...)
            {2: seq_len, 3: ctx_len},  # attention_mask: (1, 1, seq_len, context_length)
            tuple(args_dynamic_shapes),  # *args
        )

    try:
        # Names were changed in 2.9, which could ruin cached .onnx files.
        # This is no longer an issue with 2.10+ dynamo export.
        extra = {}
        if use_dynamic_shapes:
            extra = {
                "opset_version": 18,
                "dynamo": True,
                "optimize": True,
                "dynamic_shapes": dynamic_shapes,
            }
        else:
            extra = {
                "opset_version": 17,
                "dynamo": False,
            }

        with torch.no_grad():
            safe_torch_onnx_export(
                fp_model,
                tuple(example_input),
                path,
                input_names=list(input_specs.keys()),
                output_names=fp_model._get_output_names(
                    fp_model.llm_config.num_hidden_layers
                ),
                **extra,
            )

        fp_model.to(old_device)

        onnx_model = onnx.load(path)
        # Clean up multiple weights files
        for file in glob.glob(os.path.join(os.path.dirname(path), "*.weight")):
            os.remove(file)
        for file in glob.glob(os.path.join(os.path.dirname(path), "onnx__*")):
            os.remove(file)

        onnx.save_model(
            onnx_model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )

    finally:
        if already_has_data:
            os.replace(data_backup_full_path, data_full_path)

    load_external_data_for_model(onnx_model, os.path.dirname(path))

    if not return_model:
        del onnx_model
        gc.collect()
    return onnx_model if return_model else None


def _get_evaluator(
    task: str,
    context_length: int,
    sequence_length: int,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> BaseEvaluator:
    from qai_hub_models.evaluators.mmlu_evaluator import MMLUEvaluator
    from qai_hub_models.evaluators.ppl_evaluator import PerplexityEvaluator
    from qai_hub_models.evaluators.kldiv_evaluator import KLDivEvaluator

    if "wikitext" in task:
        return PerplexityEvaluator(context_length, device, tokenizer)
    if "tricky_llm_prompts" in task:
        return KLDivEvaluator(context_length, device, tokenizer, verbose=True)
    return MMLUEvaluator(context_length, device, tokenizer)


class Embedding(ABC):
    def __init__(  # noqa: B027
        self,
        head_dim: int = 128,
        max_length: int = 2048,
        config: Any = None,
    ) -> None:
        pass

    @abstractmethod
    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class PositionProcessorBase(torch.nn.Module):
    """Prepares positions (Embedding and attention mask preparation); used by ORT GenAI."""

    def __init__(self, context_length: int, config: PretrainedConfig) -> None:
        super().__init__()

    def forward(
        self, attention_mask_before_processor: torch.Tensor, position_ids: torch.Tensor
    ) -> NoReturn:
        raise NotImplementedError("Must be implemented by subclass")


class LLMConfigEditor:
    def edit_llm_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
        return llm_config  # no change by default


FPModelT = TypeVar("FPModelT", bound="LLMBase")


class SingleSlotCacheMixin:
    """Mixin that maintains a single class-level cached instance keyed by a string key.

    On cache hit (same key), returns the existing instance.
    On cache miss (different key), evicts and frees the old instance
    before the caller creates a new one.

    Subclasses must provide a ``free_memory()`` method on instances.
    """

    _cached_instance: Any = None
    _cached_key: str | None = None

    @classmethod
    def cache_lookup(cls, cache_key: str) -> Any:
        """Return cached instance if key matches, or None on miss.

        On miss with an existing cached instance, the old one is evicted.
        """
        if cls._cached_instance is not None and cls._cached_key == cache_key:
            return cls._cached_instance

        # Different key -- evict old instance
        if cls._cached_instance is not None:
            cls._cached_instance.free_memory()
            cls._cached_instance = None
            cls._cached_key = None

        return None

    @classmethod
    def cache_store(cls, instance: Any, cache_key: str) -> None:
        """Evict any existing cached instance, then store the new one."""
        if cls._cached_instance is not None and cls._cached_instance is not instance:
            cls.clear_cache()
        cls._cached_instance = instance
        cls._cached_key = cache_key

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached instance and free memory."""
        if cls._cached_instance is not None:
            cls._cached_instance.free_memory()
            cls._cached_instance = None
            cls._cached_key = None
        cleanup()


class PreSplitOnnxMixin:
    """Mixin that manages ONNX splitting for PreSplit models.

    Provides a cached convert_to_onnx_and_split() workflow using
    the template method pattern. Subclasses implement
    get_full_onnx_bundle() to supply the full (unsplit) ONNXBundle.

    Class-level configuration (override in subclasses):
        split_model_name: str  -- basename for split files
        num_splits: int        -- number of ONNX splits
        num_layers_per_split: int -- layers per split
    """

    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.onnx_splits: dict[int, ONNXBundle] = {}
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def __del__(self) -> None:
        """Ensure temp directory is cleaned up on deletion."""
        if hasattr(self, "_temp_dir") and self._temp_dir is not None:
            with contextlib.suppress(Exception):
                self._temp_dir.cleanup()

    def convert_to_onnx_and_split(self, part_id: int = 1) -> ONNXBundle:
        """Convert to ONNX and split into parts. Results are cached.

        Parameters
        ----------
        part_id
            Part ID (1-indexed) to return.

        Returns
        -------
        ONNXBundle
            The requested split as an ONNXBundle.
        """
        if part_id in self.onnx_splits:
            return self.onnx_splits[part_id]

        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self._temp_dir.name)

        full_bundle = self.get_full_onnx_bundle(temp_path)

        split_output_dir = temp_path / "splits_dynamic"
        split_output_dir.mkdir(parents=True, exist_ok=True)

        split_bundles = split_onnx(
            onnxfile=full_bundle,
            modelname=self.split_model_name,
            num_splits=self.num_splits,
            num_layers_per_split=self.num_layers_per_split,
            output_dir=str(split_output_dir),
            split_embedding=True,
        )

        for i, bundle in enumerate(split_bundles):
            self.onnx_splits[i + 1] = bundle

        return self.onnx_splits[part_id]

    def get_full_onnx_bundle(self, temp_path: Path) -> ONNXBundle:
        """Return the full (unsplit) ONNXBundle.

        Default implementation copies model_dynamic.onnx, model.data, and
        model.encodings from the checkpoint directory into a temp bundle.
        This works for quantized PreSplit models whose checkpoint already
        contains these files (checkpoint must be resolved before calling).

        FP PreSplit models should override this to export from PyTorch
        via get_onnx_model().
        """
        bundle_dir = temp_path / "bundle_dynamic"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = getattr(self, "checkpoint", None)
        if checkpoint is None or (
            isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT")
        ):
            raise NotImplementedError(
                f"{type(self).__name__}: checkpoint must be resolved to a "
                "local path before calling get_full_onnx_bundle(). "
                "Got: {checkpoint!r}"
            )
        checkpoint_path = Path(checkpoint)

        src_onnx = checkpoint_path / "model_dynamic.onnx"
        shutil.copy(src_onnx, bundle_dir / "model.onnx")

        src_data = checkpoint_path / "model.data"
        if src_data.exists():
            shutil.copy(src_data, bundle_dir / "model.data")

        src_enc = checkpoint_path / "model.encodings"
        if src_enc.exists():
            shutil.copy(src_enc, bundle_dir / "model.encodings")

        return ONNXBundle.from_bundle_path(bundle_dir, "model")

    def free_memory(self) -> None:
        """Free memory by clearing caches and releasing resources."""
        # Clear torch model (FP path)
        model = getattr(self, "model", None)
        if model is not None:
            model.to("cpu")
            del model
            setattr(self, "model", None)  # noqa: B010

        # Clear QuantSim (quantized path)
        quant_sim = getattr(self, "quant_sim", None)
        if quant_sim is not None:
            del quant_sim
            setattr(self, "quant_sim", None)  # noqa: B010

        # Clear ONNX split cache and temp directory
        self.onnx_splits.clear()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

        cleanup()


class QuantizablePreSplitMixin(
    SingleSlotCacheMixin, PreSplitOnnxMixin, Generic[FPModelT]
):
    """Mixin for quantized PreSplit models with cached from_pretrained.

    Combines SingleSlotCacheMixin (class-level cache) and PreSplitOnnxMixin
    (ONNX splitting) with generic DEFAULT-checkpoint resolution and
    save_calibrated_checkpoint for dynamic shapes.

    Must be mixed with a subclass of LLM_AIMETOnnx which provides
    FPModel, create_onnx_models, save_tokenizer_and_config,
    from_pretrained, and save_calibrated_checkpoint.

    Subclasses must set these class attributes:
        model_id: str               -- e.g. "llama_v3_2_1b_instruct2"
        model_asset_version: int    -- asset store version
        default_checkpoint: dict[Precision, str]  -- precision -> asset key
        supported_precisions: list[Precision]
        default_precision: Precision

    Override ``resolve_default_checkpoint`` to change how DEFAULT
    checkpoints are fetched. The default downloads a full .zip archive.
    """

    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict[Precision, str] = {}
    supported_precisions: list[Precision] = []
    default_precision: Precision = Precision.w4

    # Declared for type-checking; provided by LLM_AIMETOnnx at runtime.
    FPModel: type[FPModelT]
    sequence_length: int
    context_length: int

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path = "DEFAULT",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, checkpoint=checkpoint, **kwargs)
        self.precision: Precision = kwargs.get("precision", self.default_precision)

    @classmethod
    def resolve_default_checkpoint(
        cls,
        precision: Precision,
        sequence_length: int,
        context_length: int,
        host_device: torch.device,
        fp_model: FPModelT | None,
    ) -> tuple[str, FPModelT | None]:
        """Resolve a DEFAULT checkpoint string to a local directory path.

        Downloads the checkpoint assets and ensures the checkpoint directory
        contains the ONNX model, external data, and tokenizer/config files
        needed by LLM_AIMETOnnx.from_pretrained.

        The default implementation downloads a full .zip archive that
        contains ONNX + encodings + data. If ``fp_model`` is provided,
        it additionally exports ONNX models and saves tokenizer/config.

        Override in subclasses for different fetching strategies (e.g.
        downloading only encodings and exporting ONNX from a torch model).

        Parameters
        ----------
        precision
            Quantization precision (already validated).
        sequence_length
            Sequence length for the model.
        context_length
            Context length for the model.
        host_device
            Device for computation.
        fp_model
            Optional FP model passed by the evaluate framework.

        Returns
        -------
        tuple[str, FPModelT | None]
            (resolved_checkpoint_path, fp_model). The fp_model may be
            the same as input or a newly created one.
        """
        precision_checkpoint = cls.default_checkpoint[precision]
        checkpoint = str(
            CachedWebModelAsset.from_asset_store(
                cls.model_id,
                cls.model_asset_version,
                precision_checkpoint + ".zip",
            ).fetch(extract=True)
        )
        if fp_model is not None:
            cls.create_onnx_models(  # type: ignore[attr-defined]
                checkpoint=checkpoint,
                fp_model=fp_model,
                context_length=context_length,
                export_sequence_lengths=[sequence_length],
                host_device=host_device,
                llm_io_type=fp_model.llm_io_type,
            )
            cls.save_tokenizer_and_config(  # type: ignore[attr-defined]
                checkpoint=checkpoint, fp_model=fp_model
            )
        return checkpoint, fp_model

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = "DEFAULT",
        host_device: torch.device | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        precision: Precision | None = None,
        fp_model: FPModelT | None = None,
        _skip_quantsim_creation: bool = True,
    ) -> Self:
        """
        Load or return a cached Quantizable PreSplit.

        If a cached instance exists for the same checkpoint, returns it
        with updated sequence/context lengths (dynamic shapes). If a
        different checkpoint is requested, the old instance is freed first.

        Parameters
        ----------
        checkpoint
            Path to checkpoint with ONNX + encodings, or ``"DEFAULT"``
            to fetch from the asset store.
        host_device
            Device for computation.
        sequence_length
            Sequence length for the model.
        context_length
            Context length for the model.
        precision
            Quantization precision. Defaults to ``cls.default_precision``.
        fp_model
            Optional FP model, passed by the evaluate framework to avoid
            creating a duplicate. If not provided and checkpoint starts
            with ``"DEFAULT"``, one is created automatically.
        _skip_quantsim_creation
            Skip QuantSim creation (for testing).

        Returns
        -------
        Self
            The cached or newly created instance.
        """
        if precision is None:
            precision = cls.default_precision

        cache_key = str(checkpoint)
        cached = cls.cache_lookup(cache_key)
        if cached is not None:
            cached.sequence_length = sequence_length
            cached.context_length = context_length
            return cached

        if host_device is None:
            host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT"):
            precision = determine_precision_from_checkpoint(checkpoint) or precision
            if precision not in cls.supported_precisions:
                available = [str(p) for p in cls.supported_precisions]
                raise ValueError(
                    f"This model is not supported for {precision!s} precision. "
                    f"Models are available in following precisions: {','.join(available)}."
                )
            if precision not in cls.default_checkpoint:
                available = [str(p) for p in cls.default_checkpoint]
                raise ValueError(
                    f"No checkpoint is available for this model in {precision!s} precision. "
                    f"Please generate a local quantized checkpoint. "
                    f"Checkpoints are available in the following precisions: {','.join(available)}."
                )
            checkpoint, fp_model = cls.resolve_default_checkpoint(
                precision=precision,
                sequence_length=sequence_length,
                context_length=context_length,
                host_device=host_device,
                fp_model=fp_model,
            )

        instance = super().from_pretrained(  # type: ignore[misc]
            checkpoint=checkpoint,
            host_device=host_device,
            sequence_length=sequence_length,
            context_length=context_length,
            precision=precision,
            fp_model=fp_model,
            _skip_quantsim_creation=_skip_quantsim_creation,
            use_dynamic_shapes=True,
        )

        # Store precision on instance
        instance.precision = precision

        cls.cache_store(instance, cache_key)
        return instance

    def save_calibrated_checkpoint(
        self,
        output_checkpoint: str | os.PathLike | Path,
        fp_model: FPModelT | None = None,
    ) -> None:
        """Save calibrated checkpoint with dynamic shapes.

        PreSplit models always use dynamic shapes. If ``fp_model`` is not
        provided, one is automatically created from ``self.FPModel``.
        """
        if fp_model is None:
            fp_model = self.FPModel.from_pretrained(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
            )
        super().save_calibrated_checkpoint(  # type: ignore[misc]
            output_checkpoint, fp_model, use_dynamic_shapes=True
        )


class SplitForwardMixin:
    """Mixin that overrides forward() to chain inference through split Parts.

    Lazily creates Part instances on first forward call and maps inputs
    from the full model's input spec to each Part's ONNX input names.
    Intermediate hidden states flow from one Part to the next.

    Mix in *before* the PreSplit base so that this forward() wins in MRO::

        class MyWrapper(SplitForwardMixin, SomePreSplit): ...

    Subclasses must override ``_get_split_part_classes()`` to return
    the concrete Part classes.
    """

    _parts: list | None
    _input_names_for_parts: list[list] | None
    _exporting_onnx: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._parts = None
        self._input_names_for_parts = None
        self._exporting_onnx = False
        super().__init__(*args, **kwargs)

    # -- overridable hooks ------------------------------------------------

    def get_split_part_classes(self) -> list[type]:
        """Return the Part classes to instantiate, in order.

        Each class must accept ``(presplit, precision=...)`` and provide
        ``_get_onnx_input_names() -> list[str]``.
        """
        raise NotImplementedError("Subclasses must override get_split_part_classes()")

    def get_split_precision(self) -> Precision:
        """Return the precision to use when creating Parts."""
        return getattr(self, "precision", Precision.float)

    # -- implementation ---------------------------------------------------

    @staticmethod
    def _build_input_name_for_parts(
        parts: list, full_input_names: list[str]
    ) -> list[list]:
        """Build input mapping table for split parts.

        For each Part, maps its ONNX input names to positional indices in the
        full model's args.  Names not found in the full model (intermediate
        hidden states) are tagged ``"prev"`` and sourced from the previous
        Part's first output.
        """
        name_to_idx = {n: i for i, n in enumerate(full_input_names)}
        input_names_for_parts: list[list] = []
        for part in parts:
            onnx_names = part._get_onnx_input_names()
            input_names_for_parts.append(
                [name_to_idx.get(n, "prev") for n in onnx_names]
            )
        return input_names_for_parts

    @staticmethod
    def _split_forward(
        parts: list, input_names_for_parts: list[list], *args: torch.Tensor
    ) -> list[torch.Tensor]:
        """Chain Part1 -> Part2 -> ... with input mapping.

        Works identically for FP and quantized Parts -- each Part's own
        ``forward()`` handles the FP / quantized branching internally.
        """
        prev_output: torch.Tensor | None = None
        kv_outputs: list[torch.Tensor] = []

        for part, input_names in zip(parts, input_names_for_parts, strict=False):
            part_args = [args[s] if s != "prev" else prev_output for s in input_names]
            outputs = part(*part_args)
            outputs = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)

            prev_output = outputs[0]  # hidden state or logits
            kv_outputs.extend(outputs[1:])  # KV cache updates (empty for Part1)

        # prev_output is logits from the final Part
        assert prev_output is not None
        return [prev_output, *kv_outputs]

    def _ensure_parts(self) -> None:
        """Lazily create split Parts on first forward call."""
        if self._parts is not None:
            return
        precision = self.get_split_precision()
        self._parts = [
            cls(self, precision=precision) for cls in self.get_split_part_classes()
        ]
        full_names = list(
            self.get_input_spec(  # type: ignore[attr-defined]
                sequence_length=self.sequence_length,  # type: ignore[attr-defined]
                context_length=self.context_length,  # type: ignore[attr-defined]
            ).keys()
        )
        self._input_names_for_parts = self._build_input_name_for_parts(
            self._parts, full_names
        )

    def convert_to_onnx_and_split(self, part_id: int = 1) -> Any:
        """Wrap ONNX export so forward() falls back to LLMBase.forward.

        During ``get_full_onnx_bundle`` -> ``get_onnx_model`` ->
        ``torch.onnx.export(dynamo=True)``, the tracer calls ``forward()``.
        The flag makes ``forward()`` delegate to the base-class torch forward
        (``LLMBase.forward``) instead of the ORT-based split forward, which
        is not traceable and would recursively re-enter this method.
        """
        self._exporting_onnx = True
        try:
            return super().convert_to_onnx_and_split(part_id=part_id)  # type: ignore[misc]
        finally:
            self._exporting_onnx = False

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self._exporting_onnx or torch.compiler.is_compiling():
            return super().forward(input_tokens, attention_mask, *args)  # type: ignore[misc]
        self._ensure_parts()
        assert self._parts is not None
        assert self._input_names_for_parts is not None
        return self._split_forward(
            self._parts,
            self._input_names_for_parts,
            input_tokens,
            attention_mask,
            *args,
        )


class LLMBase(BaseModel, LLMConfigEditor, ABC):
    # The Hugging Face LLM class (e.g., LlamaForCausalLM)
    LMClass: Any | None = None

    # Embedding subclass
    EmbeddingClass: type[Embedding] | None = None

    # IO signature
    llm_io_type: LLMIOType = LLMIOType.genie_input_ids

    # Minimum recommended memory for exporting (in GB)
    min_memory_recommended: int = 0

    # Default prompts for demos (override in subclasses)
    default_user_prompt: str = "What is gravity?"
    default_system_prompt: str = "You are a helpful AI assistant"

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        """Return chat template tokens as a dict suitable for GenieChatTemplate(**d).

        Subclasses that define explicit chat template tokens should override
        this.  The base implementation returns an empty dict (template is only
        known to the HuggingFace tokenizer).
        """
        return {}

    @classmethod
    def get_input_prompt_with_tags(
        cls,
        user_input_prompt: str | None = None,
        system_context_prompt: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs: Any,
    ) -> str:
        """Format a prompt using the tokenizer's chat template."""
        if tokenizer is None:
            raise ValueError("tokenizer is required for get_input_prompt_with_tags")
        if user_input_prompt is None:
            user_input_prompt = cls.default_user_prompt
        if system_context_prompt is None:
            system_context_prompt = cls.default_system_prompt
        messages = []
        if system_context_prompt:
            messages.append({"role": "system", "content": system_context_prompt})
        messages.append({"role": "user", "content": user_input_prompt})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        assert isinstance(prompt, str)
        return prompt

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path,
        sequence_length: int,
        context_length: int,
        is_token_generator: bool = False,
        load_pretrained: bool = True,
        host_device: torch.device | None = None,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
        _skip_optimizations: list[str] | None = None,
    ) -> None:
        """
        This is an abstract base class of all LLM models.

        Parameters
        ----------
        checkpoint
            Can be local folder or Hugging Face repo name.
        sequence_length
            Input sequence length (in tokens).
        context_length
            Total context length (in tokens).
        is_token_generator
            Whether this is a token generator model.
        load_pretrained
            Load a pre-trained model as opposed to a randomly initialized.
        host_device
            Device to use: GPU/CPU.
        attention_mask_min_clip
            Min clip the attention mask by this value if not None.
        attention_mask_multiplier
            Bake in a multiplier for the attention mask into the network.
            This is useful to conform to Genie's unconfigurable -1000
            "infinity" for FP16 activations, when a network may require an even
            larger (in magnitude) value.
        _skip_optimizations
            Turn off one or more of {sha_attention, rank4_rms_norm}.
        """
        super().__init__()
        self.skip_optimizations = _skip_optimizations
        self.checkpoint = checkpoint

        # Ensure User has recommended memory,
        # otherwise, provide warning to user and recommend to increase swap-space as a work-around.
        has_recommended_memory(self.min_memory_recommended)

        # TODO: Make this into a context manager
        self.monkey_patch(skip_optimizations=self.skip_optimizations)
        llm_config = get_llm_config(self.checkpoint)
        self.llm_config = self.edit_llm_config(llm_config)
        self._verify_ckpt()
        self.tokenizer = get_tokenizer(checkpoint)
        assert self.LMClass is not None
        if load_pretrained:
            model = self.LMClass.from_pretrained(
                self.checkpoint,
                config=self.llm_config,
                ignore_mismatched_sizes=False,
            )
        else:
            model = self.LMClass(self.llm_config)
        model.eval()

        assert self.EmbeddingClass is not None
        self.embedding = self.EmbeddingClass(
            max_length=context_length, config=llm_config
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        for _, module in model.named_modules():
            if hasattr(module, "prepare_conv"):
                module.prepare_conv()
            if hasattr(module, "prepare_sha"):
                module.prepare_sha()

        model.to(host_device)

        self.sequence_length: int = sequence_length
        self.context_length: int = context_length
        self.split_part = 1
        self.is_token_generator = is_token_generator
        self.model = model
        self.attention_mask_min_clip = attention_mask_min_clip
        self.attention_mask_multiplier = attention_mask_multiplier

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int,
        context_length: int,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        raise NotImplementedError

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        pass

    def _verify_ckpt(self) -> None:
        # Override in baseclass to verify compatibility with config
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> Self:
        pass

    @staticmethod
    def _get_output_names(num_hidden_layers: int) -> list[str]:
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

    # Must be defined by transformers generator class
    @property
    def main_input_name(self) -> str:
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "input_embeds"
        return "input_ids"

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self.llm_io_type == LLMIOType.huggingface_input_ids:
            position_ids = args[0]
            past_key_values = args[1:]
        else:
            # contains (position_ids_cos, position_ids_sin)
            position_ids = args[:2]  # type: ignore[assignment, unused-ignore]
            past_key_values = args[2:]

        assert isinstance(self.llm_config.num_key_value_heads, int)
        if self.skip_optimizations and "sha_attention" in self.skip_optimizations:
            kv_cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(
                zip(past_key_values[::2], past_key_values[1::2], strict=False)
            ):
                k_split = [
                    k[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                v_split = [
                    v[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                k = torch.cat(k_split, dim=1).permute(0, 1, 3, 2)
                v = torch.cat(v_split, dim=1)

                kv_cache.update(k, v, layer_idx, {})
        else:
            kv_cache = SHADynamicCacheNewValueOnly()
            for layer_idx, (k, v) in enumerate(
                zip(past_key_values[::2], past_key_values[1::2], strict=False)
            ):
                k_split = [
                    k[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                v_split = [
                    v[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]

                # kv_cache doesn't report supporting lists of tensors, but it seems to work
                kv_cache.update(k_split, v_split, layer_idx, {})

        model_kwargs = {
            self.main_input_name: input_tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": kv_cache,
        }
        out = self.model(**model_kwargs)

        out_cache = out["past_key_values"]
        flat_output_past_key_values = []
        for layer in range(len(out_cache)):
            if self.skip_optimizations and "sha_attention" in self.skip_optimizations:
                if hasattr(out_cache, "key_cache"):
                    keys = out_cache.key_cache[layer]
                    values = out_cache.value_cache[layer]
                elif hasattr(out_cache.layers[layer], "keys"):
                    keys = out_cache.layers[layer].keys
                    values = out_cache.layers[layer].values
                else:
                    keys = out_cache.layers[layer][0]
                    values = out_cache.layers[layer][1]

                k = keys[:, :, -self.sequence_length :, :].permute(1, 0, 3, 2)
                v = values[:, :, -self.sequence_length :, :].permute(1, 0, 2, 3)

            elif hasattr(out_cache, "key_cache"):
                k = torch.cat(out_cache.key_cache[layer], dim=0)
                v = torch.cat(out_cache.value_cache[layer], dim=0)
            elif hasattr(out_cache.layers[layer], "keys"):
                k = torch.cat(out_cache.layers[layer].keys, dim=0)
                v = torch.cat(out_cache.layers[layer].values, dim=0)
            else:
                k = torch.cat(out_cache.layers[layer][0], dim=0)
                v = torch.cat(out_cache.layers[layer][1], dim=0)
            flat_output_past_key_values += [k, v]

        return [out["logits"], *flat_output_past_key_values]

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)  # type: ignore[operator, unused-ignore]

    @staticmethod
    def _get_input_spec(
        num_hidden_layers: int,
        sequence_length: int,
        context_length: int,
        hidden_size: int,
        num_key_value_heads: int,
        num_attention_heads: int,
        head_dim: int | None = None,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        # Use explicit head_dim if provided, otherwise derive from hidden_size
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        embed_dim = head_dim // 2
        input_spec: InputSpec = {}

        if llm_io_type == LLMIOType.genie_input_embeds:
            input_spec |= {
                "input_embeds": (
                    (1, sequence_length, hidden_size),
                    "float32",
                )
            }
        else:
            input_spec |= {"input_ids": ((1, sequence_length), "int32")}

        input_spec |= {
            "attention_mask": (
                (1, 1, sequence_length, context_length),
                "float32",
            ),
        }

        if llm_io_type == LLMIOType.huggingface_input_ids:
            input_spec |= {
                "position_ids": (
                    (1, sequence_length),
                    "int32",
                ),
            }
        else:
            input_spec |= {
                # each cos/sin are applied to a half-sliced copy of the hidden size
                # and then concatenated.
                "position_ids_cos": (
                    (1, 1, sequence_length, embed_dim),
                    "float32",
                ),
                "position_ids_sin": (
                    (1, 1, sequence_length, embed_dim),
                    "float32",
                ),
            }

        # TODO: We could support sequence_length == CONTEXT_LENGTH, but the
        # KV cache input needs to be removed.
        assert sequence_length < context_length, (
            "It is currently not supported to set input sequence length to the same as or longer than context length. There should be no KV cache input at all in such case."
        )

        for layer in range(num_hidden_layers):
            past_k_name = f"past_key_{layer}_in"
            input_spec[past_k_name] = (
                (
                    num_key_value_heads,
                    1,
                    head_dim,
                    context_length - sequence_length,
                ),
                "float32",
            )

            past_v_name = f"past_value_{layer}_in"
            input_spec[past_v_name] = (
                (
                    num_key_value_heads,
                    1,
                    context_length - sequence_length,
                    head_dim,
                ),
                "float32",
            )
        return input_spec

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """Source model format preferred for conversion on AI Hub Workbench."""
        return SourceModelFormat.ONNX

    def get_calibration_data(
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        # No calibration data needed
        return None

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_spec = self.get_input_spec(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
                llm_io_type=self.llm_io_type,
                llm_config=self.llm_config.to_dict(),
            )
        return sample_input(
            input_spec,
            self.get_input_prompt_with_tags(tokenizer=self.tokenizer),
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
            self.embedding,
        )

    def get_evaluator(
        self, task: str = "wikitext", device: torch.device = torch.device("cpu")
    ) -> BaseEvaluator:
        return _get_evaluator(
            task, self.context_length, self.sequence_length, self.tokenizer, device
        )

    @staticmethod
    def eval_datasets() -> list[str]:
        from qai_hub_models.datasets.mmmlu import mmmlu_splits

        return ["wikitext", "wikitext_ja", "tiny_mmlu", "mmlu", *mmmlu_splits]

    def __del__(self) -> None:
        # Clean up since it is prone to hang onto GPU memory otherwise
        if hasattr(self, "model") and self.model is not None:
            self.model = self.model.to("cpu")
            del self.model
            # Python can be in a weird state when __del__ gets called, so we
            # have to make sure these still exist.
            if "gc" in globals() and gc is not None:
                gc.collect()
            if "torch" in globals() and torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()


@unique
class LLMInstantiationType(Enum):
    """
    Types of an LLM "Instantiation"
    Export instantiates 2 copies of the LLM, each with a different sequence length.
    """

    PROMPT_PROCESSOR = "prompt"
    TOKEN_GENERATOR = "token"


LLM_AIMETOnnxT = TypeVar("LLM_AIMETOnnxT", bound="LLM_AIMETOnnx")


class LLM_AIMETOnnx(AIMETOnnxQuantizableMixin, LLMConfigEditor, BaseModel, ABC):
    # Embedding subclass
    EmbeddingClass: type[Embedding] | None = None

    # PyTorch equivalent of this class
    FPModel: type[LLMBase] | None = None

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        """Delegate to FPModel's get_chat_template."""
        assert cls.FPModel is not None
        return cls.FPModel.get_chat_template()

    @classmethod
    def get_input_prompt_with_tags(cls, **kwargs: Any) -> str:
        """Delegate to FPModel's get_input_prompt_with_tags."""
        assert cls.FPModel is not None
        return cls.FPModel.get_input_prompt_with_tags(**kwargs)

    def __init__(
        self,
        quant_sim: QuantizationSimModel | None,
        checkpoint: str | os.PathLike | Path | None,
        sequence_length: int,
        context_length: int,
        tokenizer: PreTrainedTokenizerBase | None = None,
        llm_config: PretrainedConfig | None = None,
        host_device: torch.device | None = None,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
    ) -> None:
        BaseModel.__init__(self)
        AIMETOnnxQuantizableMixin.__init__(self, quant_sim)
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.host_device = host_device

        assert (
            tokenizer is not None and llm_config is not None
        ) or checkpoint is not None, (
            f"{self.__class__.__name__} is unable to instantiate tokenizer/config. Must pass either checkpoint or tokenizer/config explicitly."
        )

        self.tokenizer = tokenizer or get_tokenizer(checkpoint)
        llm_config = llm_config or get_llm_config(checkpoint)
        self.llm_config = self.edit_llm_config(llm_config)
        assert self.EmbeddingClass is not None
        self.embedding = self.EmbeddingClass(
            max_length=context_length, config=llm_config
        )
        self.checkpoint = checkpoint
        self.attention_mask_min_clip = attention_mask_min_clip
        self.attention_mask_multiplier = attention_mask_multiplier

    def __del__(self) -> None:
        if hasattr(self, "quant_sim") and self.quant_sim is not None:
            del self.quant_sim
            # Python can be in a weird state when __del__ gets called, so we
            # have to make sure these still exist.
            if "gc" in globals() and gc is not None:
                gc.collect()
            if "torch" in globals() and torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def release(self) -> None:
        del self.quant_sim

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int,
        context_length: int,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        raise NotImplementedError

    @property
    def llm_io_type(self) -> LLMIOType:
        assert self.FPModel is not None
        return self.FPModel.llm_io_type

    @property
    def main_input_name(self) -> str:
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "input_embeds"
        return "input_ids"

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_spec = self.get_input_spec(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
                llm_config=self.llm_config.to_dict(),
            )
        assert self.FPModel is not None
        return sample_input(
            input_spec,
            self.get_input_prompt_with_tags(tokenizer=self.tokenizer),
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
            self.embedding,
        )

    def sample_inputs(self, input_spec: InputSpec | None = None) -> SampleInputsType:
        # This must be defined by the HubModelProtocol protocol via BaseModel
        return self._sample_inputs_impl(input_spec)

    @classmethod
    def from_pretrained(
        cls,
        host_device: torch.device,
        sequence_length: int,
        context_length: int,
        precision: Precision,
        fp_model: LLMBase | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        _skip_quantsim_creation: bool = False,
        use_dynamic_shapes: bool = False,
    ) -> Self:
        """
        Load weight from local checkpoint of Huggingface and create Aimet-ONNX QuantSim.
        Optionally load onnx model and AIMET encodings from a checkpoint.

        Parameters
        ----------
        host_device
            Device to use: GPU/CPU.
        sequence_length
            Sequence Length for the model.
        context_length
            Context Length for the model.
        precision
            Precision to use for the model.
        fp_model
            Floating point version of this model.
            This is quantized as part of this class and QuantSim model is created.
        checkpoint
            Path to previously calibrated AIMET encodings and ONNX models.
            Note that encodings are sensitive to AIMET ONNX versions.
        _skip_quantsim_creation
            If True, skip creating the QuantSim model.
        use_dynamic_shapes
            If True, export the ONNX model with dynamic sequence/context
            length dimensions.

        Returns
        -------
        Self
            Instance of the model loaded from the checkpoint.
        """
        if host_device is None:
            host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not _skip_quantsim_creation:
            if AIMET_ONNX_INSTALLED:
                AimetLogger.set_level_for_all_areas(logging.WARNING)
            onnx_path = None
            onnx_file_exists = False
            tmp_dir = tempfile.TemporaryDirectory()
            onnx_tmpfile = os.path.join(tmp_dir.name, "model.onnx")

            if checkpoint is None:
                onnx_file_exists = False
            else:
                external_data_exists = os.path.exists(
                    os.path.join(checkpoint, "model.onnx.data")
                ) or os.path.exists(os.path.join(checkpoint, "model.data"))

                # Look for fully dynamic ONNX model first (new format)
                onnx_path = os.path.join(checkpoint, "model_dynamic.onnx")
                onnx_file_exists = os.path.exists(onnx_path) and external_data_exists
                if not onnx_file_exists:
                    # Fall back to fixed-shape model (old format)
                    onnx_path = os.path.join(
                        checkpoint,
                        f"model_seqlen{sequence_length}_cl{context_length}.onnx",
                    )
                    onnx_file_exists = (
                        os.path.exists(onnx_path) and external_data_exists
                    )

            if not onnx_file_exists:
                if fp_model is None:
                    raise ValueError(
                        "The quantized checkpoint (with custom weights) must have an ONNX model."
                    )
                # Floating model is created if not passed when from_pretrained() is called and an ONNX model doesn't exist.
                onnx_model = get_onnx_model(
                    fp_model=fp_model,
                    context_length=context_length,
                    sequence_length=sequence_length,
                    path=onnx_tmpfile,
                    return_model=True,
                    llm_io_type=fp_model.llm_io_type,
                    use_dynamic_shapes=use_dynamic_shapes,
                )

            else:
                print()
                print(f"Loading onnx model from {onnx_path}")
                assert onnx_path is not None
                onnx_model = onnx.load(onnx_path, load_external_data=False)
                from onnx.external_data_helper import load_external_data_for_model

                load_external_data_for_model(onnx_model, os.path.dirname(onnx_path))

            if onnx_path is None:
                tmp_dir.cleanup()

            # Two copies are needed. One for QuantSim and one for passing to
            # quantize function for applying Sequential MSE.
            # Deepcopy causes error on GPU.
            print()
            print("Creating a QuantSim model using AIMET ONNX.")
            assert onnx_model is not None
            quant_sim = cls.create_quantsim(onnx_model, host_device, precision)

            # Cleanup the ONNX model that creates the QuantSim model
            del onnx_model
            gc.collect()

            # Encodings are not produced yet.
            if checkpoint is not None:
                aimet_encodings = os.path.join(checkpoint, "model.encodings")
                if os.path.exists(aimet_encodings):
                    print()
                    print(
                        f"Loading the encodings from path {checkpoint} to load the QuantSim model."
                    )
                    load_encodings_to_sim(quant_sim, aimet_encodings, strict=False)
        else:
            quant_sim = None

        attention_mask_min_clip, attention_mask_multiplier = (
            cls.attention_mask_min_clip_and_multiplier(precision)
        )

        return cls(
            quant_sim=quant_sim,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            checkpoint=checkpoint,
            tokenizer=fp_model.tokenizer if fp_model is not None else None,
            llm_config=fp_model.llm_config if fp_model is not None else None,
            attention_mask_min_clip=attention_mask_min_clip,
            attention_mask_multiplier=attention_mask_multiplier,
        )

    @classmethod
    def attention_mask_min_clip_and_multiplier(
        cls,
        precision: Precision,
    ) -> tuple[float | None, float]:
        if precision in {Precision.w4, Precision.float}:
            # Align with Genie (Genie/src/qualla/src/src/engines/qnn-htp/nsp-model.cpp)
            attention_mask_min_clip = -1000.0
        else:
            attention_mask_min_clip = -50.0
        return (attention_mask_min_clip, 1.0)

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        pass

    def _use_zip_file(self) -> bool:
        return False

    @staticmethod
    def _build_quantsim(
        onnx_model: onnx.ModelProto,
        providers: list[str | tuple[str, dict]],
    ) -> QuantizationSimModel:
        """Create a QuantizationSimModel with standard LLM parameters.

        Sets module-level AIMET globals and constructs the QuantSim with
        int4 weights / int16 activations. No precision-specific configuration
        is applied -- call ``_apply_precision_activations`` or
        ``_configure_quant_sim`` afterwards.
        """
        if not AIMET_ONNX_INSTALLED:
            raise ImportError(
                "Quantized models require the AIMET-ONNX package, which is only supported on Linux. "
                "Install qai-hub-models on a Linux machine to use quantized models."
            )

        AimetLogger.set_level_for_all_areas(logging.WARNING)
        default_config = get_aimet_config_path("default_config_llama")
        # Tie Quantizers for Concat Op
        quantsim.op_types_to_tie_qtzrs = ["Concat"]
        quantsim._tie_qtzrs = True
        # Ignore Slice and Constant outputs
        quantsim.op_outputs_to_ignore.append("Slice")
        quantsim.op_outputs_to_ignore.append("Constant")
        qs.encoding_version = "1.0.0"

        quant_sim = QuantizationSimModel(
            model=onnx_model,
            param_type="int4",
            activation_type="int16",
            quant_scheme=QuantScheme.min_max,
            config_file=default_config,
            providers=providers,
        )
        print(f"QuantSim session providers: {quant_sim.session.get_providers()}")
        return quant_sim

    @staticmethod
    def _apply_precision_activations(
        quant_sim: QuantizationSimModel, precision: Precision
    ) -> None:
        """Apply precision-dependent activation quantizer settings.

        Safe for both full models and split models (no structural
        heuristics). For w4: sets all activation quantizers to float16.
        """
        if precision == Precision.w4:
            for op_name, qc_op in quant_sim.qc_quantize_op_dict.items():
                if op_name in quant_sim.activation_names:
                    qc_op.reset_encoding_stats()
                    qc_op.data_type = QuantizationDataType.float
                    qc_op.bitwidth = 16

    @classmethod
    def create_quantsim(
        cls,
        onnx_model: onnx.ModelProto,
        host_device: torch.device,
        precision: Precision,
    ) -> QuantizationSimModel:
        """
        onnx_model: ONNX Model to create QuantSim model.
        host_device: Device that the QuantSim model must be placed on.
        """
        quant_sim = cls._build_quantsim(onnx_model, cls.get_ort_providers(host_device))
        return cls._configure_quant_sim(quant_sim, precision)

    @classmethod
    def _configure_quant_sim(
        cls, quant_sim: QuantizationSimModel, precision: Precision
    ) -> QuantizationSimModel:
        if precision == Precision.w4a16:
            kv_io_map = _get_kv_io_map(quant_sim)
            quant_sim = _apply_int8_kv_cache_tying_and_lm_head(quant_sim, kv_io_map)
        elif precision == Precision.w4:
            _set_lm_head_to_8b(quant_sim)
            cls._apply_precision_activations(quant_sim, precision)
        return quant_sim

    def save_calibrated_checkpoint(
        self,
        output_checkpoint: str | os.PathLike | Path,
        fp_model: LLMBase,
        use_dynamic_shapes: bool = False,
    ) -> None:
        """
        output_checkpoint: Path to the directory which must store the checkpoint.

        When use_dynamic_shapes is True, produces a single model_dynamic.onnx
        with symbolic dimensions. When False, produces per-sequence-length ONNX
        files (model_seqlen{sequence_length}_cl{context_length}.onnx).
        """
        os.makedirs(output_checkpoint, exist_ok=True)
        print(f"Creating a checkpoint of quantized model at {output_checkpoint}.")

        if not use_dynamic_shapes:
            # Static path: create per-seqlen ONNX files for each export seq_len
            export_sequence_lengths = {
                *DEFAULT_EXPORT_SEQUENCE_LENGTHS,
                self.sequence_length,
                self.context_length // 2,
            }
            export_sequence_lengths.discard(self.sequence_length)
            self.create_onnx_models(
                checkpoint=output_checkpoint,
                fp_model=fp_model,
                context_length=self.context_length,
                export_sequence_lengths=list(export_sequence_lengths),
                host_device=self.host_device or torch.device("cpu"),
                llm_io_type=self.llm_io_type,
            )

        # Remove existing .data file if it exists since QuantSim export will create a new one
        if os.path.exists(os.path.join(output_checkpoint, "model.data")):
            os.remove(os.path.join(output_checkpoint, "model.data"))

        # Export the QuantSim model (produces model.onnx + model.data + model.encodings).
        # If the input ONNX had dynamic shapes, QuantSim preserves them.
        assert self.quant_sim is not None
        # Export encodings only first, then save model with external weights
        # to avoid ByteSize() EncodeError when model exceeds protobuf 2GB limit.
        self.quant_sim.export(str(output_checkpoint), "model", export_model=False)
        from aimet_onnx.utils import save_model_with_external_weights

        with self.quant_sim._remove_quantization_nodes():
            save_model_with_external_weights(
                self.quant_sim.model.model,
                os.path.join(str(output_checkpoint), "model.onnx"),
                location="model.data",
                all_tensors_to_one_file=True,
            )
        del self.quant_sim

        onnx_path = os.path.join(output_checkpoint, "model.onnx")
        if use_dynamic_shapes:
            # Dynamic path: rename to model_dynamic.onnx
            shutil.move(
                onnx_path, os.path.join(output_checkpoint, "model_dynamic.onnx")
            )
        else:
            # Static path: rename to indicate static instance.
            shutil.move(
                onnx_path,
                os.path.join(
                    output_checkpoint,
                    f"model_seqlen{self.sequence_length}_cl{self.context_length}.onnx",
                ),
            )

        self.llm_config.save_pretrained(output_checkpoint)
        self.tokenizer.save_pretrained(output_checkpoint)

    @classmethod
    def create_onnx_models(
        cls,
        checkpoint: str | os.PathLike | Path,
        fp_model: LLMBase,
        context_length: int,
        export_sequence_lengths: list[int] | None = None,
        host_device: torch.device = torch.device("cpu"),
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
        use_dynamic_shapes: bool = False,
    ) -> None:
        """
        Export ONNX models for the checkpoint.

        When use_dynamic_shapes is True, creates a single model_dynamic.onnx
        with symbolic dimensions.

        Otherwise (default), creates one model_seqlen{seq}_cl{ctx}.onnx per
        sequence length in export_sequence_lengths.
        """
        external_weights_file = os.path.join(checkpoint, "model.data")
        onnx_file = os.path.join(checkpoint, "model.onnx")

        if use_dynamic_shapes:
            # Dynamic: single model_dynamic.onnx
            dynamic_onnx_model = os.path.join(checkpoint, "model_dynamic.onnx")
            if not os.path.exists(dynamic_onnx_model) or not os.path.exists(
                external_weights_file
            ):
                get_onnx_model(
                    fp_model=fp_model,
                    context_length=context_length,
                    sequence_length=DEFAULT_SEQUENCE_LENGTH,
                    path=onnx_file,
                    llm_io_type=llm_io_type,
                    use_dynamic_shapes=True,
                )
                shutil.move(onnx_file, dynamic_onnx_model)
        elif export_sequence_lengths is not None:
            # Static: per-seqlen ONNX files
            for seq_len in export_sequence_lengths:
                expected_onnx_model = os.path.join(
                    checkpoint, f"model_seqlen{seq_len}_cl{context_length}.onnx"
                )
                if not os.path.exists(expected_onnx_model) or not os.path.exists(
                    external_weights_file
                ):
                    get_onnx_model(
                        fp_model=fp_model,
                        context_length=context_length,
                        sequence_length=seq_len,
                        path=onnx_file,
                        llm_io_type=llm_io_type,
                    )
                    shutil.move(onnx_file, expected_onnx_model)

    @classmethod
    def save_tokenizer_and_config(
        cls, checkpoint: str | os.PathLike | Path, fp_model: LLMBase
    ) -> None:
        # Make sure tokenizer/config exist in the checkpoint
        if not os.path.isfile(os.path.join(checkpoint, "tokenizer.json")):
            fp_model.tokenizer.save_pretrained(checkpoint)
        if not os.path.isfile(os.path.join(checkpoint, "config.json")):
            fp_model.llm_config.save_pretrained(checkpoint)

    def convert_to_onnx_and_aimet_encodings(
        self,
        output_dir: str | os.PathLike | Path,
        input_spec: InputSpec | None = None,
        model_name: str | None = None,
        external_weights: bool = False,
        bundle_external_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str:
        if model_name is None:
            model_name = self.__class__.__name__

        base_path = os.path.join(output_dir, f"{model_name}.aimet")
        os.makedirs(base_path, exist_ok=True)
        assert self.checkpoint is not None

        src_onnx_filepath = os.path.join(self.checkpoint, "model_dynamic.onnx")
        if not os.path.exists(src_onnx_filepath):
            # Fall back to fixed-shape model
            src_onnx_filepath = os.path.join(
                self.checkpoint,
                f"model_seqlen{self.sequence_length}_cl{self.context_length}.onnx",
            )
        # Find the external data file (always "model.data" in our pipelines)
        src_external_weights_filepath = None
        for name in ["model.data", "model.onnx.data"]:
            candidate = os.path.join(self.checkpoint, name)
            if os.path.exists(candidate):
                src_external_weights_filepath = candidate
                break
        assert src_external_weights_filepath is not None, (
            f"External data file not found in {self.checkpoint}"
        )

        src_encodings_filepath = os.path.join(self.checkpoint, "model.encodings")

        dst_onnx_filepath = os.path.join(base_path, "model.onnx")
        dst_external_weights_filepath = os.path.join(
            base_path, os.path.basename(src_external_weights_filepath)
        )
        dst_encodings_filepath = os.path.join(base_path, "model.encodings")

        shutil.copy(src_onnx_filepath, dst_onnx_filepath)
        shutil.copy(src_external_weights_filepath, dst_external_weights_filepath)

        self._adapt_aimet_encodings(
            src_encodings_filepath, dst_encodings_filepath, dst_onnx_filepath
        )

        return base_path

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if target_runtime != TargetRuntime.GENIE:
            raise RuntimeError(
                f"Unsupported target_runtime provided: {target_runtime}."
                " Only Genie runtime is supported."
            )

        if precision not in {Precision.w4a16, Precision.w4}:
            raise RuntimeError("Only w4a16 and w4 precisions are supported")

        other_compile_options += " --quantize_full_type w8a16 --quantize_io"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        return super().get_hub_link_options(
            target_runtime,
            other_link_options,
        )

    def get_qnn_context_graph_name(self, split_index: int, num_splits: int) -> str:
        """
        Get the name of the QAIRT Context Graph applicable for the given sub-component.

        Sequence length (ar...) and context length (cl...) in graph name
        are semantically important to Genie
        """
        if self.sequence_length == 1:
            instantiation_type = LLMInstantiationType.TOKEN_GENERATOR
        else:
            instantiation_type = LLMInstantiationType.PROMPT_PROCESSOR
        return f"{instantiation_type.value}_ar{self.sequence_length}_cl{self.context_length}_{split_index + 1}_of_{num_splits}"

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        profile_options = super().get_hub_profile_options(
            target_runtime, other_profile_options, context_graph_name=context_graph_name
        )
        profile_options += " --max_profiler_iterations 50"
        return profile_options

    def get_calibration_data(
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        from qai_hub_models.models._shared.llm.generator import LLM_Generator
        from qai_hub_models.datasets.common import DatasetSplit
        from qai_hub_models.datasets import get_dataset_from_name

        if num_samples == 0:
            num_samples = math.ceil(80000 / self.context_length)

        dataset = get_dataset_from_name(
            name="wikitext",
            split=DatasetSplit.TRAIN,
            tokenizer=self.tokenizer,
            block_size=self.sequence_length,
            context_length=self.context_length,
            num_samples=num_samples,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=self.sequence_length,
            context_length=self.context_length,
            llm_io_type=self.llm_io_type,
        )
        assert input_spec is not None
        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]

        assert self.EmbeddingClass is not None
        rope_embeddings = self.EmbeddingClass(
            max_length=self.context_length, config=self.llm_config
        )
        generator = LLM_Generator(
            [self],
            self.tokenizer,
            rope_embeddings,
        )

        # Only bother removing quantization if we don't have a floating point model provided
        with self.remove_quantization():
            # for data in dataloader
            for sample in tqdm(
                dataloader, total=len(dataloader), desc="Pre-filling calibration data"
            ):
                input_ids, attention_mask, _ = sample
                for prefilled_inputs in generator.prefill(input_ids, attention_mask):
                    for i, tensor in enumerate(prefilled_inputs):
                        inputs[i].append(tensor)

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))

    def get_evaluator(
        self, task: str = "wikitext", device: torch.device = torch.device("cpu")
    ) -> BaseEvaluator:
        return _get_evaluator(
            task, self.context_length, self.sequence_length, self.tokenizer, device
        )

    eval_datasets = LLMBase.eval_datasets

    @classmethod
    def prepare_genie_assets(
        cls,
        hub_device: hub.Device,
        checkpoint: str | os.PathLike | Path,
        llm_config: PretrainedConfig,
        context_lengths: list[int],
        model_list: list[str],
        output_path: Path,
        precision: Precision,
        encodings_path: str | os.PathLike | Path,
        input_specs: dict[str, Any],
        output_specs: dict[str, Any],
        model_id: str,
        model_name: str,
    ) -> None:
        """Prepare assets to run the model end to end on-device using Genie SDK."""
        context_length = max(context_lengths)
        # Copy necessary config files
        for name in ["tokenizer.json", "tokenizer_config.json", "config.json"]:
            if (Path(checkpoint) / name).exists():
                shutil.copy(
                    Path(checkpoint) / name,
                    output_path / name,
                )
        # Save the HTP config
        save_htp_config_for_genie_bundle(hub_device, output_path)
        # Save the genie config
        config = create_genie_config(context_length, llm_config, "rope", model_list)
        with open(output_path / "genie_config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Save metadata needed for on-device run (via LLM_QNN)
        metadata = cls._create_model_metadata(
            model_id,
            model_name,
            precision,
            TargetRuntime.GENIE,
            encodings_path,
            input_specs,
            output_specs,
        )

        # --- Genie supplementary files ---
        # text-generator.json: same as genie_config but with "text-generator" key
        text_gen_config = create_genie_config(
            context_length,
            llm_config,
            "rope",
            model_list,
            top_level_key="text-generator",
        )
        with open(output_path / "text-generator.json", "w") as f:
            json.dump(text_gen_config, f, indent=4)

        # sample_prompt.txt
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        sample_prompt = cls.get_input_prompt_with_tags(tokenizer=tokenizer)
        with open(output_path / "sample_prompt.txt", "w") as f:
            f.write(sample_prompt)

        # Populate supplementary_files
        metadata.supplementary_files = {
            "tokenizer.json": "Model tokenizer.json from checkpoint.",
            "tokenizer_config.json": "Model tokenizer_config.json from checkpoint.",
            "config.json": "Model config.json from checkpoint.",
            "htp_backend_ext_config.json": "HTP backend extension config for Genie.",
            "text-generator.json": "Genie SDK config for text generator.",
            "genie_config.json": "Config for use in genie-t2t-run.",
            "genie-app-script.txt": "Genie-app pipeline script for LLM inference.",
        }

        # Populate genie metadata
        chat_spec = cls.get_chat_template()
        pipeline_nodes = {"textGenerator": "text-generator.json"}
        sample_inputs = [
            GenieSampleInput(
                node="textGenerator",
                node_io="GENIE_NODE_TEXT_GENERATOR_TEXT_INPUT",
                file="sample_prompt.txt",
            ),
        ]
        metadata.genie = GenieMetadata(
            chat_template=GenieChatTemplate(**chat_spec)
            if chat_spec
            else GenieChatTemplate(),
            context_lengths=context_lengths,
            supports_streaming=True,
            supports_vision=False,
            pipeline=GeniePipeline(
                nodes=pipeline_nodes,
                connections=[],
            ),
            sample_inputs=sample_inputs,
        )

        # genie-app-script.txt
        from qai_hub_models.utils.llm_helpers import generate_genie_app_script

        genie_script = generate_genie_app_script(pipeline_nodes, [], sample_inputs)
        with open(output_path / "genie-app-script.txt", "w") as f:
            f.write(genie_script)

        metadata.to_json(output_path / "metadata.json")

    @staticmethod
    def _create_model_metadata(
        model_id: str,
        model_name: str,
        precision: Precision,
        target_runtime: TargetRuntime,
        input_encodings_path: str | os.PathLike | Path,
        input_specs: dict[str, Any],
        output_specs: dict[str, Any],
    ) -> ModelMetadata:
        def make_io_metadata(
            specs: dict[str, Any], encodings: dict[str, Any]
        ) -> dict[str, TensorSpec]:
            uses_lists = Version(encodings["version"]) >= Version("1.0.0")
            if uses_lists:
                all_encodings = {
                    v["name"]: v for v in encodings["activation_encodings"]
                }
            else:
                all_encodings = encodings["activation_encodings"]

            entries: dict[str, TensorSpec] = {}
            for name, (shape, dtype_str) in specs.items():
                entry = TensorSpec(shape=tuple(shape), dtype=dtype_str)

                fixed_name = name
                if name == "logits":
                    # The logits are missing. We need to start correcting this when
                    # making the encodings. For legacy encodings, we try to look it up.
                    for key in all_encodings:
                        if "lm_head" in key and "Conv_output_0" in key:
                            fixed_name = key
                            break

                if node_encodings := all_encodings.get(fixed_name):
                    if isinstance(node_encodings, list):
                        scale = node_encodings[0].get("scale")
                        offset = node_encodings[0].get("offset")

                        if scale is not None and offset is not None:
                            entry.quantization_parameters = QuantizationParameters(
                                scale=scale, zero_point=-offset
                            )
                    else:
                        scale = node_encodings.get("scale")
                        offset = node_encodings.get("offset")

                        if scale is not None and offset is not None:
                            entry.quantization_parameters = QuantizationParameters(
                                scale=scale[0], zero_point=-offset[0]
                            )

                entries[name] = entry
            return entries

        with open(input_encodings_path) as f:
            encodings = json.load(f)

        input_metadata = {
            k: make_io_metadata(v, encodings) for k, v in input_specs.items()
        }
        output_metadata = {
            k: make_io_metadata(v, encodings) for k, v in output_specs.items()
        }

        model_files: dict[str, ModelFileMetadata] = {}
        for k, v in input_metadata.items():
            model_files[k] = ModelFileMetadata(inputs=v, outputs=output_metadata[k])

        return ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            precision=precision,
            runtime=target_runtime,
            tool_versions=ToolVersions(),
            model_files=model_files,
        )

    @functools.cache
    def _get_embedding_table(self) -> torch.nn.Embedding:
        if self.quant_sim is None:
            raise RuntimeError(
                "Cannot get embedding table from LLM object created with _skip_quantsim_creation=True"
            )
        assert isinstance(self.quant_sim, QuantizationSimModel)

        lm_head_weights = list(_get_lm_head_weights(self.quant_sim.model.model))
        if len(lm_head_weights) != 1:
            raise RuntimeError("Unable to isolate LM Head weights from ONNX Model.")

        embedding_table = torch.from_numpy(
            onnx.numpy_helper.to_array(lm_head_weights[0]).copy()
        )
        return torch.nn.Embedding(
            self.llm_config.vocab_size,
            self.llm_config.hidden_size,
            self.llm_config.pad_token_id,
            _weight=embedding_table.T,
        )

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._get_embedding_table()(input_ids)


class LLM_QNN(LLMConfigEditor, BaseModel, ABC):
    # Embedding subclass
    FPModel: type[LLMBase] | None = None
    EmbeddingClass: type[Embedding] | None = None
    num_layers_per_split: int
    llm_io_type: LLMIOType = LLMIOType.genie_input_ids

    @classmethod
    def get_input_prompt_with_tags(cls, **kwargs: Any) -> str:
        """Delegate to FPModel's get_input_prompt_with_tags."""
        assert cls.FPModel is not None
        return cls.FPModel.get_input_prompt_with_tags(**kwargs)

    def __init__(
        self,
        part_models: dict[tuple[str, int], OnnxModelTorchWrapper],
        checkpoint: str | os.PathLike | Path | None,
        metadata: ModelMetadata,
        sequence_length: int,
        context_length: int,
        precision: Precision,
        tokenizer: PreTrainedTokenizerBase | None = None,
        llm_config: PretrainedConfig | None = None,
        host_device: torch.device | None = None,
        _temporary_paths: list[Path] | None = None,
    ) -> None:
        BaseModel.__init__(self)
        self.part_models = part_models
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.host_device = host_device
        self.metadata = metadata
        self.precision = precision
        self._temporary_paths = _temporary_paths

        assert (
            tokenizer is not None and llm_config is not None
        ) or checkpoint is not None, (
            f"{self.__class__.__name__} is unable to instantiate tokenizer/config. Must pass either checkpoint or tokenizer/config explicitly."
        )

        self.tokenizer = tokenizer or get_tokenizer(checkpoint)
        llm_config = llm_config or get_llm_config(checkpoint)
        self.llm_config = self.edit_llm_config(llm_config)
        assert self.EmbeddingClass is not None
        self.embedding = self.EmbeddingClass(
            max_length=context_length, config=llm_config
        )
        self.checkpoint = checkpoint

    def release(self) -> None:
        self.part_models.clear()

    @staticmethod
    def get_input_spec(
        llm_config: dict,
        sequence_length: int,
        context_length: int,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        raise NotImplementedError

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_spec = self.get_input_spec(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
                llm_config=self.llm_config.to_dict(),
            )
        assert self.FPModel is not None
        return sample_input(
            input_spec,
            self.get_input_prompt_with_tags(tokenizer=self.tokenizer),
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
            self.embedding,
        )

    def sample_inputs(self, input_spec: InputSpec | None = None) -> SampleInputsType:
        # This must be defined by the HubModelProtocol protocol via BaseModel
        return self._sample_inputs_impl(input_spec)

    @classmethod
    def from_pretrained(
        cls,
        host_device: torch.device,
        sequence_length: int,
        context_length: int,
        fp_model: LLMBase | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        _skip_quantsim_creation: bool = False,
    ) -> Self:
        """
        Load weight from local checkpoint of Huggingface and create Aimet-ONNX QuantSim.
        Optionally load onnx model and AIMET encodings from a checkpoint.

        Parameters
        ----------
        host_device
            Device to use: GPU/CPU.
        sequence_length
            Sequence Length for the model.
        context_length
            Context Length for the model.
        fp_model
            Floating point version of this model.
            This is quantized as part of this class and QuantSim model is created.
        checkpoint
            Path to previously calibrated AIMET encodings and
            ONNX models. Note that encodings are sensitive to AIMET ONNX versions.
        _skip_quantsim_creation
            If True, skip creating the QuantSim model.

        Returns
        -------
        LLM_QNN : Self
            Instance of the model loaded from the checkpoint.
        """
        _verify_onnxruntime_qnn_installed()

        if host_device is None:
            host_device = torch.device("cpu")

        inst = "prompt" if sequence_length > 1 else "token"

        assert checkpoint is not None
        checkpoint_path = Path(checkpoint)

        # Construct ONNX model wrapping the context binaries
        context_bin_paths = sorted(checkpoint_path.glob("*.bin"))

        metadata = ModelMetadata.from_json(checkpoint_path / "metadata.json")

        tool_versions = ToolVersions.from_yaml(checkpoint_path / "tool-versions.yaml")
        assert tool_versions.qairt is not None
        qairt_version = tool_versions.qairt.full_version

        part_models: dict[tuple[str, int], OnnxModelTorchWrapper] = {}
        temporary_paths: list[Path] = []

        num_splits = len(context_bin_paths)
        for part_i, context_bin_path in enumerate(context_bin_paths):
            graph_name = cls.construct_qnn_context_graph_name(
                part_i, len(context_bin_paths), sequence_length, context_length
            )
            onnx_path = checkpoint_path / (
                os.path.splitext(os.path.basename(context_bin_path))[0]
                + f"_{inst}.onnx"
            )

            io_metadata = metadata.model_files[f"{inst}_{part_i + 1}_of_{num_splits}"]

            def _io_metadata_to_specs(
                entries: dict[str, TensorSpec],
            ) -> dict[str, ModelIODetails]:
                onnx_specs: dict[str, Any] = {}
                for k, v in entries.items():
                    qdq_params = None
                    if v.quantization_parameters is not None:
                        qdq_params = ModelIODetails.QDQParams(
                            scale=v.quantization_parameters.scale,
                            zero_point=v.quantization_parameters.zero_point,
                        )

                    onnx_specs[k] = ModelIODetails(
                        shape=v.shape,
                        dtype=np.dtype(v.dtype),
                        qdq_params=qdq_params,
                    )
                return onnx_specs

            onnx_input_specs = _io_metadata_to_specs(io_metadata.inputs)
            onnx_output_specs = _io_metadata_to_specs(io_metadata.outputs)

            rel_context_bin_path = os.path.relpath(context_bin_path, checkpoint)
            generate_wrapper_onnx_file(
                graph_name,
                onnx_path,
                onnx_input_specs,
                onnx_output_specs,
                rel_context_bin_path,
                qairt_version,
            )
            # Quantization parameters for intermediates are not available and
            # will raise warnings. Leaving them quantized is fine.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                part_models[(inst, part_i)] = OnnxModelTorchWrapper.OnNPU(onnx_path)
            temporary_paths.append(onnx_path)

        return cls(
            part_models=part_models,
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            metadata=metadata,
            context_length=context_length,
            host_device=host_device,
            precision=metadata.precision,
            _temporary_paths=temporary_paths,
        )

    def __del__(self) -> None:
        if hasattr(self, "_temporary_paths") and self._temporary_paths:
            for path in self._temporary_paths:
                os.remove(path)

    def forward(self, *args: torch.Tensor) -> list[torch.Tensor]:
        input_ids = args[0]
        sequence_length = input_ids.shape[1]
        num_splits = len(self.part_models)
        inst = "prompt" if sequence_length > 1 else "token"

        part1_model = self.part_models[(inst, 0)]
        part1_out = part1_model(input_ids)

        last_intermediate = part1_out

        # Shared between between parts
        attn_mask = args[1]
        position_ids_cos = args[2]
        position_ids_sin = args[3]

        kv_output: list[torch.Tensor] = []

        for part_i in range(1, num_splits):
            torch_part_i_inputs = [
                last_intermediate,
                attn_mask,
                position_ids_cos,
                position_ids_sin,
            ]

            kv_indices = range(
                (part_i - 1) * self.num_layers_per_split,
                min(
                    part_i * self.num_layers_per_split,
                    self.llm_config.num_hidden_layers,
                ),
            )
            for idx in kv_indices:
                torch_part_i_inputs.append(args[4 + idx * 2])
                torch_part_i_inputs.append(args[4 + idx * 2 + 1])

            part_i_model = self.part_models[(inst, part_i)]
            part_i_out = part_i_model(*torch_part_i_inputs)

            last_intermediate = part_i_out[0]
            kv_output += part_i_out[1:]

        logits = last_intermediate
        return [cast(torch.Tensor, logits), *kv_output]

    @classmethod
    def construct_qnn_context_graph_name(
        cls,
        split_index: int,
        num_splits: int,
        sequence_length: int,
        context_length: int,
    ) -> str:
        """Similar to get_qnn_context_graph_name, but can be run without instance."""
        if sequence_length == 1:
            instantiation_type = LLMInstantiationType.TOKEN_GENERATOR
        else:
            instantiation_type = LLMInstantiationType.PROMPT_PROCESSOR
        return f"{instantiation_type.value}_ar{sequence_length}_cl{context_length}_{split_index + 1}_of_{num_splits}"

    def get_qnn_context_graph_name(self, split_index: int, num_splits: int) -> str:
        """
        Get the name of the QNN Context Graph applicable for the given sub-component.

        Sequence length (ar...) and context length (cl...) in graph name
        are semantically important to Genie.
        """
        return self.construct_qnn_context_graph_name(
            split_index, num_splits, self.sequence_length, self.context_length
        )

    def get_evaluator(
        self, task: str = "wikitext", device: torch.device = torch.device("cpu")
    ) -> BaseEvaluator:
        return _get_evaluator(
            task, self.context_length, self.sequence_length, self.tokenizer, device
        )

    eval_datasets = LLMBase.eval_datasets

    @property
    def main_input_name(self) -> str:
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "input_embeds"
        return "input_ids"

    @functools.cache
    def _get_embedding_table(self) -> torch.nn.Embedding:
        if self.quant_sim is None:
            raise RuntimeError(
                "Cannot get embedding table from LLM object created with _skip_quantsim_creation=True"
            )
        assert isinstance(self.quant_sim, QuantizationSimModel)

        lm_head_weights = list(_get_lm_head_weights(self.quant_sim.model.model))
        if len(lm_head_weights) != 1:
            raise RuntimeError("Unable to isolate LM Head weights from ONNX Model.")

        embedding_table = torch.from_numpy(
            onnx.numpy_helper.to_array(lm_head_weights[0]).copy()
        )
        return torch.nn.Embedding(
            self.llm_config.vocab_size,
            self.llm_config.hidden_size,
            self.llm_config.pad_token_id,
            _weight=embedding_table.T,
        )

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._get_embedding_table()(input_ids)

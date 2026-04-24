# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Metadata schema for model I/O specifications.

This module defines the structure for metadata.json files that document
model input/output specifications and quantization parameters for a single
exported model.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qai_hub_models_bk.configs.tensor_spec import (
    QuantizationParameters,
    TensorSpec,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.base_config import BaseQAIHMConfig

if TYPE_CHECKING:
    import qai_hub as hub

    from qai_hub_models.utils.input_spec import InputSpec

# Output spec: maps output name -> TensorSpec with metadata only (shape/dtype
# come from the compiled model). Used by model.get_output_spec().
OutputSpec = dict[str, TensorSpec]


class ModelFileMetadata(BaseQAIHMConfig):
    """
    Metadata for a single exported model file.

    This represents the I/O specifications for one exported model component,
    including input/output tensor shapes, dtypes, and quantization parameters.
    """

    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]

    @classmethod
    def from_hub_model(cls, hub_model: hub.Model) -> ModelFileMetadata:
        """
        Create ModelFileMetadata directly from hub.Model.

        Parameters
        ----------
        hub_model
            The hub.Model from compile_job.get_target_model()

        Returns
        -------
        metadata: ModelFileMetadata
            Created model file metadata
        """
        inputs = {}
        outputs = {}

        # Extract inputs from hub.Model.input_spec
        for tensor_specs in hub_model.input_spec.values():
            for tensor_spec in tensor_specs:
                quant_params = None
                if tensor_spec.scale is not None and tensor_spec.zero_point is not None:
                    quant_params = QuantizationParameters(
                        scale=tensor_spec.scale,
                        zero_point=int(tensor_spec.zero_point),
                    )
                assert tensor_spec.shape is not None
                inputs[tensor_spec.name] = TensorSpec(
                    shape=tuple(tensor_spec.shape),
                    dtype=tensor_spec.dtype,
                    quantization_parameters=quant_params,
                )

        # Extract outputs from hub.Model.output_spec
        for tensor_specs in hub_model.output_spec.values():
            for tensor_spec in tensor_specs:
                quant_params = None
                if tensor_spec.scale is not None and tensor_spec.zero_point is not None:
                    quant_params = QuantizationParameters(
                        scale=tensor_spec.scale,
                        zero_point=int(tensor_spec.zero_point),
                    )
                assert tensor_spec.shape is not None
                outputs[tensor_spec.name] = TensorSpec(
                    shape=tuple(tensor_spec.shape),
                    dtype=tensor_spec.dtype,
                    quantization_parameters=quant_params,
                )

        return cls(inputs=inputs, outputs=outputs)

    def to_yaml(
        self,
        path: str | Path,
        write_if_empty: bool = False,
        delete_if_empty: bool = True,
        flow_lists: bool = True,
        **kwargs: Any,
    ) -> bool:
        """
        Override to_yaml to default flow_lists=True for metadata files.

        This ensures lists (like tensor shapes) are formatted in flow style
        (e.g., [1, 3, 224, 224]) instead of block style for better readability.
        """
        return super().to_yaml(
            path=path,
            write_if_empty=write_if_empty,
            delete_if_empty=delete_if_empty,
            flow_lists=flow_lists,
            **kwargs,
        )


class GenieChatTemplate(BaseQAIHMConfig):
    """Chat template tokens and defaults for Genie SDK."""

    global_prefix: str = ""
    system_prefix: str = ""
    system_suffix: str = ""
    user_prefix: str = ""
    user_suffix: str = ""
    assistant_prefix: str = ""
    assistant_suffix: str = ""
    vision_start: str = ""
    vision_end: str = ""
    default_system_prompt: str = ""


class GeniePipelineConnection(BaseQAIHMConfig):
    """A connection between two nodes in a Genie pipeline."""

    producer_node: str
    producer_node_io: str
    consumer_node: str
    consumer_node_io: str


class GenieSampleInput(BaseQAIHMConfig):
    """A sample input binding for a Genie pipeline node."""

    node: str
    node_io: str
    file: str


class GeniePipeline(BaseQAIHMConfig):
    """Pipeline topology for Genie SDK."""

    nodes: dict[str, str]
    connections: list[GeniePipelineConnection]


class GenieVisionPreprocessing(BaseQAIHMConfig):
    """Vision encoder preprocessing parameters for Genie SDK."""

    image_width: int
    image_height: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    normalize_mean: list[float]
    normalize_std: list[float]


class GenieMetadata(BaseQAIHMConfig):
    """Genie SDK metadata for on-device deployment."""

    chat_template: GenieChatTemplate
    context_lengths: list[int]
    supports_streaming: bool = True
    supports_vision: bool = False
    pipeline: GeniePipeline | None = None
    sample_inputs: list[GenieSampleInput] | None = None
    vision_preprocessing: GenieVisionPreprocessing | None = None


class ModelMetadata(BaseQAIHMConfig):
    """
    Metadata for a model that may have multiple model files.

    For single-file models (e.g., ResNet50), this will have one entry.
    For multi-file models (e.g., Stable Diffusion with text_encoder.bin, unet.bin, vae_decoder.bin),
    this will have multiple entries mapping model file names to their metadata.

    The keys in model_files are the actual file names (relative to the root of the export directory),
    making it clear what each metadata entry corresponds to in the exported files.

    This is the class that gets saved to disk as metadata.json.

    Attributes
    ----------
    model_id
        The model identifier (folder name), e.g. "mobilenet_v2".
    model_name
        The human-readable model name from info.yaml, e.g. "MobileNet-v2".
    runtime
        The target runtime for which the model was exported.
    precision
        The precision configuration for which the model was exported
    tool_versions
        ToolVersions object containing version information for tools used to compile/export the model.
        Includes fields like tflite, onnx, onnx_runtime, qairt, and ai_hub_models versions.
    model_files
        Dictionary mapping model file names to their metadata (I/O specs, quantization params).
    supplementary_files
        Optional dictionary mapping supplementary file names to their descriptions.
            key: file name (relative to root of export directory)
            value: description of the file's contents and purpose
        This can be populated by the model's write_supplementary_files method to document any
        additional files included in the export that are not model files (e.g., labels, config files).
    genie
        Optional Genie SDK metadata for on-device deployment (chat template, pipeline topology, etc.).
    """

    model_id: str = ""
    model_name: str = ""
    runtime: TargetRuntime
    precision: Precision
    tool_versions: ToolVersions
    model_files: dict[str, ModelFileMetadata]
    supplementary_files: dict[str, str] = {}
    genie: GenieMetadata | None = None

    def to_yaml(
        self,
        path: str | Path,
        write_if_empty: bool = False,
        delete_if_empty: bool = True,
        flow_lists: bool = True,
        **kwargs: Any,
    ) -> bool:
        """
        Override to_yaml for metadata files:
        - flow_lists=True for readable shapes (e.g., [1, 3, 224, 224])
        - exclude_unset=True (instead of exclude_defaults) so that fields
          explicitly set by merge_input_metadata (like io_type) are preserved
          even when they match the default value.
        """
        return super().to_yaml(
            path=path,
            write_if_empty=write_if_empty,
            delete_if_empty=delete_if_empty,
            flow_lists=flow_lists,
            exclude_defaults=False,
            exclude_unset=True,
            **kwargs,
        )


def merge_input_metadata(
    model_file_metadata: ModelFileMetadata,
    input_spec: InputSpec,
) -> None:
    """
    Merge semantic metadata from get_input_spec() into ModelFileMetadata.

    This function enriches the TensorSpec entries in model_file_metadata.inputs
    with additional metadata (io_type, image_metadata, value_range, description)
    from the model's get_input_spec() return value.

    TensorSpec entries in input_spec may have metadata fields set (io_type,
    image_metadata, etc.). This function copies those fields to the corresponding
    TensorSpec in model_file_metadata.inputs.

    Parameters
    ----------
    model_file_metadata
        The ModelFileMetadata to enrich (modified in place).
    input_spec
        The InputSpec from model.get_input_spec(). TensorSpec entries with
        metadata fields will have their metadata merged.

    Raises
    ------
    ValueError
        If an input in input_spec is not found in model_file_metadata.inputs.
        This indicates a bug - the input names should match.
    """
    # Link job models may have empty I/O specs (workbench bug).
    # Skip metadata merge entirely — compiled shapes can differ from
    # get_input_spec() (e.g. NCHW -> NHWC conversion during compile).
    if not model_file_metadata.inputs:
        return

    for input_name, spec in input_spec.items():
        # Find matching input in model_file_metadata
        if input_name not in model_file_metadata.inputs:
            raise ValueError(
                f"Input '{input_name}' from get_input_spec() not found in compiled "
                f"model metadata. Available inputs: {list(model_file_metadata.inputs.keys())}"
            )

        # Skip if not a TensorSpec (plain tuple has no metadata to merge)
        if not isinstance(spec, TensorSpec):
            continue

        target_spec = model_file_metadata.inputs[input_name]

        # Copy metadata fields if present
        if spec.io_type is not None:
            target_spec.io_type = spec.io_type
        if spec.image_metadata is not None:
            target_spec.image_metadata = spec.image_metadata
        if spec.description is not None:
            target_spec.description = spec.description
        if spec.value_range != (float("-inf"), float("inf")):
            target_spec.value_range = spec.value_range


def merge_output_metadata(
    model_file_metadata: ModelFileMetadata,
    output_spec: OutputSpec,
) -> None:
    """
    Merge semantic metadata from get_output_spec() into ModelFileMetadata.

    This function enriches the TensorSpec entries in model_file_metadata.outputs
    with additional metadata (io_type, bbox_metadata, description) from the
    model's get_output_spec() return value.

    Parameters
    ----------
    model_file_metadata
        The ModelFileMetadata to enrich (modified in place).
    output_spec
        The OutputSpec from model.get_output_spec(). TensorSpec entries with
        metadata fields will have their metadata merged.

    Raises
    ------
    ValueError
        If an output in output_spec is not found in model_file_metadata.outputs.
    """
    if not model_file_metadata.outputs:
        return

    for output_name, spec in output_spec.items():
        if output_name not in model_file_metadata.outputs:
            raise ValueError(
                f"Output '{output_name}' from get_output_spec() not found in compiled "
                f"model metadata. Available outputs: {list(model_file_metadata.outputs.keys())}"
            )

        target_spec = model_file_metadata.outputs[output_name]

        if spec.io_type is not None:
            target_spec.io_type = spec.io_type
        if spec.bbox_metadata is not None:
            target_spec.bbox_metadata = spec.bbox_metadata
        if spec.description is not None:
            target_spec.description = spec.description
        if spec.value_range != (float("-inf"), float("inf")):
            target_spec.value_range = spec.value_range

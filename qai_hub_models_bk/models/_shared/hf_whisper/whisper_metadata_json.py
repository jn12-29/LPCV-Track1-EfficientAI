# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Whisper metadata schema.

This module defines the structure for the config that document
Whisper model capabilities, asset locations and so on.
"""

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.base_config import BaseQAIHMConfig


class WhisperCapabilities(BaseQAIHMConfig):
    """Supported capabilities for a Whisper model.

    All fields are required with no defaults — different Whisper model sizes
    have very different latency and capability profiles, so each model variant
    must explicitly declare its own capabilities rather than relying on
    universal defaults.
    """

    streaming: bool
    file_based: bool
    real_time: bool
    language_detection: bool
    confidence_scores: bool


class ModelParameters(BaseQAIHMConfig):
    """Parameters for a Whisper model."""

    max_file_size_mb: int = 25
    supported_formats: list[str] = ["wav"]
    sample_rates: list[int] = [16000]


class ModelAssets(BaseQAIHMConfig):
    """Paths to Whisper asset files."""

    encoder_path: str | None = None
    vocab_path: str | None = None
    decoder_path: str | None = None


class QNNVersion(BaseQAIHMConfig):
    """Version of QNN SDK."""

    major: int
    minor: int
    patch: int = 0


class RuntimeInfo(BaseQAIHMConfig):
    """Runtime configuration information."""

    qnn_version: QNNVersion
    arch: int = 64


class WhisperMetadata(BaseQAIHMConfig):
    # Whisper metadata for Whisper models.

    name: str
    display_name: str
    version: str = "1.0.0"
    description: str
    capabilities: WhisperCapabilities | None = None
    parameters: ModelParameters | None = None
    model_type: str = "whisper"
    assets: ModelAssets | None = None
    runtime: RuntimeInfo | None = None

    # ------------------------------------------------------------------
    # Builds a WhisperMetadata instance from model files
    # ------------------------------------------------------------------
    @classmethod
    def from_whisper_model(
        cls,
        model_name: str,
        display_name: str,
        description: str,
        tool_versions: ToolVersions,
        capabilities: WhisperCapabilities,
        parameters: ModelParameters | None = None,
        assets: ModelAssets | None = None,
        runtime: RuntimeInfo | None = None,
    ) -> "WhisperMetadata":
        """
        Construct a ``WhisperMetadata`` object from the information
        available in a Whisper model.

        Parameters
        ----------
        model_name
            Identifier for the model (e.g. ``whisper_small``).
        display_name
            Human-readable name.
        description
            Short description of the model.
        tool_versions
            tool-version information.
        capabilities
            Model-specific capabilities; must be explicitly provided because
            different Whisper variants have different capability profiles.
        parameters
            Optional parameters information; if omitted a minimal default is used.
        assets
            Optional asset paths; if omitted a minimal default is used.
        runtime
            Optional runtime information; if omitted a minimal default is used.

        Returns
        -------
        WhisperMetadata
            Fully populated metadata instance.
        """
        if parameters is None:
            parameters = ModelParameters()

        if assets is None:
            assets = ModelAssets()

        if runtime is None:
            if tool_versions.qairt is None:
                raise ValueError("QAIRT not configured")
            runtime = RuntimeInfo(
                qnn_version=QNNVersion(
                    major=int(tool_versions.qairt.framework.major),
                    minor=int(tool_versions.qairt.framework.minor),
                    patch=int(
                        tool_versions.qairt.framework.patch
                        if tool_versions.qairt.framework.patch
                        else 0
                    ),
                ),
            )

        return cls(
            name=model_name,
            display_name=display_name,
            description=description,
            capabilities=capabilities,
            parameters=parameters,
            assets=assets,
            runtime=runtime,
        )


# ----------------------------------------------------------------------
# Helper to create metadata and write the JSON file
# ----------------------------------------------------------------------
def create_whisper_metadata(
    model_type: str,
    metadata: ModelMetadata,
    capabilities: WhisperCapabilities,
    param: ModelParameters | None = None,
    display_name: str | None = None,
) -> WhisperMetadata:
    """
    Generate ``WhisperMetadata`` for a Whisper model.

    Parameters
    ----------
    model_type
        Machine-readable model identifier
        (e.g., ``"whisper-small"`` or ``"whisper-small-quantized"``).
    metadata
        ``ModelMetadata`` instance containing model files and tool versions.
    capabilities
        Model-specific capabilities; must be explicitly provided because
        different Whisper variants have different capability profiles.
    param
        model parameters.
    display_name
        Human-readable label shown in UIs (e.g., ``"Whisper Small (Quantized)"``).
        If omitted, a title-cased version of ``model_type`` is used as a fallback
        (e.g., ``"whisper-small-quantized"`` → ``"Whisper Small Quantized"``).

    Returns
    -------
    WhisperMetadata
        The generated Whisper metadata object.
    """
    assets = ModelAssets()
    for file_name in metadata.model_files:
        lower = file_name.lower()
        if "encoder" in lower:
            assets.encoder_path = file_name
        elif "decoder" in lower:
            assets.decoder_path = file_name

    assets.vocab_path = "vocab.bin"

    model_name = model_type
    # Use the provided display_name, or fall back to a title-cased conversion
    # of model_type so the output is at least human-readable.
    display_name = display_name or model_type.replace("-", " ").title()
    description = f"OpenAI {model_type} model for general speech recognition"

    return WhisperMetadata.from_whisper_model(
        model_name=model_name,
        display_name=display_name,
        description=description,
        tool_versions=metadata.tool_versions,
        capabilities=capabilities,
        parameters=param,
        assets=assets,
    )

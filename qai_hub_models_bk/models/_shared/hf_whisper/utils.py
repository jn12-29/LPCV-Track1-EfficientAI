# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import base64
import os
import shutil
from pathlib import Path

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models._shared.hf_whisper.model import (
    SAMPLE_RATE,
    VOCAB_BIN_NAME,
)
from qai_hub_models.models._shared.hf_whisper.whisper_metadata_json import (
    ModelParameters,
    WhisperCapabilities,
    create_whisper_metadata,
)
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset


def write_whisper_supplementary_files(
    output_dir: str | os.PathLike,
    metadata: ModelMetadata,
    model_type: str,
    capabilities: WhisperCapabilities,
    tiktoken_asset: CachedWebModelAsset,
    display_name: str | None = None,
) -> None:
    """
    Write supplementary files for HfWhisper-based models.

    Generates vocab.bin from the provided tiktoken asset and writes
    config.json with model metadata, then registers both files in
    metadata.supplementary_files.

    Parameters
    ----------
    output_dir
        Directory to write supplementary files to
    metadata
        Model metadata object to update with supplementary file info.
        Files are only written when runtime is VOICE_AI.
    model_type
        Machine-readable model identifier used in the config.json
        (e.g. "whisper-small" or "whisper-small-quantized").
    capabilities
        Model-specific capabilities; must be explicitly provided because
        different Whisper variants have different capability profiles.
    tiktoken_asset
        Model-specific tiktoken asset. Multilingual Whisper models use
        ``TIKTOKEN_URL`` from ``model.py``; English-only variants
        (e.g. whisper-small.en) use a different asset.
    display_name
        Human-readable label shown in UIs (e.g., "Whisper Small (Quantized)").
        If omitted, a title-cased version of model_type is used as a fallback.
    """
    if metadata.runtime != TargetRuntime.VOICE_AI:
        return

    # Copy the tiktoken asset to the output directory first, then process the
    # local copy.  Reading directly from the cached path and writing to
    # vocab.bin in the same directory would corrupt the cache if the two paths
    # ever resolve to the same file, and would produce wrong output on a second
    # run if vocab.bin were accidentally used as the source.
    whisper_tiktoken = tiktoken_asset.fetch()
    local_tiktoken = Path(output_dir) / Path(whisper_tiktoken).name
    shutil.copy2(whisper_tiktoken, local_tiktoken)

    with open(local_tiktoken, "rb") as f:
        lines = f.readlines()

    with open(Path(output_dir) / VOCAB_BIN_NAME, "wb") as f:
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            token = base64.b64decode(parts[0])
            f.write(token)
            if b"\0" not in token:
                f.write(b"\0")

    # Remove the temporary tiktoken copy; only vocab.bin should remain
    local_tiktoken.unlink(missing_ok=True)

    # Register vocab.bin in supplementary files
    metadata.supplementary_files[VOCAB_BIN_NAME] = (
        "Whisper vocabulary binary for converting model output token IDs to token strings"
    )

    # Write config.json and register it in supplementary files
    param = ModelParameters(
        max_file_size_mb=25, supported_formats=["wav"], sample_rates=[SAMPLE_RATE]
    )
    whisper_metadata = create_whisper_metadata(
        model_type, metadata, capabilities, param, display_name
    )
    whisper_metadata_path = Path(output_dir) / "config.json"
    whisper_metadata.to_json(whisper_metadata_path, exclude_defaults=False)
    metadata.supplementary_files[whisper_metadata_path.name] = (
        f"{model_type} metadata JSON specifically for the VoiceAI runtime"
    )

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

# flake8: noqa: E402
# (module level import not at top of file)

from __future__ import annotations

import os
from pathlib import Path

# isort: off
# This verifies aimet is installed, and this must be included first.
MODEL_ID = __name__.split(".")[-2]
from qai_hub_models.utils.quantization_aimet_onnx import ensure_aimet_onnx_installed

ensure_aimet_onnx_installed(model_id=MODEL_ID)
# isort: on

import torch
from transformers import WhisperConfig
from typing_extensions import Self

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models._shared.hf_whisper.model import (
    TIKTOKEN_URL,
)
from qai_hub_models.models._shared.hf_whisper.utils import (
    write_whisper_supplementary_files,
)
from qai_hub_models.models._shared.hf_whisper.whisper_metadata_json import (
    WhisperCapabilities,
)
from qai_hub_models.models._shared.hf_whisper_quantized.model import (
    WhisperDecoderQuantizableBase,
    WhisperEncoderQuantizableBase,
)
from qai_hub_models.models.whisper_small.model import WhisperSmall
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import CollectionModel, PretrainedCollectionModel

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
WHISPER_VERSION = "openai/whisper-small"
ENCODER_AIMET = "encoder.aimet"
DECODER_AIMET = "decoder.aimet"

# Explicit capability profile for whisper-small-quantized.
# Declared here (not as class-level defaults) so that each model variant
# owns its own capability declaration.
WHISPER_SMALL_QUANTIZED_CAPABILITIES = WhisperCapabilities(
    streaming=True,
    file_based=True,
    real_time=True,
    language_detection=True,
    confidence_scores=False,
)


class WhisperSmallEncoderQuantizable(WhisperEncoderQuantizableBase):
    @classmethod
    def make_torch_model(cls) -> torch.nn.Module:
        return WhisperSmall.from_pretrained().encoder

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[Path, Path]:
        # Returns .onnx and .encodings paths
        onnx_file = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(ENCODER_AIMET, "model.onnx"),
        ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(ENCODER_AIMET, "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings


class WhisperSmallDecoderQuantizable(WhisperDecoderQuantizableBase):
    @classmethod
    def make_torch_model(cls) -> torch.nn.Module:
        return WhisperSmall.from_pretrained().decoder

    @classmethod
    def get_calibrated_aimet_model(cls) -> tuple[Path, Path]:
        # Returns .onnx and .encodings paths
        onnx_file = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(DECODER_AIMET, "model.onnx"),
        ).fetch()
        aimet_encodings = CachedWebModelAsset.from_asset_store(
            MODEL_ID,
            MODEL_ASSET_VERSION,
            os.path.join(DECODER_AIMET, "model.encodings"),
        ).fetch()
        return onnx_file, aimet_encodings


@CollectionModel.add_component(WhisperSmallEncoderQuantizable, "encoder")
@CollectionModel.add_component(WhisperSmallDecoderQuantizable, "decoder")
class WhisperSmallQuantized(PretrainedCollectionModel):
    def __init__(
        self,
        encoder: WhisperEncoderQuantizableBase,
        decoder: WhisperDecoderQuantizableBase,
        config: WhisperConfig,
        hf_source: str,
    ) -> None:
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.hf_source = hf_source

    @classmethod
    def get_hf_whisper_version(cls) -> str:
        return WHISPER_VERSION

    @classmethod
    def from_pretrained(cls) -> Self:
        encoder = WhisperSmallEncoderQuantizable.from_pretrained()
        decoder = WhisperSmallDecoderQuantizable.from_pretrained()
        return cls(encoder, decoder, encoder.config, WHISPER_VERSION)

    def write_supplementary_files(
        self, output_dir: str | os.PathLike, metadata: ModelMetadata
    ) -> None:
        write_whisper_supplementary_files(
            output_dir,
            metadata,
            "whisper-small-quantized",
            WHISPER_SMALL_QUANTIZED_CAPABILITIES,
            TIKTOKEN_URL,
            display_name="Whisper Small (Quantized)",
        )

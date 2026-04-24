# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

from qai_hub import Device
from transformers import WhisperForConditionalGeneration

from qai_hub_models.models._shared.hf_whisper.model import (
    MASK_NEG,
    HfWhisper,
    HfWhisperDecoder,
    HfWhisperEncoder,
    InputSpec,
)
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.base_model import CollectionModel

from .model_patch import monkey_patch_distil_whisper_model

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DISTIL_WHISPER_VERSION = "distil-whisper/distil-small.en"


class DistilWhisperEncoder(HfWhisperEncoder):
    @classmethod
    def from_pretrained(cls) -> DistilWhisperEncoder:
        model = DistilWhisper.load_whisper_model()
        return cls(model.config, model.get_encoder())

    @staticmethod
    def get_input_spec(num_mel_bin: int = 80) -> InputSpec:
        return HfWhisperEncoder.get_input_spec(num_mel_bin)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super(HfWhisperEncoder, self).get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors --truncate_64bit_io"
        return compile_options


class DistilWhisperDecoder(HfWhisperDecoder):
    @classmethod
    def from_pretrained(cls) -> DistilWhisperDecoder:
        model = DistilWhisper.load_whisper_model()
        return cls(model.config, model.get_decoder())

    @staticmethod
    def get_input_spec(
        num_blocks: int = 4,
        attention_dim: int = 768,
        num_heads: int = 12,
    ) -> InputSpec:
        return HfWhisperDecoder.get_input_spec(num_blocks, attention_dim, num_heads)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super(HfWhisperDecoder, self).get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors --truncate_64bit_io"
        return compile_options


@CollectionModel.add_component(DistilWhisperEncoder, "encoder")
@CollectionModel.add_component(DistilWhisperDecoder, "decoder")
class DistilWhisper(HfWhisper):
    @classmethod
    def get_hf_whisper_version(cls) -> str:
        return DISTIL_WHISPER_VERSION

    @classmethod
    def from_pretrained(cls) -> DistilWhisper:
        whisper = cls.load_whisper_model()
        config = whisper.config
        encoder = DistilWhisperEncoder(config, whisper.get_encoder())
        decoder = DistilWhisperDecoder(config, whisper.get_decoder())
        return cls(encoder, decoder, config, cls.get_hf_whisper_version())

    @classmethod
    def load_whisper_model(
        cls, hf_whisper_version: str | None = None
    ) -> WhisperForConditionalGeneration:
        hf_whisper_version = (
            cls.get_hf_whisper_version()
            if hf_whisper_version is None
            else hf_whisper_version
        )
        orig_whisper = cast(
            WhisperForConditionalGeneration,
            WhisperForConditionalGeneration.from_pretrained(hf_whisper_version),
        )
        orig_whisper.config.return_dict = False
        orig_whisper.config.tie_word_embeddings = False
        orig_whisper.config.mask_neg = MASK_NEG
        monkey_patch_distil_whisper_model(orig_whisper.model)
        return cast(WhisperForConditionalGeneration, orig_whisper)

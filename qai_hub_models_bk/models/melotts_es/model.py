# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import copy
import os

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models._shared.melotts.model import (
    MAX_NUM_INPUT_IDS,
    Decoder,
    Encoder,
    Flow,
    T5Decoder,
    T5Encoder,
    get_t5model,
    get_tts_object,
)
from qai_hub_models.models._shared.melotts.utils import (
    write_melotts_supplementary_files,
)
from qai_hub_models.utils.base_model import CollectionModel, PretrainedCollectionModel

MODEL_ID = __name__.split(".")[-2]


class Encoder_ES(Encoder):
    @classmethod
    def from_pretrained(cls) -> "Encoder_ES":
        return cls(get_tts_object("SPANISH"), speed_adjustment=0.85)


class Flow_ES(Flow):
    @classmethod
    def from_pretrained(cls) -> "Flow_ES":
        return cls(get_tts_object("SPANISH"))


class Decoder_ES(Decoder):
    @classmethod
    def from_pretrained(cls) -> "Decoder_ES":
        return cls(get_tts_object("SPANISH"))


class T5Encoder_ES(T5Encoder):
    @classmethod
    def from_pretrained(cls) -> "T5Encoder_ES":
        return cls(get_t5model())


class T5Decoder_ES(T5Decoder):
    @classmethod
    def from_pretrained(cls) -> "T5Decoder_ES":
        # here the t5model is passed to T5Decoder by reference
        # use deepcopy to prevent cached t5model being modified, so the cache can be reused
        return cls(copy.deepcopy(get_t5model()), MAX_NUM_INPUT_IDS)


@CollectionModel.add_component(Encoder_ES, "encoder")
@CollectionModel.add_component(Flow_ES, "flow")
@CollectionModel.add_component(Decoder_ES, "decoder")
@CollectionModel.add_component(T5Encoder_ES, "t5_encoder")
@CollectionModel.add_component(T5Decoder_ES, "t5_decoder")
class MeloTTS_ES(PretrainedCollectionModel):
    def __init__(
        self,
        encoder: Encoder,
        flow: Flow,
        decoder: Decoder,
        charsiu_encoder: T5Encoder,
        charsiu_decoder: T5Decoder,
    ) -> None:
        super().__init__(encoder, flow, decoder, charsiu_encoder, charsiu_decoder)
        self.encoder = encoder
        self.flow = flow
        self.decoder = decoder
        self.charsiu_encoder = charsiu_encoder
        self.charsiu_decoder = charsiu_decoder
        self.speaker_id = encoder.speaker_id
        self.tts_object = get_tts_object("SPANISH")

    @classmethod
    def language(cls) -> str:
        return "SPANISH"

    @classmethod
    def from_pretrained(cls) -> "MeloTTS_ES":
        return cls(
            Encoder_ES.from_pretrained(),
            Flow_ES.from_pretrained(),
            Decoder_ES.from_pretrained(),
            T5Encoder_ES.from_pretrained(),
            T5Decoder_ES.from_pretrained(),
        )

    def write_supplementary_files(
        self, output_dir: str | os.PathLike, metadata: ModelMetadata
    ) -> None:
        write_melotts_supplementary_files("SPANISH", output_dir, metadata)

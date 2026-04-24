# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models._shared.melotts.model import (
    BertWrapper,
    Decoder,
    Encoder,
    Flow,
    get_bert_model,
    get_tts_object,
)
from qai_hub_models.models._shared.melotts.utils import (
    write_melotts_supplementary_files,
)
from qai_hub_models.utils.base_model import CollectionModel, PretrainedCollectionModel

MODEL_ID = __name__.split(".")[-2]


class Encoder_ZH(Encoder):
    @classmethod
    def from_pretrained(cls) -> "Encoder_ZH":
        return cls(get_tts_object("CHINESE"), speed_adjustment=0.85)


class Flow_ZH(Flow):
    @classmethod
    def from_pretrained(cls) -> "Flow_ZH":
        return cls(get_tts_object("CHINESE"))


class Decoder_ZH(Decoder):
    @classmethod
    def from_pretrained(cls) -> "Decoder_ZH":
        return cls(get_tts_object("CHINESE"))


class BertWrapper_ZH(BertWrapper):
    @classmethod
    def from_pretrained(cls) -> "BertWrapper_ZH":
        return cls(get_bert_model("CHINESE"))


@CollectionModel.add_component(Encoder_ZH, "encoder")
@CollectionModel.add_component(Flow_ZH, "flow")
@CollectionModel.add_component(Decoder_ZH, "decoder")
@CollectionModel.add_component(BertWrapper_ZH, "bert_wrapper")
class MeloTTS_ZH(PretrainedCollectionModel):
    def __init__(
        self, encoder: Encoder, flow: Flow, decoder: Decoder, bert_model: BertWrapper
    ) -> None:
        super().__init__(encoder, flow, decoder, bert_model)
        self.encoder = encoder
        self.flow = flow
        self.decoder = decoder
        self.speaker_id = encoder.speaker_id
        self.tts_object = get_tts_object("CHINESE")
        self.bert_model = bert_model

    @classmethod
    def language(cls) -> str:
        return "CHINESE"

    @classmethod
    def from_pretrained(cls) -> "MeloTTS_ZH":
        return cls(
            Encoder_ZH.from_pretrained(),
            Flow_ZH.from_pretrained(),
            Decoder_ZH.from_pretrained(),
            BertWrapper_ZH.from_pretrained(),
        )

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        write_melotts_supplementary_files("CHINESE", output_dir, metadata)

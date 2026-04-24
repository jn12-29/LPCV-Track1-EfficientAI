# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp
from qai_hub_models.models._shared.hf_whisper.test_utils import (
    load_sample_audio_input,
    run_test_wrapper_numerics,
)
from qai_hub_models.models.distil_whisper.demo import main as demo_main
from qai_hub_models.models.distil_whisper.model import (
    DistilWhisper,
)


def test_numerics() -> None:
    run_test_wrapper_numerics(DistilWhisper)


def test_transcribe() -> None:
    # The original HF model generate() produces an incorrect truncated text for this audio.
    # so, we directly check against the correct transcription.
    model = DistilWhisper.from_pretrained()
    hf_whisper_version = DistilWhisper.get_hf_whisper_version()
    app = HfWhisperApp(model.encoder, model.decoder, hf_whisper_version)

    audio, _, sample_rate = load_sample_audio_input(app, hf_whisper_version)
    transcription = app.transcribe(audio, sample_rate)

    expected_text = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
    assert transcription == expected_text


def test_demo() -> None:
    demo_main(is_test=True)

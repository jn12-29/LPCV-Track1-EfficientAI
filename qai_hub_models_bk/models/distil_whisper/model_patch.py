# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoder,
    WhisperEncoder,
    WhisperModel,
)

from qai_hub_models.models._shared.hf_whisper.model_adaptation import (
    QcWhisperDecoder,
    QcWhisperEncoder,
    SHAAttention,
)

# Scales KV cache values preventing FP16 underflow. Empirically chosen to be within FP16 range.
KV_CACHE_SCALE = 100.0


class ScaledSHAAttention(SHAAttention):
    """
    SHAAttention with scaled KV cache storage.
    Values are scaled up before storage and scaled down after retrieval.
    This helps preserve precision for small values in FP16.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # If we have past_key_value, we need to unscale it before use
        unscaled_past_key_value = None
        if past_key_value is not None:
            # past_key_value is (key, value)
            past_kv = cast(tuple[torch.Tensor, torch.Tensor], past_key_value)
            # Unscale for computation
            unscaled_past_key_value = cast(
                tuple[torch.Tensor],
                (
                    past_kv[0] / KV_CACHE_SCALE,
                    past_kv[1] / KV_CACHE_SCALE,
                ),
            )

        attn_output, past_key_value_rt = super().forward(
            hidden_states, unscaled_past_key_value, attention_mask
        )

        # Scale the new KV cache before returning/storing
        if past_key_value_rt is not None:
            past_key_value_rt = (
                past_key_value_rt[0] * KV_CACHE_SCALE,
                past_key_value_rt[1] * KV_CACHE_SCALE,
            )

        return attn_output, past_key_value_rt


class ScaledQcWhisperDecoder(QcWhisperDecoder):
    """Decoder that replaces all attention modules with ScaledSHAAttention."""

    def __init__(self, orig_decoder: WhisperDecoder) -> None:
        super().__init__(orig_decoder)
        for orig_layer, layer in zip(orig_decoder.layers, self.layers, strict=True):
            layer.self_attn = ScaledSHAAttention(
                cast(WhisperAttention, orig_layer.self_attn)
            )
            layer.encoder_attn = ScaledSHAAttention(
                cast(WhisperAttention, orig_layer.encoder_attn)
            )


class ScaledQcWhisperEncoder(QcWhisperEncoder):
    """
    Encoder that scales KV cache outputs by KV_CACHE_SCALE for FP16 precision.
    Values are scaled up before output and scaled down in the decoder's
    cross-attention before use.
    """

    def forward(
        self,
        input_features: torch.Tensor,
    ) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...],]:
        (next_cache,) = super().forward(input_features)
        assert next_cache is not None
        scaled_cache = tuple(
            (k * KV_CACHE_SCALE, v * KV_CACHE_SCALE) for k, v in next_cache
        )
        return (scaled_cache,)


def replace_decoder_with_scaled(model: WhisperModel) -> None:
    """Replaces the decoder in the Whisper model with ScaledQcWhisperDecoder."""
    orig_decoder_module = WhisperDecoder
    qc_decoder_module = ScaledQcWhisperDecoder
    get_module = model.get_decoder
    module = get_module()

    for name, submodule in model.named_children():
        if isinstance(submodule, orig_decoder_module):
            setattr(model, name, qc_decoder_module(module))


def replace_encoder_with_scaled(model: WhisperModel) -> None:
    """
    Replaces the encoder in the Whisper model with ScaledQcWhisperEncoder.
    The decoder must already be a QcWhisperDecoder (or subclass) before calling.
    """
    qc_decoder = model.get_decoder()
    if not isinstance(qc_decoder, QcWhisperDecoder):
        raise TypeError(
            "Please update the decoder to QcWhisperDecoder before updating the encoder."
        )
    orig_encoder = model.get_encoder()
    for name, submodule in model.named_children():
        if isinstance(submodule, WhisperEncoder):
            setattr(model, name, ScaledQcWhisperEncoder(orig_encoder, qc_decoder))


def monkey_patch_distil_whisper_model(model: WhisperModel) -> None:
    """
    Applies modifications to Distil-Whisper model:
    1. Replaces decoder with scaled version (for KV cache precision)
    2. Replaces encoder with scaled version (for KV cache precision)
    """
    # Replace decoder with ScaledQcWhisperDecoder
    replace_decoder_with_scaled(model)

    # Replace encoder with ScaledQcWhisperEncoder
    replace_encoder_with_scaled(model)

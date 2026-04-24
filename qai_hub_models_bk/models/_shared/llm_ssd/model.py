# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np
import qai_hub as hub
import torch

from qai_hub_models.models._shared.llm.model import LLM_AIMETOnnx, LLMBase
from qai_hub_models.utils.base_model import Precision

with contextlib.suppress(ImportError):
    from transformers import PretrainedConfig

GENIE_CONFIG_JSON = "genie_config.json"


def _quantize_kv_cache(f: Any, encoding: Any, bw: int = 8) -> Any:
    def _round(x: Any) -> Any:
        sign = np.where(x < 0, -1, 1).astype(np.float32)
        return np.floor(np.abs(x) + 0.5) * sign

    def _quantize(f: Any, scale: Any, offset: Any, dtype: np.dtype) -> Any:
        q = _round(f / scale - offset)
        return q.clip(np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)

    if isinstance(encoding, list):
        scale, offset = encoding[0]["scale"], encoding[0]["offset"]
        assert encoding[0]["bitwidth"] == bw
    elif isinstance(encoding, dict):
        scale, offset = encoding["scale"][0], encoding["offset"][0]
        assert encoding["bw"] == bw
    else:
        raise TypeError(f"Unknown encoding format: {type(encoding)}")

    f = np.array(f)
    _BW_TO_DTYPE: dict[int, np.dtype[Any]] = {
        8: np.dtype(np.uint8),
        16: np.dtype(np.uint16),
        32: np.dtype(np.uint32),
        64: np.dtype(np.uint64),
    }
    if bw not in _BW_TO_DTYPE:
        raise ValueError(
            f"Unsupported bitwidth: {bw}. Supported: {list(_BW_TO_DTYPE.keys())}"
        )
    bw_dtype = _BW_TO_DTYPE[bw]
    return _quantize(f, scale, offset, bw_dtype)


def _save_kv_cache(
    kvcache: Any, encodings: Any, filename: str, num_layers: int = 10000
) -> None:
    key_value_encodings = [
        [encodings[f"past_key_{layer_n}_in"], encodings[f"past_value_{layer_n}_in"]]
        for layer_n in range(num_layers)
    ]
    key_q = [
        _quantize_kv_cache(cache[0], encoding[0])
        for cache, encoding in zip(kvcache, key_value_encodings, strict=False)
    ]
    value_q = [
        _quantize_kv_cache(cache[1], encoding[1])
        for cache, encoding in zip(kvcache, key_value_encodings, strict=False)
    ]

    key_cache = np.concatenate(key_q)
    value_cache = np.concatenate(value_q)

    CACHE_FILE_SPEC = "IIBxHHH"
    CACHE_FILE_SPEC_SIZE = struct.calcsize(CACHE_FILE_SPEC)
    assert CACHE_FILE_SPEC_SIZE == 16
    DATATYPES = [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        None,
        np.float16,
        np.float32,
        np.float64,
        bool,
    ]

    _DTYPE_TO_ID = {np.dtype(t): i for i, t in enumerate(DATATYPES) if t is not None}
    with open(filename, "wb") as handle:
        dtype = _DTYPE_TO_ID.get(key_cache.dtype)
        if dtype is None:
            raise ValueError(
                f"Unsupported cache dtype: {key_cache.dtype}. "
                f"Supported: {list(_DTYPE_TO_ID.keys())}"
            )
        n_layer, n_head, n_tok, n_kv_dim = value_cache.shape
        num_tensors = n_layer * 2
        handle.write(
            struct.pack(
                CACHE_FILE_SPEC, num_tensors, 0xC0DE, dtype, n_head, n_kv_dim, n_tok
            )
        )
        key_cache.tofile(handle)
        value_cache.tofile(handle)


class LLM_SSD_Base(LLMBase):
    """Extends LLMBase with SSD (Self Speculative Decoding) forecast support."""

    def __init__(
        self,
        *args: Any,
        ssd_forecast_ckpt: str | os.PathLike | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        *args
            Positional arguments forwarded to LLMBase.
        ssd_forecast_ckpt
            Path to SSD forecast file. If provided, the SSD forecast token
            embeddings are concatenated to the model's embedding table.
        **kwargs
            Keyword arguments forwarded to LLMBase.
        """
        super().__init__(*args, **kwargs)
        if ssd_forecast_ckpt is not None:
            ssd_param = torch.load(
                ssd_forecast_ckpt, map_location="cpu", weights_only=True
            )
            ssd_forecast_embeddings = ssd_param["forecast_embedding"]
            if len(ssd_forecast_embeddings) >= 1:
                embed_table = cast(torch.nn.Embedding, self.model.model.embed_tokens)  # type: ignore[union-attr, unused-ignore]
                assert (
                    embed_table.weight.shape[1] == ssd_forecast_embeddings.shape[1]
                ), "Mismatching token embedding size for embed_tokens"
                embed_table.weight.data = torch.cat(
                    [
                        embed_table.weight.data,
                        ssd_forecast_embeddings.to(embed_table.weight.dtype),
                    ],
                    dim=0,
                )
                embed_table.num_embeddings = embed_table.weight.shape[0]


class LLM_SSD_AIMETOnnx(LLM_AIMETOnnx):
    """Extends LLM_AIMETOnnx with SSD (Self Speculative Decoding) support."""

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
        super().prepare_genie_assets(
            hub_device,
            checkpoint,
            llm_config,
            context_lengths,
            model_list,
            output_path,
            precision,
            encodings_path,
            input_specs,
            output_specs,
            model_id=model_id,
            model_name=model_name,
        )
        if cls.FPModel is None or not hasattr(cls.FPModel, "_ssd_forecast_ckpt"):
            return
        ssd_forecast_ckpt = cls.FPModel._ssd_forecast_ckpt()
        if ssd_forecast_ckpt is None:
            return

        # Load SSD params once
        ssd_param = torch.load(ssd_forecast_ckpt, map_location="cpu", weights_only=True)
        ssd_prefix = ssd_param["forecast_prefix"].to(torch.float32)
        n_layer, _, _, _, len_prefix, _ = ssd_prefix.shape
        ssd_prefix_tuple = tuple(
            (ssd_prefix[i][0].permute(0, 1, 3, 2), ssd_prefix[i][1])
            for i in range(n_layer)
        )
        num_ssd_forecast_tokens = len(ssd_param["forecast_embedding"])

        # Load activation_encodings (to scan for all 'past_key_*_in' layers)
        with open(encodings_path) as f:
            encodings = json.load(f)
        if isinstance(encodings["activation_encodings"], list):
            # Convert encodings to dictionary
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
        actv_encodings = encodings["activation_encodings"]
        num_layers = sum(
            1
            for ae_key in actv_encodings
            if ae_key.startswith("past_value_") and ae_key.endswith("_in")
        )

        # Create 'forecast-prefix' folder and save kvcache prefix
        ssd_prefix_des_dir = output_path / "forecast-prefix"
        shutil.rmtree(ssd_prefix_des_dir, ignore_errors=True)
        ssd_prefix_des_dir.mkdir(parents=True, exist_ok=True)
        _save_kv_cache(
            ssd_prefix_tuple,
            actv_encodings,
            str(ssd_prefix_des_dir / "kv-cache.primary.qnn-htp"),
            num_layers,
        )

        # Update genie config with SSD params
        with open(output_path / GENIE_CONFIG_JSON) as f:
            genie_config = json.load(f)
        genie_config["dialog"]["type"] = "ssd-q1"
        genie_config["dialog"]["ssd-q1"] = {
            "version": 1,
            "ssd-version": 1,
            "forecast-token-count": num_ssd_forecast_tokens,
            "forecast-prefix": len_prefix,
            "forecast-prefix-name": ssd_prefix_des_dir.name,
            "branches": [3, 2],
            "n-streams": 1,
            "p-threshold": 0.0,
        }
        with open(output_path / GENIE_CONFIG_JSON, "w") as f:
            json.dump(genie_config, f, indent=4)

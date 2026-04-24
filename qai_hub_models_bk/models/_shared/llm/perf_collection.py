# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Shared utilities for LLM performance collection.

Provides:
- LLMPerfConfig: env-var driven configuration dataclass
- get_llm_perf_parametrization: generates (precision, device) pytest params
- update_perf_yaml: writes TPS/TTFT metrics into a model's perf.yaml

The compile/QDC test logic lives in _shared/llm/test.py (run_llm_perf_test).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.scorecard.device import sanitize_chipset_name
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath


@dataclass
class LLMPerfConfig:
    """Configuration for LLM performance collection.

    Loads configuration from environment variables:
    - QAIHM_LLM_MODELS: Comma-separated model IDs or "all"
    - QAIHM_TEST_DEVICES: Comma-separated device names
    - LLM_CONTEXT_LENGTH: Comma-separated context length(s) (default: 4096)
    - LLM_SEQUENCE_LENGTH: Comma-separated sequence length(s) (default: 128,1)
    - SKIP_PERF_UPDATE: If set, skip updating perf.yaml files
    - QAIRT_SDK_PATH: Path to QAIRT SDK for auto devices
    """

    models: list[str] = field(default_factory=list)
    devices: list[str] = field(default_factory=list)
    export_context_lengths: list[int] | None = None
    export_sequence_lengths: list[int] | None = None
    skip_perf_update: bool = False
    qairt_sdk_path: str | None = None

    @classmethod
    def from_environment(cls) -> LLMPerfConfig:
        """Create config from environment variables."""
        models_str = os.environ.get("QAIHM_LLM_MODELS", "")
        devices_str = os.environ.get("QAIHM_TEST_DEVICES", "")

        models = [m.strip() for m in models_str.split(",") if m.strip()]
        devices = [d.strip() for d in devices_str.split(",") if d.strip()]

        raw_cl = os.environ.get("LLM_CONTEXT_LENGTH")
        raw_sl = os.environ.get("LLM_SEQUENCE_LENGTH")

        return cls(
            models=models,
            devices=devices,
            export_context_lengths=(
                [int(x) for x in raw_cl.split(",")] if raw_cl else None
            ),
            export_sequence_lengths=(
                [int(x) for x in raw_sl.split(",")] if raw_sl else None
            ),
            skip_perf_update=bool(os.environ.get("SKIP_PERF_UPDATE")),
            qairt_sdk_path=os.environ.get("QAIRT_SDK_PATH"),
        )


def get_supported_precisions(model_id: str) -> list[Precision]:
    """Get the supported precisions for a model from code-gen.yaml."""
    code_gen = QAIHMModelCodeGen.from_model(model_id)
    return code_gen.supported_precisions


def get_llm_perf_parametrization(
    model_id: str,
    default_devices: list[ScorecardDevice] | None = None,
    default_precisions: list[Precision] | None = None,
) -> list[tuple[Precision, ScorecardDevice]]:
    """Generate pytest parametrization for LLM performance tests.

    Uses environment variables if set, otherwise falls back to defaults.

    Environment variables:
    - QAIHM_LLM_MODELS: Comma-separated model IDs or "all". If set and this
      model is not in the list, returns [] so the test is skipped.
    - QAIHM_TEST_DEVICES: Comma-separated device names to override defaults
    - QAIHM_TEST_PRECISIONS: Comma-separated precisions to override defaults
    """
    models_str = os.environ.get("QAIHM_LLM_MODELS", "")
    if models_str and models_str.strip().lower() != "all":
        allowed = [m.strip() for m in models_str.split(",") if m.strip()]
        if model_id not in allowed:
            return []

    devices_str = os.environ.get("QAIHM_TEST_DEVICES", "")
    if devices_str:
        device_names = [d.strip() for d in devices_str.split(",") if d.strip()]
        devices = [
            ScorecardDevice._registry[name]
            for name in device_names
            if name in ScorecardDevice._registry
        ]
    else:
        devices = default_devices or []

    precisions = get_supported_precisions(model_id)

    return [(precision, device) for precision in precisions for device in devices]


def update_perf_yaml(
    model_id: str,
    device_name: str,
    precision: Precision,
    context_length: int,
    tps: float,
    ttft_ms: float,
) -> None:
    """Update the perf.yaml file for a model with new LLM metrics."""
    perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)

    info = QAIHMModelInfo.from_model(model_id)
    component_name = info.name

    device = ScorecardDevice.get(device_name, return_unregistered=True)
    if device not in perf.supported_devices:
        perf.supported_devices.append(device)

    chipset = sanitize_chipset_name(device.chipset)
    if chipset not in perf.supported_chipsets:
        perf.supported_chipsets.append(chipset)

    if precision not in perf.precisions:
        perf.precisions[precision] = QAIHMModelPerf.PrecisionDetails()

    precision_details = perf.precisions[precision]

    if component_name not in precision_details.components:
        precision_details.components[component_name] = QAIHMModelPerf.ComponentDetails()

    component_details = precision_details.components[component_name]

    if device not in component_details.performance_metrics:
        component_details.performance_metrics[device] = {}

    device_metrics = component_details.performance_metrics[device]
    genie_path = ScorecardProfilePath.GENIE

    if genie_path not in device_metrics:
        device_metrics[genie_path] = QAIHMModelPerf.PerformanceDetails()

    perf_details = device_metrics[genie_path]

    # Max TTFT is estimated assuming linear scaling with prompt length.
    estimated_max_ttft_ms = ttft_ms * (context_length / 128)
    llm_metric = QAIHMModelPerf.PerformanceDetails.LLMMetricsPerContextLength(
        context_length=context_length,
        tokens_per_second=tps,
        time_to_first_token_range_milliseconds=QAIHMModelPerf.PerformanceDetails.TimeToFirstTokenRangeMilliseconds(
            min=ttft_ms,
            max=estimated_max_ttft_ms,
        ),
    )

    if perf_details.llm_metrics is None:
        perf_details.llm_metrics = [llm_metric]
    else:
        found = False
        for i, existing in enumerate(perf_details.llm_metrics):
            if existing.context_length == context_length:
                perf_details.llm_metrics[i] = llm_metric
                found = True
                break
        if not found:
            perf_details.llm_metrics.append(llm_metric)

    perf.to_model_yaml(model_id)
    print(f"Updated perf.yaml for {model_id}")

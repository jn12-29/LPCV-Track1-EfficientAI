# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from enum import Enum, unique

from typing_extensions import assert_never

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard.envvars import (
    DeploymentEnvvar,
    QAIRTVersionEnvvar,
)
from qai_hub_models.utils.hub_clients import (
    default_hub_client_as,
    get_hub_client_or_raise,
)


@unique
class ScorecardCompilePath(Enum):
    ONNX_FOR_QUANTIZATION = "onnx_for_quantization"  # used only as input to the quantization step, not an actual compile path that produces an asset
    TFLITE = "tflite"
    QNN_DLC = "qnn_dlc"
    QNN_DLC_VIA_QNN_EP = "qnn_dlc_via_qnn_ep"
    QNN_CONTEXT_BINARY = "qnn_context_binary"
    ONNX = "onnx"
    PRECOMPILED_QNN_ONNX = "precompiled_qnn_onnx"
    GENIE = "genie"
    VOICE_AI = "voice_ai"

    ONNX_FP16 = "onnx_fp16"

    LLAMA_CPP_CPU = "llama_cpp_cpu"
    LLAMA_CPP_GPU = "llama_cpp_gpu"
    LLAMA_CPP_NPU = "llama_cpp_npu"

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def enabled(self) -> bool:
        from qai_hub_models.scorecard.path_profile import ScorecardProfilePath

        profile_paths = [
            x for x in ScorecardProfilePath if x.enabled and x.compile_path == self
        ]
        return len(profile_paths) > 0

    def should_run_path_for_model(
        self,
        precision: Precision,
        model_supported_runtimes: dict[Precision, list[TargetRuntime]],
    ) -> bool:
        """
        Whether this compile path should be run for a model with the given
        supported runtimes. Delegates to profile paths that use this compile path.

        Parameters
        ----------
        precision
            The precision to check.
        model_supported_runtimes
            Mapping of precision to supported runtimes for the model.

        Returns
        -------
        should_run : bool
            True if this compile path should be run for the model at the given precision.
        """
        from qai_hub_models.scorecard.path_profile import ScorecardProfilePath

        return any(
            x.compile_path == self
            and x.should_run_path_for_model(precision, model_supported_runtimes)
            for x in ScorecardProfilePath
        )

    @property
    def runtime(self) -> TargetRuntime:
        if self == ScorecardCompilePath.TFLITE:
            return TargetRuntime.TFLITE
        if (
            self == ScorecardCompilePath.ONNX  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardCompilePath.ONNX_FP16
            or self == ScorecardCompilePath.ONNX_FOR_QUANTIZATION
        ):
            return TargetRuntime.ONNX
        if self == ScorecardCompilePath.PRECOMPILED_QNN_ONNX:
            return TargetRuntime.PRECOMPILED_QNN_ONNX
        if self == ScorecardCompilePath.QNN_CONTEXT_BINARY:
            return TargetRuntime.QNN_CONTEXT_BINARY
        if (
            self == ScorecardCompilePath.QNN_DLC  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardCompilePath.QNN_DLC_VIA_QNN_EP
        ):
            return TargetRuntime.QNN_DLC
        if self == ScorecardCompilePath.GENIE:
            return TargetRuntime.GENIE
        if self == ScorecardCompilePath.LLAMA_CPP_CPU:
            return TargetRuntime.LLAMA_CPP_CPU
        if self == ScorecardCompilePath.LLAMA_CPP_GPU:
            return TargetRuntime.LLAMA_CPP_GPU
        if self == ScorecardCompilePath.LLAMA_CPP_NPU:
            return TargetRuntime.LLAMA_CPP_NPU
        if self == ScorecardCompilePath.VOICE_AI:
            return TargetRuntime.VOICE_AI
        assert_never(self)

    @property
    def is_universal(self) -> bool:
        """Whether a single asset produced by this path is applicable to any device."""
        return not self.runtime.is_aot_compiled

    def supports_precision(self, precision: Precision) -> bool:
        if self == ScorecardCompilePath.ONNX_FP16:
            return not precision.has_quantized_activations

        return self.runtime.supports_precision(precision)

    @property
    def has_nonstandard_compile_options(self) -> bool:
        """
        If this path passes additional options beyond what the underlying TargetRuntime
        passes (eg --compute_unit), then it's considered nonstandard.
        """
        return self.value not in TargetRuntime._value2member_map_

    def _get_qairt_version_option(
        self,
        include_default: bool = False,
    ) -> str:
        """
        Resolve the QAIRT version option string for this path.

        When QAIHM_TEST_QAIRT_VERSION is set to a non-default value (e.g. "latest"),
        the explicit version flag is always included. When it is the default,
        the flag is only included if ``include_default`` is True.

        Parameters
        ----------
        include_default
            If True, include the version flag even when using the default QAIRT version.

        Returns
        -------
        option : str
            A string like ``--qairt_version 2.45``, or empty if no override is needed.
        """
        if not self.runtime.qairt_version_changes_compilation:
            return ""

        qairt_version_str = QAIRTVersionEnvvar.get()
        with default_hub_client_as(get_hub_client_or_raise(DeploymentEnvvar.get())):
            qairt_version = QAIRTVersionEnvvar.get_qairt_version(
                self.runtime, qairt_version_str
            )

        if QAIRTVersionEnvvar.is_default(qairt_version_str):
            if include_default:
                return f" {qairt_version.explicit_hub_option}"
            return ""

        return f" {qairt_version.explicit_hub_option}"

    def get_compile_options(
        self,
        include_target_runtime: bool = False,
        include_default_qaihm_qnn_version: bool = False,
    ) -> str:
        out = ""
        if (
            include_target_runtime
            and self.runtime.aihub_target_runtime_flag is not None
        ):
            out += self.runtime.aihub_target_runtime_flag

        out += self._get_qairt_version_option(
            include_default=include_default_qaihm_qnn_version
        )

        if self == ScorecardCompilePath.QNN_DLC_VIA_QNN_EP:
            out = out + " --use_qnn_onnx_ep_converter"

        return out.strip()

    def get_link_options(self) -> str:
        """
        Extra options to pass to the link step.

        Ensures the link step uses the same QAIRT version as the compile step
        when a non-default version is configured (e.g. ``QAIHM_TEST_QAIRT_VERSION=latest``).
        """
        return self._get_qairt_version_option().strip()

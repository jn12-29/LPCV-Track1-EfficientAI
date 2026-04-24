# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Generator
from enum import Enum, EnumMeta, unique
from typing import cast

from typing_extensions import assert_never

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import (
    Precision,
    QAIRTVersion,
    TargetRuntime,
)
from qai_hub_models.scorecard.envvars import (
    DeploymentEnvvar,
    EnabledPathsEnvvar,
    SpecialPathSetting,
)
from qai_hub_models.scorecard.path_compile import (
    QAIRTVersionEnvvar,
    ScorecardCompilePath,
)
from qai_hub_models.utils.base_config import EnumListWithParseableAll
from qai_hub_models.utils.hub_clients import (
    default_hub_client_as,
    get_scorecard_client_or_raise,
)


class ScorecardProfilePathMeta(EnumMeta):
    def __iter__(self) -> Generator[ScorecardProfilePath]:
        return (  # type:ignore[var-annotated]
            cast(ScorecardProfilePath, member) for member in super().__iter__()
        )


@unique
class ScorecardProfilePath(Enum, metaclass=ScorecardProfilePathMeta):
    TFLITE = "tflite"
    QNN_DLC = "qnn_dlc"
    QNN_DLC_VIA_QNN_EP = "qnn_dlc_via_qnn_ep"
    QNN_CONTEXT_BINARY = "qnn_context_binary"
    ONNX = "onnx"
    PRECOMPILED_QNN_ONNX = "precompiled_qnn_onnx"
    GENIE = "genie"
    VOICE_AI = "voice_ai"
    ONNX_DML_GPU = "onnx_dml_gpu"
    QNN_DLC_GPU = "qnn_dlc_gpu"
    LLAMA_CPP_CPU = "llama_cpp_cpu"
    LLAMA_CPP_GPU = "llama_cpp_gpu"
    LLAMA_CPP_NPU = "llama_cpp_npu"

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_runtime(runtime: TargetRuntime) -> ScorecardProfilePath:
        """Get the scorecard path that corresponds with default behavior of the given target runtime."""
        return ScorecardProfilePath(runtime.value)

    @property
    def tool_versions(self) -> ToolVersions:
        """Get the versions (currently enabled on Hub) of each runtime that this scorecard path will use."""
        tool_versions = ToolVersions()
        with default_hub_client_as(
            get_scorecard_client_or_raise(DeploymentEnvvar.get())
        ):
            tool_versions.qairt = QAIRTVersionEnvvar.get_qairt_version(self.runtime)
        return tool_versions

    @staticmethod
    def default_paths() -> list[ScorecardProfilePath]:
        """The list of paths enabled by default for scorecard."""
        return [
            ScorecardProfilePath.TFLITE,
            ScorecardProfilePath.ONNX,
            ScorecardProfilePath.PRECOMPILED_QNN_ONNX,
            ScorecardProfilePath.QNN_DLC,
            ScorecardProfilePath.QNN_CONTEXT_BINARY,
            ScorecardProfilePath.GENIE,
            ScorecardProfilePath.VOICE_AI,
            ScorecardProfilePath.LLAMA_CPP_CPU,
            ScorecardProfilePath.LLAMA_CPP_GPU,
            ScorecardProfilePath.LLAMA_CPP_NPU,
        ]

    @property
    def spreadsheet_name(self) -> str:
        """Returns the name used for the 'runtime' column in the scorecard results spreadsheet."""
        if self in ScorecardProfilePath.default_paths():
            # Maps both precompiled_context_binary and dlc to "qnn",
            # and both onnx and precompiled onnx to "onnx".
            return self.runtime.inference_engine.value
        return self.value

    @property
    def enabled(self) -> bool:
        valid_test_runtimes = EnabledPathsEnvvar.get()

        default_paths = ScorecardProfilePath.default_paths()
        if SpecialPathSetting.DEFAULT in valid_test_runtimes and self in default_paths:
            return True

        # Allows users to set 'qnn' to enable both precompiled and dlc,
        # or to set 'onnx' to enable onnx and precompiled onnx.
        if (
            self in default_paths
            and self.runtime.inference_engine.value in valid_test_runtimes
        ):
            return True

        return self.value in valid_test_runtimes

    def should_run_path_for_model(
        self,
        precision: Precision,
        model_supported_runtimes: dict[Precision, list[TargetRuntime]],
    ) -> bool:
        """
        Whether this path should be run for a model with the given supported
        runtimes at the given precision.

        Resolution order based on what is in the EnabledPathsEnvvar:
        1. "default" keyword: enables default paths whose runtime is in the
           model's supported runtimes.
        2. Explicit path name (e.g. qnn_dlc_via_qnn_ep): enables the path
           if the model supports any runtime with the same inference engine.
        3. Engine prefix (e.g. "qnn"): enables paths whose runtime
           is in the model's supported runtimes.

        Parameters
        ----------
        precision
            The precision to check.
        model_supported_runtimes
            Mapping of precision to supported runtimes for the model.

        Returns
        -------
        should_run : bool
            True if this path should be run for the model at the given precision.
        """
        # supported_runtimes lists all valid paths for this model at the
        # given precision. Paths are only excluded from this list for hard
        # constraints (e.g. timeouts, or unsupported combos like w8a16 on
        # tflite). We intersect requested paths with this list so that even
        # explicitly requested paths are skipped when those constraints apply.
        supported_runtimes = model_supported_runtimes.get(precision, [])
        if not supported_runtimes or not self.supports_precision(precision):
            return False

        valid_test_runtimes = EnabledPathsEnvvar.get()

        # 1. "default": exact runtime match against supported runtimes
        if (
            SpecialPathSetting.DEFAULT in valid_test_runtimes
            and self in self.default_paths()
            and self.runtime in supported_runtimes
        ):
            return True

        # 2. Explicit path name: use inference engine match (looser) so that
        #    e.g. requesting "qnn_dlc_via_qnn_ep" works for a model that only
        #    lists qnn_context_binary (same QNN engine). However, if the
        #    engine itself is excluded (e.g. QNN timed out), no QNN path runs.
        if self.value in valid_test_runtimes:
            supported_engines = {r.inference_engine for r in supported_runtimes}
            return self.runtime.inference_engine in supported_engines

        # 3. Engine prefix: exact runtime match against supported runtimes
        return (
            self.runtime.inference_engine.value in valid_test_runtimes
            and self.runtime in supported_runtimes
        )

    def supports_precision(self, precision: Precision) -> bool:
        """Whether this profile path applies to the given model precision."""
        if self == ScorecardProfilePath.QNN_DLC_GPU:
            return not precision.has_quantized_activations

        return self.compile_path.supports_precision(precision)

    @property
    def is_published(self) -> bool:
        """Whether a path is included in perf.yaml, numerics.yaml, allowed in export scripts, etc."""
        return self in ScorecardProfilePath.default_paths()

    @property
    def runtime(self) -> TargetRuntime:
        if self == ScorecardProfilePath.TFLITE:
            return TargetRuntime.TFLITE
        if (
            self == ScorecardProfilePath.ONNX  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardProfilePath.ONNX_DML_GPU
        ):
            return TargetRuntime.ONNX
        if self == ScorecardProfilePath.PRECOMPILED_QNN_ONNX:
            return TargetRuntime.PRECOMPILED_QNN_ONNX
        if self == ScorecardProfilePath.QNN_CONTEXT_BINARY:
            return TargetRuntime.QNN_CONTEXT_BINARY
        if (
            self == ScorecardProfilePath.QNN_DLC  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardProfilePath.QNN_DLC_GPU
            or self == ScorecardProfilePath.QNN_DLC_VIA_QNN_EP
        ):
            return TargetRuntime.QNN_DLC
        if self == ScorecardProfilePath.GENIE:
            return TargetRuntime.GENIE
        if self == ScorecardProfilePath.VOICE_AI:
            return TargetRuntime.VOICE_AI
        if self == ScorecardProfilePath.LLAMA_CPP_CPU:
            return TargetRuntime.LLAMA_CPP_CPU
        if self == ScorecardProfilePath.LLAMA_CPP_GPU:
            return TargetRuntime.LLAMA_CPP_GPU
        if self == ScorecardProfilePath.LLAMA_CPP_NPU:
            return TargetRuntime.LLAMA_CPP_NPU
        assert_never(self)

    @property
    def compile_path(self) -> ScorecardCompilePath:
        if self == ScorecardProfilePath.TFLITE:
            return ScorecardCompilePath.TFLITE
        if self == ScorecardProfilePath.ONNX:
            return ScorecardCompilePath.ONNX
        if self == ScorecardProfilePath.PRECOMPILED_QNN_ONNX:
            return ScorecardCompilePath.PRECOMPILED_QNN_ONNX
        if self == ScorecardProfilePath.ONNX_DML_GPU:
            return ScorecardCompilePath.ONNX_FP16
        if self == ScorecardProfilePath.QNN_CONTEXT_BINARY:
            return ScorecardCompilePath.QNN_CONTEXT_BINARY
        if (
            self == ScorecardProfilePath.QNN_DLC  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardProfilePath.QNN_DLC_GPU
        ):
            return ScorecardCompilePath.QNN_DLC
        if self == ScorecardProfilePath.QNN_DLC_VIA_QNN_EP:
            return ScorecardCompilePath.QNN_DLC_VIA_QNN_EP
        if self == ScorecardProfilePath.GENIE:
            return ScorecardCompilePath.GENIE
        if self == ScorecardProfilePath.VOICE_AI:
            return ScorecardCompilePath.VOICE_AI
        if self == ScorecardProfilePath.LLAMA_CPP_CPU:
            return ScorecardCompilePath.LLAMA_CPP_CPU
        if self == ScorecardProfilePath.LLAMA_CPP_GPU:
            return ScorecardCompilePath.LLAMA_CPP_GPU
        if self == ScorecardProfilePath.LLAMA_CPP_NPU:
            return ScorecardCompilePath.LLAMA_CPP_NPU
        assert_never(self)

    @property
    def has_nonstandard_profile_options(self) -> bool:
        """
        If this path passes additional options beyond what the underlying TargetRuntime
        passes (eg --compute_unit), then it's considered nonstandard.
        """
        return self.value not in TargetRuntime._value2member_map_

    def get_profile_options(
        self, include_default_qaihm_qnn_version: bool = False
    ) -> str:
        # LLAMA_CPP paths don't run through AI Hub profiling
        if self in [
            ScorecardProfilePath.LLAMA_CPP_CPU,
            ScorecardProfilePath.LLAMA_CPP_GPU,
            ScorecardProfilePath.LLAMA_CPP_NPU,
        ]:
            raise ValueError(
                f"{self.name} does not support AI Hub profiling. "
                "LLAMA_CPP paths use pre-built GGUF models and run outside of scorecard."
            )

        out = ""
        if self == ScorecardProfilePath.ONNX_DML_GPU:
            out = out + " --onnx_execution_providers directml"
        if self == ScorecardProfilePath.QNN_DLC_GPU:
            out = (
                out
                + " --compute_unit gpu --qnn_options default_graph_gpu_precision=FLOAT16"
            )

        qairt_version_str = QAIRTVersionEnvvar.get()
        if QAIRTVersionEnvvar.is_default(qairt_version_str):
            # We typically don't want the default QAIRT version added here if it matches with the AI Hub Models default.
            # This allows the export script (which scorecard relies on) to pass in the default version that users will see when they use the CLI.
            #
            # Static models do need this included explicitly because they don't rely on export scripts.
            if include_default_qaihm_qnn_version:
                # Certain runtimes use their own default version of QAIRT.
                # If the user picks our the qaihm_default tag, we need use the runtime's
                # default QAIRT version instead.
                qairt_version = self.runtime.default_qairt_version
                out = out + f" {qairt_version.explicit_hub_option}"
        else:
            qairt_version = QAIRTVersion(qairt_version_str)

            # The explicit option will always pass `--qairt_version 2.XX`,
            # regardless of whether this is the AI Hub Workbench default.
            #
            # This is useful for tracking what QAIRT version applies for scorecard jobs.
            out = out + f" {qairt_version.explicit_hub_option}"

        return out.strip()

    @property
    def website_runtime_name(self) -> str:
        """The name of the runtime on the website that corresponds to this path."""
        if self in [
            ScorecardProfilePath.VOICE_AI,
            ScorecardProfilePath.LLAMA_CPP_CPU,
            ScorecardProfilePath.LLAMA_CPP_GPU,
            ScorecardProfilePath.LLAMA_CPP_NPU,
        ]:
            return self.value
        return self.runtime.inference_engine.value


class ScorecardProfilePathJITParseableAllList(
    EnumListWithParseableAll[ScorecardProfilePath]
):
    """
    Helper class for parsing.
    If "all" is in the list in the yaml, then it will parse to all JIT
    (on device prepare) profile path values.
    """

    EnumType = ScorecardProfilePath
    ALL = [
        x
        for x in ScorecardProfilePath
        if not x.runtime.is_aot_compiled and not x.runtime.is_orchestrator_runtime
    ]

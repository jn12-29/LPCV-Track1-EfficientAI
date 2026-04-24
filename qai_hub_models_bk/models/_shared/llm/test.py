# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any, cast
from unittest import mock
from unittest.mock import MagicMock, Mock, _patch, patch

import numpy as np
import pytest
import qai_hub as hub

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models._shared.llm.common import cleanup
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS,
    DEFAULT_EXPORT_SEQUENCE_LENGTHS,
    LLM_AIMETOnnx,
    LLMBase,
)
from qai_hub_models.models._shared.llm.quantize import quantize
from qai_hub_models.models._shared.llm.split_onnx_utils.utils import ONNXBundle
from qai_hub_models.models.common import Precision, QAIRTVersion, TargetRuntime
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.testing import patch_qai_hub

GENIE_BUNDLES_ROOT = "genie_bundles"


def _mock_from_pretrained(
    model_cls: type[LLM_AIMETOnnx], context_length: int, sequence_length: int
) -> Mock:
    model = MagicMock()
    model.__signature__ = signature(model_cls.from_pretrained)
    mock_from_pretrained = Mock()
    mock_from_pretrained.__signature__ = signature(model_cls.from_pretrained)
    mock_from_pretrained.return_value = model
    return mock_from_pretrained


def from_bundle_path_patch(bundle_path: str | os.PathLike) -> ONNXBundle:
    return ONNXBundle(
        bundle_path=Path(bundle_path),
        onnx_graph_name="model.onnx",
        aimet_encodings_name="model.encodings",
    )


def split_onnx_patch(*args: Any, num_splits: int, **kwargs: Any) -> list[ONNXBundle]:
    return [from_bundle_path_patch(f"{i}") for i in range(num_splits)]


# reusable patching function
def _create_patches(
    model_cls: type[LLM_AIMETOnnx],
    base_name: str,
    context_length: int,
    sequence_length: int,
    tmp_path: Path,
) -> tuple[
    Mock,
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
]:
    mock_from_pretrained = _mock_from_pretrained(
        model_cls, context_length, sequence_length
    )
    mock_from_pretrained.return_value.sample_inputs.return_value = {
        "input0": [np.array([1.0, 2.0])]
    }
    # patching to not download from huggingface.
    patch_model = patch(
        f"qai_hub_models.models.{base_name}.Model.from_pretrained",
        mock_from_pretrained,
    )

    patch_fp_model = patch(
        f"qai_hub_models.models.{base_name}.FP_Model.from_pretrained",
        return_value=Mock(),
    )

    patch_onnx_checker = patch("onnx.checker.check_model")
    patch_onnx_load = patch("onnx.load")

    patch_split_onnx = patch(
        "qai_hub_models.models._shared.llm.split_onnx_utils.utils.split_onnx",
        side_effect=split_onnx_patch,
    )

    patch_onnx_files = patch.object(
        ONNXBundle, "from_bundle_path", side_effect=from_bundle_path_patch
    )

    patch_get_or_create_cached_model = patch(
        "qai_hub_models.models._shared.llm.export.get_or_create_cached_model",
        return_value=Mock(),
    )
    patch_tool_versions = patch(
        "qai_hub_models.configs.tool_versions.ToolVersions.from_job",
        return_value=ToolVersions(
            qairt=QAIRTVersion("2.34", validate_exists_on_ai_hub=False)
        ),
    )

    return (
        mock_from_pretrained,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    )


def test_cli_device_with_skips_unsupported_precision_device(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
) -> None:
    (
        _,
        patch_model,
        patch_fp_model,
        _,
        _,
        _,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, 4096, 128, tmp_path)

    os.makedirs("build", exist_ok=True)
    with (
        patch_model,
        patch_fp_model,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            "SA8295P ADP",
            "--skip-profiling",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--checkpoint",
            "DEFAULT_W4A16",
        ]

        with pytest.raises(
            ValueError,
            match=r"The selected precision \(w4a16\) is not supported on this target device",
        ):
            export_main()  # Call the main function to submit the compile jobs


def test_cli_device_with_skips_unsupported_context_length(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
) -> None:
    (
        _,
        patch_model,
        patch_fp_model,
        _,
        _,
        _,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, 4096, 128, tmp_path)

    os.makedirs("build", exist_ok=True)
    with (
        patch_model,
        patch_fp_model,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            "SA8295P ADP",
            "--skip-profiling",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--checkpoint",
            "DEFAULT_W4",
        ]

        with pytest.raises(
            ValueError,
            match=rf"The {base_name}'s context length is too large to deploy on SA8295P\. Please set the context length to 1024 or lower\.",
        ):
            export_main()  # Call the main function to submit the compile jobs


def test_cli_device_with_skips(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    device: hub.Device,
    skip_inferencing: bool,
    skip_profiling: bool,
    target_runtime: TargetRuntime,
    precision: Precision,
) -> None:
    context_length = 4096
    sequence_length = 128
    (
        _,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            device.name,
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--target-runtime",
            target_runtime.value,
            "--context-length",
            str(context_length),
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
        ]
        if not skip_inferencing:
            sys.argv.append("--do-inferencing")
        if skip_profiling:
            sys.argv.append("--skip-profiling")

        export_main()  # Call the main function to submit the compile jobs

        # Compile is called parts * 2 times (num_splits token parts, num_splits prompt parts)
        assert mock_hub.submit_compile_job.call_count == parts * 2
        call_args_list = mock_hub.submit_compile_job.call_args_list

        assert all(c.kwargs["device"].name == device.name for c in call_args_list)

        # Link parts times - num_splits
        assert mock_hub.submit_link_job.call_count == parts

        call_args_list = mock_hub.submit_link_job.call_args_list

        expected_model_name = f"{base_name}_{precision}"
        assert all(
            call.kwargs["name"] == f"{expected_model_name}_part_{i + 1}_of_{parts}"
            for i, call in enumerate(call_args_list)
        )

        # Skip profile and inference combinations.
        if skip_inferencing:
            mock_hub.submit_inference_job.assert_not_called()
        else:
            mock_hub.submit_inference_job.assert_called()

        if skip_profiling:
            mock_hub.submit_profile_job.assert_not_called()
        else:
            mock_hub.submit_profile_job.assert_called()
        assert tmp_path.exists()
        assert tmp_path.is_dir()


def test_cli_multiple_context_lengths_link_jobs(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    device: hub.Device,
    target_runtime: TargetRuntime,
    precision: Precision = Precision.w4a16,  # noqa: PT028
) -> None:
    """Link jobs should combine across both sequence and context lengths.

    With num_splits=N and M context lengths, there should still be exactly N
    link jobs (one per split), each combining all (seq_len, ctx_len) variants.
    """
    context_lengths = [1024, 4096]
    sequence_lengths = [1, 128]
    (
        _,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(
        model_cls, base_name, context_lengths[0], sequence_lengths[0], tmp_path
    )

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, max(context_lengths))
        }

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            device.name,
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--target-runtime",
            target_runtime.value,
            "--context-length",
            ",".join(str(cl) for cl in context_lengths),
            "--sequence-length",
            ",".join(str(sl) for sl in sequence_lengths),
            "--skip-profiling",
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
        ]

        export_main()

        num_instantiations = len(context_lengths) * len(sequence_lengths)
        assert mock_hub.submit_compile_job.call_count == parts * num_instantiations

        # Regardless of how many context lengths are used, there must be exactly
        # `parts` link jobs — one per split, combining all (seq, ctx) variants.
        assert mock_hub.submit_link_job.call_count == parts
        expected_model_name = f"{base_name}_{precision}"
        assert all(
            call.kwargs["name"] == f"{expected_model_name}_part_{i + 1}_of_{parts}"
            for i, call in enumerate(mock_hub.submit_link_job.call_args_list)
        )

        # Each link job must receive compile-job outputs for every instantiation.
        for call in mock_hub.submit_link_job.call_args_list:
            linked_models = call.args[0]
            assert len(linked_models) == num_instantiations


def test_cli_chipset_with_options(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    chipset: str,
    context_length: int,
    sequence_length: int,
    target_runtime: TargetRuntime,
    precision: Precision = Precision.w4a16,  # noqa: PT028
) -> None:
    (
        mock_from_pretrained,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker as mock_onnx_checker,
        patch_onnx_load,
        patch_split_onnx as mock_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        mock_onnx_checker.return_value = None
        mock_split_onnx.return_value = None
        compile_options = "compile_extra"
        profile_options = "profile_extra"
        link_options = "link_extra"

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",  # script name
            "--chipset",
            chipset,
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--compile-options",
            compile_options,
            "--profile-options",
            profile_options,
            "--link-options",
            link_options,
            "--sequence-length",
            f"{sequence_length},1",
            "--context-length",
            str(context_length),
            "--target-runtime",
            target_runtime.value,
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
            "--do-inferencing",
        ]

        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        export_main()  # Call the main function to submit the compile jobs

        assert mock_hub.submit_compile_job.call_count == parts * 2
        assert mock_hub.submit_profile_job.call_count == parts * 2
        assert mock_hub.submit_inference_job.call_count == parts * 2

        for mock in [
            mock_hub.submit_compile_job,
            mock_hub.submit_profile_job,
            mock_hub.submit_inference_job,
        ]:
            assert all(
                f"chipset:{chipset}" in call.kwargs["device"].attributes
                for call in mock.call_args_list
            )

        # Link parts times - num_splits
        assert mock_hub.submit_link_job.call_count == parts
        expected_model_name = f"{base_name}_{precision}"
        assert all(
            call.kwargs["name"] == f"{expected_model_name}_part_{i + 1}_of_{parts}"
            for i, call in enumerate(mock_hub.submit_link_job.call_args_list)
        )
        mock_get_hub_link_options = (
            mock_from_pretrained.return_value.get_hub_link_options
        )
        assert mock_get_hub_link_options.call_count == parts
        assert all(
            call.args == (target_runtime, link_options)
            for call in mock_get_hub_link_options.call_args_list
        )

        mock_get_hub_compile_options = (
            mock_from_pretrained.return_value.get_hub_compile_options
        )

        assert mock_get_hub_compile_options.call_count == parts * 2
        for call in mock_get_hub_compile_options.call_args_list:
            assert call.args == (target_runtime, precision, compile_options)
            assert "context_graph_name" in call.kwargs
            assert (
                call.kwargs["context_graph_name"]._mock_new_parent._mock_name
                == mock_from_pretrained.return_value.get_qnn_context_graph_name._mock_name
            )

        # Profile parts * 2 times
        assert mock_hub.submit_profile_job.call_count == parts * 2
        mock_get_hub_profile_options = (
            mock_from_pretrained.return_value.get_hub_profile_options
        )
        assert mock_get_hub_profile_options.call_count == parts * 2
        for call in mock_get_hub_profile_options.call_args_list:
            assert len(call.args) == 3
            assert call.args[:2] == (target_runtime, profile_options)
            assert (
                call.args[2]._mock_new_parent._mock_name
                == mock_from_pretrained.return_value.get_qnn_context_graph_name._mock_name
            )

        assert mock_hub.submit_inference_job.call_count == parts * 2

        assert tmp_path.exists()
        assert tmp_path.is_dir()

        # TODO (#15224): Remove from_pretrained as part of inference?
        assert mock_from_pretrained.call_count == 4
        assert (
            mock_from_pretrained.call_args_list[0].kwargs["context_length"]
            == context_length
        )
        assert (
            mock_from_pretrained.call_args_list[1].kwargs["context_length"]
            == context_length
        )
        assert (
            mock_from_pretrained.call_args_list[0].kwargs["sequence_length"]
            == sequence_length
        )
        # In instantiations list (160) from _shared/llm/export.py
        assert mock_from_pretrained.call_args_list[1].kwargs["sequence_length"] == 1


# for llama3 all components are tested, i.e. no option to select individual components 'part_1_of_5', 'part_2_of_5', 'part_3_of_5', 'part_4_of_5', 'part_5_of_5'
def test_cli_default_device_select_component(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    device: hub.Device,
    cache_mode: CacheMode,
    skip_download: bool,
    skip_summary: bool,
    target_runtime: TargetRuntime,
    decode_sequence_length: int,
    precision: Precision,
) -> None:
    context_length = 4096
    sequence_length = 128
    (
        _,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    patch_torch_inference = patch("qai_hub_models.utils.compare.torch_inference")

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker as mock_onnx_checker,
        patch_onnx_load,
        patch_split_onnx as mock_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model as mock_get_or_create_cached_model,
        patch_torch_inference as mock_torch_inference,
        patch_tool_versions,
    ):
        mock_onnx_checker.return_value = None
        mock_split_onnx.return_value = None
        mock_torch_inference.return_value = [
            np.array([1.0, 2.0])
        ]  # return mock value for torch inference.

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--model-cache-mode",
            str(cache_mode.name.lower()),
            "--device",
            device.name,
            "--target-runtime",
            target_runtime.value,
            "--context-length",
            str(context_length),
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
        ]
        if skip_download:
            sys.argv.append("--skip-downloading")
        if skip_summary:
            sys.argv.append("--skip-summary")

        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        export_main()  # Call the main function to submit the compile jobs

        assert mock_hub.submit_compile_job.call_count == parts * 2
        assert mock_hub.submit_link_job.call_count == parts
        assert mock_hub.submit_profile_job.call_count == parts * 2

        # Check names
        expected_model_name = f"{base_name}_{precision}"
        for mock in [
            mock_hub.submit_compile_job,
            mock_hub.submit_profile_job,
        ]:
            for i, call in enumerate(mock.call_args_list):
                instantiation_name = (
                    f"ar{sequence_length}_cl{context_length}"
                    if i < parts
                    else f"ar{decode_sequence_length}_cl{context_length}"
                )
                assert (
                    call.kwargs["name"]
                    == f"{expected_model_name}_{instantiation_name}_{(i % parts) + 1}_of_{parts}"
                )

        assert mock_get_or_create_cached_model.call_count == parts * 2
        # Check cache mode is correct.
        for call in mock_get_or_create_cached_model.call_args_list:
            assert call.kwargs["model_name"] == expected_model_name
            assert call.kwargs["cache_mode"] == cache_mode

        # check compile jobs have correct device name.
        assert all(
            call.kwargs["device"].name == device.name
            for call in mock_hub.submit_compile_job.call_args_list
        )

        assert tmp_path.exists()
        assert tmp_path.is_dir()


def setup_test_quantization(
    model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    output_path: str,
    precision: Precision,
    checkpoint: str | None = None,
    num_samples: int = 0,
    use_seq_mse: bool = False,
    use_dynamic_shapes: bool = False,
) -> str:
    if not (
        (Path(output_path) / "model.encodings").exists()
        and (Path(output_path) / "model.data").exists()
        and (
            (
                (Path(output_path) / "model_seqlen1_cl4096.onnx").exists()
                and (Path(output_path) / "model_seqlen128_cl4096.onnx").exists()
            )
            or (Path(output_path) / "model_dynamic.onnx").exists()
        )
    ):
        quantize(
            quantized_model_cls=model_cls,
            fp_model_cls=fp_model_cls,
            context_length=4096,
            seq_len=2048,
            precision=precision,
            output_dir=output_path,
            allow_cpu_to_quantize=True,
            checkpoint=checkpoint,
            num_samples=num_samples,
            use_seq_mse=use_seq_mse,
            use_dynamic_shapes=use_dynamic_shapes,
        )
        cleanup()
    return output_path


# ---------------------------------------------------------------------------
# LLM performance collection helpers
# ---------------------------------------------------------------------------


@dataclass
class CachedExportData:
    """Data captured from the first export run, reused across devices.

    DLC compilation (ONNX → QNN) is expensive and device-agnostic.
    The genie bundle metadata (llm_config, input/output specs, encodings,
    checkpoint path) is also invariant across devices.

    Caching all of this lets subsequent devices skip model loading, ONNX
    export, and compilation entirely — only the fast device-specific link
    step and bundle assembly need to rerun.
    """

    model_name: str
    model_display_name: str
    link_options: str
    checkpoint: str | os.PathLike | Path
    llm_config: Any
    input_specs: dict[str, Any]
    output_specs: dict[str, Any]
    input_encodings_path: str
    context_lengths: list[int]
    compile_jobs_to_link: dict[str, list[hub.CompileJob]]


class CompileJobCache:
    """Session-scoped cache: keyed by (model_id, precision).

    Populated on the first device; replayed for every subsequent device
    so compilation only happens once per (model, precision) pair.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, Precision], CachedExportData] = {}

    def get(self, model_id: str, precision: Precision) -> CachedExportData | None:
        return self._cache.get((model_id, precision))

    def set(
        self, model_id: str, precision: Precision, cached: CachedExportData
    ) -> None:
        self._cache[(model_id, precision)] = cached


def _link_download_and_prepare_bundle(
    cached: CachedExportData,
    model_id: str,
    model_cls: type[LLM_AIMETOnnx],
    device: ScorecardDevice,
    precision: Precision,
    output_dir: Path,
) -> None:
    """Link cached compile jobs for a new device, download DLCs, prepare genie bundle.

    This is the fast path for device 2+: no model loading, no ONNX export,
    no compilation. Only the device-specific link step and bundle assembly run.
    """
    hub_device = hub.get_devices(name=device.execution_device.name)[-1]
    chipset_attr = next(
        (attr for attr in hub_device.attributes if "chipset" in attr), None
    )
    chipset = chipset_attr.split(":")[-1] if chipset_attr else None
    target_runtime = TargetRuntime.GENIE

    link_jobs: dict[str, hub.client.LinkJob] = {}
    for component_name, cjobs in cached.compile_jobs_to_link.items():
        models: list[hub.Model | str | Path | None] = [
            cast(hub.Model, cjob.get_target_model()) for cjob in cjobs
        ]
        full_name = f"{cached.model_name}_{component_name}"
        link_job = hub.submit_link_job(
            models,
            name=full_name,
            options=cached.link_options,
            device=hub_device,
        )
        link_jobs[component_name] = link_job

    target_model_list: list[str] = []
    output_path = output_dir / ASSET_CONFIG.get_release_asset_name(
        cached.model_name, target_runtime, precision, chipset
    )
    output_path.mkdir(parents=True, exist_ok=True)
    for component_name, link_job in link_jobs.items():
        if not link_job.wait().success:
            raise RuntimeError(
                f"Link job {link_job.job_id} failed. See: {link_job.url}"
            )
        target_model = link_job.get_target_model()
        assert target_model is not None
        target_model_filename = f"{cached.model_name}_{component_name}.bin"
        target_model_list.append(target_model_filename)
        target_model.download(str(output_path / target_model_filename))

    first_link_job = next(iter(link_jobs.values()))
    tool_versions = ToolVersions.from_job(first_link_job)
    tool_versions.to_yaml(output_path / "tool-versions.yaml")

    model_cls.prepare_genie_assets(
        hub_device=hub_device,
        checkpoint=cached.checkpoint,
        llm_config=cached.llm_config,
        context_lengths=cached.context_lengths,
        model_list=target_model_list,
        output_path=output_path,
        precision=precision,
        encodings_path=cached.input_encodings_path,
        input_specs=cached.input_specs,
        output_specs=cached.output_specs,
        model_id=model_id,
        model_name=cached.model_display_name,
    )


def run_llm_perf_test(
    model_id: str,
    export_model_func: Callable,
    device: ScorecardDevice,
    precision: Precision,
    compile_job_cache: CompileJobCache,
    output_dir: Path | str,
    model_cls: type[LLM_AIMETOnnx],
    model_asset_version: int | str,
    num_splits: int,
    export_context_lengths: list[int] | None = None,
    export_sequence_lengths: list[int] | None = None,
    fp_model_cls: type[LLMBase] | None = None,
    position_processor_cls: type | None = None,
    num_layers_per_split: int | None = None,
    qairt_sdk_path: str | None = None,
    skip_perf_update: bool = False,
) -> tuple[float | None, float | None]:
    """Compile, run QDC, and update perf.yaml for one (model, precision, device).

    All context/sequence lengths are compiled into a single Genie bundle;
    perf.yaml is updated for each context length individually.

    For device 2..N, model loading, ONNX export, and compilation are skipped
    entirely. Only the fast device-specific link step, bundle assembly, and
    QDC run execute.

    Returns (tokens_per_second, time_to_first_token_ms).
    """
    from qai_hub_models.models._shared.llm.perf_collection import update_perf_yaml
    from qai_hub_models.utils.qdc.genie_jobs import submit_genie_bundle_to_qdc_device

    if export_context_lengths is None:
        export_context_lengths = DEFAULT_EXPORT_CONTEXT_LENGTHS
    if export_sequence_lengths is None:
        export_sequence_lengths = DEFAULT_EXPORT_SEQUENCE_LENGTHS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cached = compile_job_cache.get(model_id, precision)
    if cached is not None:
        # Fast path: skip model loading, ONNX export, and compilation.
        _link_download_and_prepare_bundle(
            cached, model_id, model_cls, device, precision, output_dir
        )
        model_name = cached.model_name
    else:
        # First device: full export, capture compile jobs + metadata.
        export_kwargs: dict[str, Any] = dict(
            checkpoint=f"DEFAULT_{str(precision).upper()}",
            sequence_length=export_sequence_lengths,
            context_length=export_context_lengths,
            _skip_quantsim_creation=True,
            model_cls=model_cls,
            model_id=model_id,
            model_asset_version=model_asset_version,
            num_splits=num_splits,
            output_dir=str(output_dir),
        )
        if num_layers_per_split is not None:
            export_kwargs["num_layers_per_split"] = num_layers_per_split
        if fp_model_cls is not None:
            export_kwargs["fp_model"] = fp_model_cls.from_pretrained(
                sequence_length=max(export_sequence_lengths),
                context_length=max(export_context_lengths),
            )
        if position_processor_cls is not None:
            export_kwargs["position_processor_cls"] = position_processor_cls

        common_export_flags = dict(
            device=device.execution_device,
            precision=precision,
            skip_downloading=False,
            skip_profiling=True,
            skip_inferencing=True,
            skip_summary=True,
            target_runtime=TargetRuntime.GENIE,
        )

        captured_compile_jobs_to_link: dict[str, list[hub.CompileJob]] = {}
        original_compile = hub.submit_compile_job
        original_link = hub.submit_link_job

        def _capture_compile(*args: Any, **kwargs: Any) -> hub.CompileJob:
            job = original_compile(*args, **kwargs)
            name = kwargs.get("name", "")
            for part_idx in range(num_splits):
                suffix = f"_{part_idx + 1}_of_{num_splits}"
                component_name = f"part{suffix}"
                if name.endswith(suffix):
                    captured_compile_jobs_to_link.setdefault(component_name, []).append(
                        job
                    )
                    break
            return job

        captured_link_options: list[str] = []

        def _capture_link(*args: Any, **kwargs: Any) -> hub.client.LinkJob:
            if "options" in kwargs:
                captured_link_options.append(kwargs["options"])
            return original_link(*args, **kwargs)

        captured_metadata: dict[str, Any] = {}
        original_prepare = model_cls.prepare_genie_assets

        def _capture_prepare(*args: Any, **kwargs: Any) -> None:
            captured_metadata["checkpoint"] = kwargs.get("checkpoint")
            captured_metadata["llm_config"] = kwargs.get("llm_config")
            captured_metadata["input_specs"] = kwargs.get("input_specs")
            captured_metadata["output_specs"] = kwargs.get("output_specs")
            captured_metadata["encodings_path"] = kwargs.get("encodings_path")
            captured_metadata["model_display_name"] = kwargs.get("model_name")
            return original_prepare(*args, **kwargs)

        with (
            mock.patch("qai_hub.submit_compile_job", side_effect=_capture_compile),
            mock.patch("qai_hub.submit_link_job", side_effect=_capture_link),
            mock.patch.object(
                model_cls, "prepare_genie_assets", side_effect=_capture_prepare
            ),
        ):
            export_model_func(**common_export_flags, **export_kwargs)

        from qai_hub_models.utils.args import get_export_model_name

        model_name = get_export_model_name(
            model_cls, model_id, precision, export_kwargs
        )

        compile_job_cache.set(
            model_id,
            precision,
            CachedExportData(
                model_name=model_name,
                model_display_name=captured_metadata.get("model_display_name", ""),
                link_options=captured_link_options[0] if captured_link_options else "",
                checkpoint=captured_metadata.get("checkpoint", ""),
                llm_config=captured_metadata.get("llm_config"),
                input_specs=captured_metadata.get("input_specs", {}),
                output_specs=captured_metadata.get("output_specs", {}),
                input_encodings_path=captured_metadata.get("encodings_path", ""),
                context_lengths=export_context_lengths,
                compile_jobs_to_link=captured_compile_jobs_to_link,
            ),
        )

    genie_bundle_path = output_dir / ASSET_CONFIG.get_release_asset_name(
        model_name, TargetRuntime.GENIE, precision, device.chipset
    )

    # QDC run
    api_token = os.environ.get("QDC_API_TOKEN")
    if not api_token:
        raise ValueError("QDC_API_TOKEN environment variable is not set")

    tps, ttft = submit_genie_bundle_to_qdc_device(
        api_token,
        device.reference_device.name,
        str(genie_bundle_path),
        job_name=f"Genie {model_id} {precision}",
        qairt_sdk_path=qairt_sdk_path,
    )

    # Update perf.yaml with only the max context length
    if not skip_perf_update and tps is not None and ttft is not None:
        update_perf_yaml(
            model_id,
            device.reference_device_name,
            precision,
            max(export_context_lengths),
            tps,
            ttft,
        )

    return tps, ttft

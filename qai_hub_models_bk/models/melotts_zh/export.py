# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, cast

import qai_hub as hub
import torch

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import (
    ModelFileMetadata,
    ModelMetadata,
    merge_input_metadata,
    merge_output_metadata,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.models.melotts_zh import MODEL_ID, App, Model
from qai_hub_models.utils import quantization as quantization_utils
from qai_hub_models.utils.args import (
    export_parser,
    get_component_input_spec_kwargs,
    get_export_model_name,
    get_model_kwargs,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.base_model import BaseModel, PretrainedCollectionModel
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.export_result import CollectionExportResult, ExportResult
from qai_hub_models.utils.export_without_hub_access import export_without_hub_access
from qai_hub_models.utils.input_spec import (
    InputSpec,
    make_torch_inputs,
    to_hub_input_specs,
)
from qai_hub_models.utils.onnx.helpers import download_and_unzip_workbench_onnx_model
from qai_hub_models.utils.path_helpers import get_next_free_path
from qai_hub_models.utils.printing import (
    print_inference_metrics,
    print_profile_metrics_from_job,
    print_tool_versions,
)
from qai_hub_models.utils.qai_hub_helpers import (
    assert_success_and_get_target_models,
    can_access_qualcomm_ai_hub,
)


def quantize_model(
    precision: Precision | dict[str, Precision],
    model: PretrainedCollectionModel,
    model_name: str,
    onnx_models: dict[str, hub.Model],
    num_calibration_samples: int | None,
    extra_options: str = "",
    input_specs: dict[str, InputSpec] | None = None,
    components: list[str] | None = None,
) -> dict[str, hub.client.QuantizeJob]:
    component_precisions = (
        model.get_component_precisions(precision)
        if isinstance(precision, Precision)
        else precision
    )
    quantize_jobs: dict[str, hub.client.QuantizeJob] = {}
    if not input_specs:
        input_specs = {
            name: model.components[name].get_input_spec() for name in model.components
        }
    for component_name in components or model.component_class_names:
        component_precision = component_precisions[component_name]
        component = model.components[component_name]

        if component_precision != Precision.float:
            print(f"Quantizing {component_name}.")
            if (
                not component_precision.activations_type
                or not component_precision.weights_type
            ):
                raise ValueError(
                    "Quantization is only supported if both weights and activations are quantized."
                )

            calibration_data = quantization_utils.get_calibration_data(
                model,
                input_specs,
                num_calibration_samples,
                component_name=component_name,
                app=App,
            )
            quantize_jobs[component_name] = hub.submit_quantize_job(
                model=onnx_models[component_name],
                calibration_data=calibration_data,
                activations_dtype=component_precision.activations_type,
                weights_dtype=component_precision.weights_type,
                name=f"{model_name}_{component_name}",
                options=component.get_hub_quantize_options(
                    component_precision, extra_options
                ),
            )
    return quantize_jobs


def compile_model(
    model: PretrainedCollectionModel,
    model_name: str,
    device: hub.Device,
    target_runtime: TargetRuntime,
    precision: Precision,
    source_models: dict[str, hub.Model] | None = None,
    input_specs: dict[str, InputSpec] | None = None,
    components: list[str] | None = None,
    extra_options: str = "",
) -> dict[str, hub.client.CompileJob]:
    compile_jobs: dict[str, hub.client.CompileJob] = {}
    for component_name in components or Model.component_class_names:
        component = model.components[component_name]
        input_spec = (input_specs or {}).get(component_name, component.get_input_spec())
        if source_models and (source_model := source_models.get(component_name)):
            model_to_compile = source_model
        else:
            # Trace the model
            model_to_compile = torch.jit.trace(
                component.to("cpu"), make_torch_inputs(input_spec)
            )

        model_compile_options = component.get_hub_compile_options(
            target_runtime,
            precision,
            extra_options,
            device,
            f"{MODEL_ID}_{component_name.lower()}",
        )
        print(f"Optimizing model {component_name} to run on-device")
        submitted_compile_job = hub.submit_compile_job(
            model=model_to_compile,
            input_specs=to_hub_input_specs(input_spec),
            device=device,
            name=f"{model_name}_{component_name}",
            options=model_compile_options,
        )
        compile_jobs[component_name] = cast(
            hub.client.CompileJob, submitted_compile_job
        )
    return compile_jobs


def link_model(
    compiled_models: dict[str, hub.Model],
    device: hub.Device,
    model_name: str,
    model: PretrainedCollectionModel,
    target_runtime: TargetRuntime,
    extra_options: str = "",
) -> dict[str, hub.client.LinkJob]:
    """Link compiled DLCs to context binary for AOT."""
    assert target_runtime.is_aot_compiled, (
        f"link_model() requires an AOT runtime, got {target_runtime}"
    )
    link_jobs: dict[str, hub.client.LinkJob] = {}
    for component_name, compiled_model in compiled_models.items():
        component = model.components[component_name]

        link_options = component.get_hub_link_options(target_runtime, extra_options)
        print(f"Linking {component_name} to context binary")
        link_jobs[component_name] = hub.submit_link_job(
            [compiled_model],
            device=device,
            name=f"{model_name}_{component_name}",
            options=link_options,
        )
    return link_jobs


def profile_model(
    model_name: str,
    device: hub.Device,
    options: dict[str, str],
    target_models: dict[str, hub.Model],
    components: list[str] | None = None,
) -> dict[str, hub.client.ProfileJob]:
    profile_jobs: dict[str, hub.client.ProfileJob] = {}
    for component_name in components or Model.component_class_names:
        print(f"Profiling model {component_name} on a hosted device.")
        submitted_profile_job = hub.submit_profile_job(
            model=target_models[component_name],
            device=device,
            name=f"{model_name}_{component_name}",
            options=options.get(component_name, ""),
        )
        profile_jobs[component_name] = cast(
            hub.client.ProfileJob, submitted_profile_job
        )
    return profile_jobs


def inference_model(
    inputs: dict[str, SampleInputsType],
    model_name: str,
    device: hub.Device,
    options: dict[str, str],
    target_models: dict[str, hub.Model],
    components: list[str] | None = None,
) -> dict[str, hub.client.InferenceJob]:
    inference_jobs: dict[str, hub.client.InferenceJob] = {}
    for component_name in components or Model.component_class_names:
        print(
            f"Running inference for {component_name} on a hosted device with example inputs."
        )
        submitted_inference_job = hub.submit_inference_job(
            model=target_models[component_name],
            inputs=inputs[component_name],
            device=device,
            name=f"{model_name}_{component_name}",
            options=options.get(component_name, ""),
        )
        inference_jobs[component_name] = cast(
            hub.client.InferenceJob, submitted_inference_job
        )
    return inference_jobs


def download_model(
    output_dir: os.PathLike | str,
    model: PretrainedCollectionModel,
    runtime: TargetRuntime,
    precision: Precision,
    tool_versions: ToolVersions,
    target_models: dict[str, hub.Model],
    zip_assets: bool,
) -> Path:
    output_folder_name = os.path.basename(output_dir)
    output_path = get_next_free_path(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = Path(tmpdir) / output_folder_name
        dst_path.mkdir()

        # Download models and capture filenames, then generate metadata
        model_file_metadata = {}
        for component_name, target_model in target_models.items():
            if target_model.model_type == hub.SourceModelType.ONNX:
                onnx_result = download_and_unzip_workbench_onnx_model(
                    target_model, dst_path, component_name
                )
                model_file_name = onnx_result.onnx_graph_name
            else:
                downloaded_path = target_model.download(
                    os.path.join(dst_path, component_name)
                )
                model_file_name = os.path.basename(downloaded_path)

            # Generate metadata using the actual downloaded filename
            model_file_metadata[model_file_name] = ModelFileMetadata.from_hub_model(
                target_model
            )
            # Merge semantic metadata from get_input_spec()
            component = model.components[component_name]
            assert isinstance(component, BaseModel)
            merge_input_metadata(
                model_file_metadata[model_file_name], component.get_input_spec()
            )
            merge_output_metadata(
                model_file_metadata[model_file_name], component.get_output_spec()
            )

        # Extract and save metadata alongside downloaded model
        metadata_path = dst_path / "metadata.json"
        model_metadata = ModelMetadata(
            model_id=MODEL_ID,
            model_name="MeloTTS-ZH",
            runtime=runtime,
            precision=precision,
            tool_versions=tool_versions,
            model_files=model_file_metadata,
        )

        # Dump supplementary files into the model folder
        model.write_supplementary_files(dst_path, model_metadata)

        model_metadata.to_json(metadata_path)
        if zip_assets:
            output_path = Path(
                shutil.make_archive(
                    str(output_path),
                    "zip",
                    root_dir=tmpdir,
                    base_dir=output_folder_name,
                )
            )
        else:
            shutil.move(dst_path, output_path)

    return output_path


def export_model(
    device: hub.Device,
    components: list[str] | None = None,
    precision: Precision = Precision.mixed_with_float,
    num_calibration_samples: int | None = None,
    quantized_model_id: dict[str, str] | None = None,
    skip_compiling: bool = False,
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    target_runtime: TargetRuntime = TargetRuntime.VOICE_AI,
    compile_options: str = "",
    quantize_options: str = "",
    profile_options: str = "",
    fetch_static_assets: str | None = None,
    zip_assets: bool = False,
    **additional_model_kwargs: Any,
) -> CollectionExportResult:
    """
    This function executes the following recipe:

        1. Instantiates a PyTorch model and converts it to a traced TorchScript format
        2. Converts the PyTorch model to ONNX and quantizes the ONNX model.
        3. Compiles the model to an asset that can be run on device
        4. Profiles the model performance on a real device
        5. Inferences the model on sample inputs
        6. Extracts relevant tool (eg. SDK) versions used to compile and profile this model
        7. Downloads the model asset to the local directory
        8. Summarizes the results from profiling and inference

    Each of the last 6 steps can be optionally skipped using the input options.

    Parameters
    ----------
    device
        Device for which to export the model (e.g., hub.Device("Samsung Galaxy S25")).
        Full list of available devices can be found by running `hub.get_devices()`.
    components
        List of sub-components of the model that will be exported.
        Each component is compiled and profiled separately.
        Defaults to all components of the CollectionModel if not specified.
    precision
        The precision to which this model should be quantized.
        Quantization is skipped if the precision is float.
    num_calibration_samples
        The number of calibration data samples
        to use for quantization. If not set, uses the default number
        specified by the dataset. If model doesn't have a calibration dataset
        specified, this must be None.
    quantized_model_id
        A quantized ONNX hub model id, skips quantizing model.
    skip_compiling
        If set, skips compiling of model to format that can run on device.
    skip_profiling
        If set, skips profiling of compiled model on real devices.
    skip_inferencing
        If set, skips computing on-device outputs from sample data.
    skip_downloading
        If set, skips downloading of compiled model.
    skip_summary
        If set, skips waiting for and summarizing results
        from profiling and inference.
    output_dir
        Directory to store generated assets (e.g. compiled model).
        Defaults to `<cwd>/export_assets`.
    target_runtime
        Which on-device runtime to target. Default is TFLite.
    compile_options
        Additional options to pass when submitting the compile job.
    quantize_options
        Additional options to pass when submitting the quantize job.
    profile_options
        Additional options to pass when submitting the profile job.
    fetch_static_assets
        If set, known assets are fetched from the given version rather than re-computing them. Can be passed as "latest" or "v<version>".
    zip_assets
        If set, zip the assets after downloading.
    **additional_model_kwargs
        Additional optional kwargs used to customize
        `model_cls.from_pretrained` and per-component `get_input_spec`

    Returns
    -------
    CollectionExportResult
        A Mapping from component_name to:
            * A CompileJob object containing metadata about the compile job submitted to hub (None if compiling skipped).
            * An InferenceJob containing metadata about the inference job (None if inferencing skipped).
            * A ProfileJob containing metadata about the profile job (None if profiling skipped).
            * A QuantizeJob object containing metadata about the quantize job submitted to hub
        * The path to the downloaded model folder (or zip), or None if one or more of: skip_downloading is True, fetch_static_assets is set, or AI Hub Workbench is not accessible
    """
    model_name = get_export_model_name(
        Model, MODEL_ID, precision, additional_model_kwargs
    )

    output_path = Path(output_dir or Path.cwd() / "export_assets")
    component_arg = components
    components = components or Model.component_class_names
    for component_name in components:
        if component_name not in Model.component_class_names:
            raise ValueError(f"Invalid component {component_name}.")
    if fetch_static_assets or not can_access_qualcomm_ai_hub():
        static_model_path = export_without_hub_access(
            MODEL_ID,
            device,
            skip_profiling,
            skip_inferencing,
            skip_downloading,
            skip_summary,
            output_path,
            target_runtime,
            precision,
            quantize_options + compile_options + profile_options,
            component_arg,
            qaihm_version_tag=fetch_static_assets,
        )
        return CollectionExportResult(
            components={
                component_name: ExportResult() for component_name in components
            },
            download_path=static_model_path,
        )

    hub_device = hub.get_devices(
        name=device.name, attributes=device.attributes, os=device.os
    )[-1]
    chipset_attr = next(
        (attr for attr in hub_device.attributes if "chipset" in attr), None
    )
    chipset = chipset_attr.split(":")[-1] if chipset_attr else None

    # 1. Instantiates a PyTorch model and converts it to a traced TorchScript format
    model = Model.from_pretrained(
        **get_model_kwargs(Model, dict(**additional_model_kwargs, precision=precision))
    )
    input_specs: dict[str, InputSpec] = {
        name: model.components[name].get_input_spec(
            **get_component_input_spec_kwargs(Model, name, additional_model_kwargs)
        )
        for name in components
    }

    # 2. Converts the PyTorch model to ONNX and quantizes the ONNX model.
    quantize_jobs: dict[str, hub.client.QuantizeJob] = {}
    quantized_models: dict[str, hub.Model] | None = None
    if precision != Precision.float:
        if quantized_model_id:
            quantized_models = {
                component: hub_model
                for component in components
                if (hub_model := hub.get_model(quantized_model_id[component]))
                is not None
            }
        else:
            component_precisions = model.get_component_precisions(precision)
            onnx_compile_jobs = compile_model(
                model,
                model_name,
                device,
                TargetRuntime.ONNX,
                precision,
                input_specs=input_specs,
                components=[
                    c
                    for c, p in component_precisions.items()
                    if c in components and p != Precision.float
                ],
            )
            onnx_models = assert_success_and_get_target_models(onnx_compile_jobs)
            quantize_jobs = quantize_model(
                component_precisions,
                model,
                model_name,
                onnx_models,
                num_calibration_samples,
                quantize_options,
                input_specs,
                components,
            )
            if skip_compiling:
                return CollectionExportResult(
                    components={
                        component_name: ExportResult(
                            quantize_job=quantize_jobs.get(component_name)
                        )
                        for component_name in components
                    },
                )
            quantized_models = assert_success_and_get_target_models(quantize_jobs)

    # 3. Compiles the model to an asset that can be run on device
    compile_jobs: dict[str, hub.client.CompileJob] = compile_model(
        model,
        model_name,
        device,
        target_runtime,
        precision,
        quantized_models,
        input_specs=input_specs,
        components=components,
        extra_options=compile_options,
    )

    link_jobs: dict[str, hub.client.LinkJob] | None = None
    target_models: dict[str, hub.Model]
    if target_runtime.uses_hub_link:
        compiled_models = assert_success_and_get_target_models(compile_jobs)
        link_jobs = link_model(
            compiled_models,
            device,
            model_name,
            model,
            target_runtime,
        )
    target_models = assert_success_and_get_target_models(
        link_jobs if link_jobs else compile_jobs
    )

    # 4. Profiles the model performance on a real device
    profile_jobs: dict[str, hub.client.ProfileJob] = {}
    if not skip_profiling:
        profile_jobs = profile_model(
            model_name,
            device,
            model.get_hub_profile_options(target_runtime, profile_options),
            target_models,
            components,
        )

    # 5. Inferences the model on sample inputs
    inference_jobs: dict[str, hub.client.InferenceJob] = {}
    if not skip_inferencing:
        inference_jobs = inference_model(
            model.sample_inputs(
                input_specs=input_specs,
                use_channel_last_format=target_runtime.channel_last_native_execution,
            ),
            model_name,
            device,
            model.get_hub_profile_options(target_runtime, profile_options),
            target_models,
            components,
        )

    # 6. Extracts relevant tool (eg. SDK) versions used to compile and profile this model
    tool_versions: ToolVersions | None = None
    tool_versions_are_from_device_job = False
    if not skip_summary or not skip_downloading:
        profile_job = next(iter(profile_jobs.values())) if profile_jobs else None
        inference_job = next(iter(inference_jobs.values())) if inference_jobs else None
        compile_job = next(iter(compile_jobs.values())) if compile_jobs else None
        if profile_job is not None and profile_job.wait():
            tool_versions = ToolVersions.from_job(profile_job)
            tool_versions_are_from_device_job = True
        elif inference_job is not None and inference_job.wait():
            tool_versions = ToolVersions.from_job(inference_job)
            tool_versions_are_from_device_job = True
        elif compile_job and compile_job.wait():
            tool_versions = ToolVersions.from_job(compile_job)

    # 7. Downloads the model asset to the local directory
    downloaded_model_path: Path | None = None
    if not skip_downloading and tool_versions is not None:
        model_directory = output_path / ASSET_CONFIG.get_release_asset_name(
            MODEL_ID, target_runtime, precision, chipset
        )
        downloaded_model_path = download_model(
            model_directory,
            model,
            target_runtime,
            precision,
            tool_versions,
            target_models,
            zip_assets,
        )

    # 8. Summarizes the results from profiling and inference
    if not skip_summary and not skip_profiling:
        for component_name in components:
            profile_job = profile_jobs[component_name]
            assert profile_job.wait().success, "Job failed: " + profile_job.url
            profile_data: dict[str, Any] = profile_job.download_profile()
            print_profile_metrics_from_job(profile_job, profile_data)

    if not skip_summary and not skip_inferencing:
        for component_name in components:
            component = model.components[component_name]

            inference_job = inference_jobs[component_name]
            sample_inputs = component.sample_inputs(
                input_specs[component_name], use_channel_last_format=False
            )
            torch_out = torch_inference(
                component,
                sample_inputs,
                return_channel_last_output=target_runtime.channel_last_native_execution,
            )
            assert inference_job.wait().success, "Job failed: " + inference_job.url
            inference_result = inference_job.download_output_data()
            assert inference_result is not None
            print_inference_metrics(
                inference_job, inference_result, torch_out, component.get_output_names()
            )

    if not skip_summary:
        print_tool_versions(tool_versions, tool_versions_are_from_device_job)

    if downloaded_model_path:
        print(f"{model_name} was saved to {downloaded_model_path}\n")

    return CollectionExportResult(
        components={
            component_name: ExportResult(
                compile_job=compile_jobs[component_name],
                link_job=link_jobs.get(component_name) if link_jobs else None,
                inference_job=inference_jobs.get(component_name, None),
                profile_job=profile_jobs.get(component_name, None),
                quantize_job=quantize_jobs.get(component_name, None),
            )
            for component_name in components
        },
        download_path=downloaded_model_path,
        tool_versions=tool_versions,
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.mixed_with_float: [
            TargetRuntime.VOICE_AI,
        ],
    }

    parser = export_parser(
        model_cls=Model,
        export_fn=export_model,
        supported_precision_runtimes=supported_precision_runtimes,
        default_export_device="Samsung Galaxy S25 (Family)",
    )
    args = parser.parse_args()
    export_model(**vars(args))


if __name__ == "__main__":
    main()

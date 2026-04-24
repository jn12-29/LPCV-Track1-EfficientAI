# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import copy
import importlib
import os
import subprocess
import sys
from typing import Any

import qai_hub as hub
from ruamel.yaml import YAML

from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.export_result import CollectionExportResult, ExportResult
from qai_hub_models.utils.measurement import (
    get_checkpoint_file_size,
    get_tflite_unique_parameters,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_MODELS_ROOT


def get_model_size_and_parameters(
    job_or_model: hub.Job | hub.Model | None,
) -> tuple[str, str | None]:
    assert job_or_model is not None
    if isinstance(job_or_model, (hub.ProfileJob, hub.InferenceJob)):
        model = job_or_model.model
    elif isinstance(job_or_model, hub.CompileJob):
        if (target_model := job_or_model.get_target_model()) is None:
            raise ValueError("Failed compile job.")
        model = target_model
    elif isinstance(job_or_model, hub.Model):
        model = job_or_model
    else:
        raise TypeError(f"Invalid type for `job_or_model`: {type(job_or_model)}")
    with qaihm_temp_dir() as tmp_dirname:
        path = model.download(os.path.join(tmp_dirname, "model"))
        size = get_checkpoint_file_size(path)
        if model.model_type == hub.SourceModelType.TFLITE:
            parameters = str(get_tflite_unique_parameters(path))
        else:
            parameters = None
        return str(size), parameters


def add_details_to_info_yaml(
    info: dict[str, Any], details: dict[str, dict[str, tuple[str, str | None]]]
) -> dict[str, Any]:
    # Clean the old keys
    new_info = copy.deepcopy(info)
    if "technical_details" in info:
        for info_type in info["technical_details"]:
            if "model size" in info_type.lower() or "parameters" in info_type.lower():
                del new_info["technical_details"][info_type]

    # Add more standard keys here
    for precision_str, precision_data in details.items():
        for name, (size, parameters) in precision_data.items():
            if "technical_details" in new_info:
                param_key = (
                    "Number of parameters"
                    if name == "model"
                    else f"Number of parameters ({name})"
                )
                size_key = "Model size" if name == "model" else f"Model size ({name})"
                if parameters is not None:
                    new_info["technical_details"][param_key] = parameters
                new_info["technical_details"][f"{size_key} ({precision_str})"] = size

    return new_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        type=str,
        default=None,
        help="Models for which to autofill info.yaml.",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="If set, generates files for all models.",
    )
    parser.add_argument(
        "--skip-compile-jobs",
        "-s",
        action="store_true",
        help="Skip running compile jobs.",
        default=False,
    )
    args = parser.parse_args()
    skip_compile = args.skip_compile_jobs

    assert args.all or args.models, "Must specify -a or -m."
    models: list[str]
    models = parser.parse_args().models if args.models else MODEL_IDS

    for model_name in models:
        model_dir = QAIHM_MODELS_ROOT / model_name
        yaml = YAML()
        yaml.width = sys.maxsize
        export_options = {}
        if (model_dir / "code-gen.yaml").exists():
            with open(model_dir / "code-gen.yaml") as f:
                export_options = yaml.load(f)
        perf: dict[str, Any] = {}
        if (model_dir / "perf.yaml").exists():
            with open(model_dir / "perf.yaml") as f:
                perf = yaml.load(f)

        model_info = QAIHMModelInfo.from_model(model_name)
        precisions = model_info.code_gen_config.supported_precisions
        details: dict[str, dict[str, tuple[str, str | None]]] = {
            str(precision): {} for precision in precisions
        }

        if export_options.get("is_precompiled", False):
            print(f"Skipping {model_name} since its a is_precompiled asset.")
            continue
        try:
            if not skip_compile:
                for precision in precisions:
                    # Install dependencies
                    requirements_file = None

                    if (model_dir / "requirements.txt").exists():
                        requirements_file = os.path.join(model_dir, "requirements.txt")
                    if requirements_file:
                        subprocess.run(
                            ["pip", "install", "-r", requirements_file], check=False
                        )

                    # imports the module from the given path
                    model_module = importlib.import_module(
                        f"qai_hub_models.models.{model_name}.export"
                    )
                    results = model_module.export_model(
                        device=hub.Device("Samsung Galaxy S25 (Family)"),
                        skip_downloading=True,
                        skip_profiling=True,
                        skip_inferencing=True,
                        skip_summary=True,
                        target_runtime=TargetRuntime.TFLITE,
                        precision=precision,
                    )

                    if isinstance(results, ExportResult):
                        details[str(precision)]["model"] = (
                            get_model_size_and_parameters(results.compile_job)
                        )
                    elif isinstance(results, CollectionExportResult):
                        for component_name in results.components:
                            details[str(precision)][component_name] = (
                                get_model_size_and_parameters(
                                    results.components[component_name].compile_job
                                )
                            )
                    else:
                        raise NotImplementedError(  # noqa: TRY301
                            f"Unknown export script result type: {type(results)}"
                        )
            else:
                if not perf:
                    print(f"No perf data for model {model_name}. Skipping.")
                    continue
                for precision_str, precision_data in perf["precisions"].items():
                    for submodel_name, submodel_data in precision_data[
                        "components"
                    ].items():
                        details_key = (
                            submodel_name
                            if len(precision_data["components"]) > 1
                            else "model"
                        )
                        for runtime_type in ["tflite", "qnn_dlc", "onnx"]:
                            if runtime_type in submodel_data["universal_assets"]:
                                details[precision_str][details_key] = (
                                    get_model_size_and_parameters(
                                        hub.get_model(
                                            submodel_data["universal_assets"][
                                                runtime_type
                                            ]
                                        )
                                    )
                                )
                                break

            if (model_dir / "info.yaml").exists():
                with open(model_dir / "info.yaml") as f:
                    info = yaml.load(f)
                    new_info = add_details_to_info_yaml(info=info, details=details)
                with open(model_dir / "info.yaml", "wb") as f:
                    yaml.dump(new_info, f)
            else:
                new_info = add_details_to_info_yaml(info={}, details=details)
                with open(model_dir / "info.yaml", "w") as f:
                    yaml.dump(new_info, f)

        except Exception as e:
            raise ValueError(f"Failed for model {model_name}") from e


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import datetime
import multiprocessing
import os
import shutil
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import ruamel.yaml
from qai_hub import JobType

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.envvars import (
    ArtifactsDirEnvvar,
    BranchEnvvar,
    DateFormatEnvvar,
    DeploymentEnvvar,
    EnabledModelsEnvvar,
    EnabledPrecisionsEnvvar,
    IgnoreExistingIntermediateJobsDuringCollectionEnvvar,
    SpecialModelSetting,
    SpecialPrecisionSetting,
    StaticModelsDirEnvvar,
)
from qai_hub_models.scorecard.execution_helpers import (
    get_compile_parameterized_pytest_config,
    get_enabled_paths_for_testing,
    get_evaluation_parameterized_pytest_config,
    get_profile_parameterized_pytest_config,
    get_quantize_parameterized_pytest_config,
)
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.scorecard.results import PerformanceDiff
from qai_hub_models.scorecard.results.code_gen import (
    update_code_gen_failure_reasons,
    update_model_publish_status,
)
from qai_hub_models.scorecard.results.spreadsheet import ResultsSpreadsheet
from qai_hub_models.scorecard.results.yaml import (
    COMPILE_YAML_BASE,
    ENVIRONMENT_ENV_BASE,
    INFERENCE_YAML_BASE,
    LINK_YAML_BASE,
    PROFILE_YAML_BASE,
    QUANTIZE_YAML_BASE,
    TOOL_VERSIONS_BASE,
    CompileScorecardJobYaml,
    InferenceScorecardJobYaml,
    LinkScorecardJobYaml,
    ProfileScorecardJobYaml,
    QuantizeScorecardJobYaml,
    ScorecardJobYaml,
)
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.scorecard.static.model_config import ScorecardModelConfig
from qai_hub_models.scorecard.static.model_exec import (
    get_static_model_test_parameterizations,
)
from qai_hub_models.utils.collection_model_helpers import get_components
from qai_hub_models.utils.hub_clients import (
    default_hub_client_as,
    deployment_is_prod,
    get_default_hub_deployment,
    get_scorecard_client_or_raise,
    set_default_hub_client,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS
from qai_hub_models.utils.testing_async_utils import (
    get_compile_job_ids_file,
    get_environment_file,
    get_inference_job_ids_file,
    get_link_job_ids_file,
    get_profile_job_ids_file,
    get_quantize_job_ids_file,
    get_tool_versions_file,
)

# If the precision is any one of these two values, add it to the branch column
# to allow tableau to differentiate different types of scorecards
SPECIAL_PRECISIONS = ["bench", "default_quantized"]


def read_jobs_config(config_path: str) -> dict:
    """Read yaml files."""
    yaml = ruamel.yaml.YAML()
    with open(config_path) as file:
        return yaml.load(file)


def write_jobs_config(config: dict, path: str) -> None:
    """Write yaml files with special characters like copyright logo, etc."""
    yaml = ruamel.yaml.YAML()
    with open(path, "w") as file:
        yaml.dump(config, file)


def clear_jobs(
    yamls: ScorecardJobYaml,
    model_list: list[str],
    all_models: bool,
) -> None:
    if all_models:
        yamls.clear_jobs()
    else:
        for model in model_list:
            yamls.clear_jobs(model)


def remove_failed_jobs(config: dict) -> None:
    """
    Failed jobs need to be in the config to get their job ids for summary but
    we want to delete them before writing to perf.yaml
    """
    for model_config in config["models"]:
        for perf_metrics in model_config["performance_metrics"]:
            for key in list(perf_metrics.keys()):
                if (
                    "job_id" in perf_metrics[key]
                    and perf_metrics[key]["inference_time"] == "null"
                ):
                    del perf_metrics[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    EnabledModelsEnvvar.add_arg(parser)
    parser.add_argument(
        "--quantize-ids",
        type=str,
        help="Comma-separated list of paths to additional quantize job yamls. Jobs in these YAMLS will be added to or override jobs in the existing quantize job intermediates YAML. YAML specified later in the list will override identical jobs in previous YAML.",
    )
    parser.add_argument(
        "--profile-ids",
        type=str,
        help="Comma-separated list of paths to additional profile job yamls. Jobs in these YAMLS will be added to or override jobs in the existing profile job intermediates YAML. YAML specified later in the list will override identical jobs in previous YAML.",
    )
    parser.add_argument(
        "--compile-ids",
        type=str,
        help="Comma-separated list of paths to additional compile job yamls. Jobs in these YAMLS will be added to or override jobs in the existing compile job intermediates YAML. YAML specified later in the list will override identical jobs in previous YAML.",
    )
    parser.add_argument(
        "--link-ids",
        type=str,
        help="Comma-separated list of paths to additional link job yamls. Jobs in these YAMLS will be added to or override jobs in the existing link job intermediates YAML. YAML specified later in the list will override identical jobs in previous YAML.",
    )
    parser.add_argument(
        "--inference-ids",
        type=str,
        help="Comma-separated list of paths to additional inference job yamls. Jobs in these YAMLS will be added to or override jobs in the existing inference job intermediates YAML. YAML specified later in the list will override identical jobs in previous YAML.",
    )
    IgnoreExistingIntermediateJobsDuringCollectionEnvvar.add_arg(parser)
    StaticModelsDirEnvvar.add_arg(parser)
    parser.add_argument(
        "--gen-csv",
        action="store_true",
        help="Generate a csv that summarizes the profile and compile steps.",
    )
    parser.add_argument(
        "--gen-perf-summary",
        action="store_true",
        help="Generate a summary of the performance per model, and update perf yaml files.",
    )
    parser.add_argument(
        "--sync-code-gen",
        action="store_true",
        help="Sync code generation YAML with failures & successes in the scorecard YAML. If compile job fails for any device, that path is skipped. If the profile job for the default export device fails, that path is skipped.",
    )
    DeploymentEnvvar.add_arg(parser, default=get_default_hub_deployment())
    EnabledPrecisionsEnvvar.add_arg(parser)
    BranchEnvvar.add_arg(parser)
    DateFormatEnvvar.add_arg_group(parser)
    ArtifactsDirEnvvar.add_arg(parser)
    return parser.parse_args()


def process_model(
    model_id: str,
    deployment: str,
    static_models_dir: Path,
    quantize_job_yamls: QuantizeScorecardJobYaml,
    compile_job_yamls: CompileScorecardJobYaml,
    link_job_yamls: LinkScorecardJobYaml,
    profile_job_yamls: ProfileScorecardJobYaml,
    inference_job_yamls: InferenceScorecardJobYaml,
    gen_csv: bool,
    sync_code_gen: bool,
    gen_perf_summary: bool,
    write_model_card: bool,
) -> tuple[ResultsSpreadsheet, QAIHMModelPerf | None, QAIHMModelPerf | None]:
    """
    Process results for a model with an end-to-end pyTorch recipe.

    Parameters
    ----------
    model_id
        Model identifier.
    deployment
        Deployment environment.
    static_models_dir
        Directory containing static model configurations.
    quantize_job_yamls
        YAML containing quantize job information.
    compile_job_yamls
        YAML containing compile job information.
    link_job_yamls
        YAML containing link job information.
    profile_job_yamls
        YAML containing profile job information.
    inference_job_yamls
        YAML containing inference job information.
    gen_csv
        Whether to generate CSV spreadsheet.
    sync_code_gen
        Whether to sync code generation.
    gen_perf_summary
        Whether to generate performance summary.
    write_model_card
        Whether to write model card.

    Returns
    -------
    spreadsheet : ResultsSpreadsheet
        Model spreadsheet, or empty spreadsheet if not gen_csv.
    previous_perf : QAIHMModelPerf | None
        Previous (on disk) perf yaml, or None if not gen_perf_summary.
        Always None for static models.
    current_perf : QAIHMModelPerf | None
        Current (from this scorecard) perf yaml, or None if not gen_perf_summary.
        Always None for static models.
    """
    try:
        if model_id in MODEL_IDS:
            # This model has an end to end pyTorch recipe.
            return process_e2e_recipe_model(
                model_id,
                quantize_job_yamls,
                compile_job_yamls,
                link_job_yamls,
                profile_job_yamls,
                inference_job_yamls,
                gen_csv,
                sync_code_gen,
                gen_perf_summary,
                write_model_card,
            )
        # This model was uploaded statically (as a single file).
        if gen_csv:
            spreadsheet = process_static_file_model(
                model_id,
                deployment,
                static_models_dir,
                quantize_job_yamls,
                compile_job_yamls,
                link_job_yamls,
                profile_job_yamls,
                inference_job_yamls,
            )
        else:
            spreadsheet = ResultsSpreadsheet()
        return (spreadsheet, None, None)
    except Exception as e:
        raise ValueError(f"{model_id} result processing failed.") from e


def _get_pytorch_tags(model_info: QAIHMModelInfo) -> list[str]:
    tags = [tag.value for tag in model_info.tags]
    tags.append("pytorch")
    tags.append(model_info.status.value)
    return tags


def _get_static_tags(model_info: ScorecardModelConfig) -> list[str]:
    tags = [tag.name for tag in model_info.tags]
    tags.append("static")
    tags.append("private")
    tags.append(f"bu-{model_info.bu_owner.value}")
    return tags


@dataclass
class ModelTestParameterizations:
    component_names: list[str] | None
    precisions: set[Precision]
    compile_tests: list[tuple[Precision, ScorecardCompilePath, ScorecardDevice]]
    profile_tests: list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]
    inference_tests: list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]
    enabled_paths: dict[Precision, list[ScorecardProfilePath]]

    @staticmethod
    def from_recipe_model(model_info: QAIHMModelInfo) -> ModelTestParameterizations:
        model_id = model_info.id
        cj = model_info.code_gen_config

        # Get enabled test paths for this model
        model_supported_paths = cj.get_supported_paths_for_testing(
            only_include_passing=False
        )
        model_passing_paths = cj.get_supported_paths_for_testing(
            only_include_passing=True
        )
        enabled_paths = get_enabled_paths_for_testing(
            model_id,
            model_supported_paths,
            model_passing_paths,
            ScorecardProfilePath,
            cj.can_use_quantize_job,
        )

        # Get all enabled tests paramaterizations -- (precision + path + device pairings) -- for this model
        component_names = get_components(model_id)
        precisions = get_quantize_parameterized_pytest_config(
            model_id, model_supported_paths, model_passing_paths
        )
        compile_tests = get_compile_parameterized_pytest_config(
            model_id,
            model_supported_paths,
            model_passing_paths,
            cj.can_use_quantize_job,
            include_mirror_devices=True,
        )
        profile_tests = get_profile_parameterized_pytest_config(
            model_id,
            model_supported_paths,
            model_passing_paths,
            cj.can_use_quantize_job,
            include_mirror_devices=True,
        )
        inference_tests = get_evaluation_parameterized_pytest_config(
            model_id,
            ScorecardDevice.get(cj.default_device),
            model_supported_paths,
            model_passing_paths,
            cj.can_use_quantize_job,
        )

        return ModelTestParameterizations(
            component_names=component_names,
            precisions=set(precisions),
            compile_tests=compile_tests,
            profile_tests=profile_tests,
            inference_tests=inference_tests,
            enabled_paths=enabled_paths,
        )

    @staticmethod
    def from_static_model(
        model_info: ScorecardModelConfig,
    ) -> ModelTestParameterizations:
        return ModelTestParameterizations(
            component_names=None,
            precisions={model_info.precision},
            compile_tests=get_static_model_test_parameterizations(
                model_info.id,
                JobType.COMPILE,
                ScorecardCompilePath,
                model_info.precision,
                model_info.devices,
                list({x.compile_path for x in model_info.enabled_profile_runtimes}),
            ),
            profile_tests=get_static_model_test_parameterizations(
                model_info.id,
                JobType.PROFILE,
                ScorecardProfilePath,
                model_info.precision,
                model_info.devices,
                model_info.enabled_profile_runtimes,
            ),
            inference_tests=get_static_model_test_parameterizations(
                model_info.id,
                JobType.INFERENCE,
                ScorecardProfilePath,
                model_info.precision,
                [model_info.eval_device],
                model_info.enabled_profile_runtimes,
            ),
            enabled_paths={},
        )


def process_e2e_recipe_model(
    model_id: str,
    quantize_job_yamls: QuantizeScorecardJobYaml,
    compile_job_yamls: CompileScorecardJobYaml,
    link_job_yamls: LinkScorecardJobYaml,
    profile_job_yamls: ProfileScorecardJobYaml,
    inference_job_yamls: InferenceScorecardJobYaml,
    gen_csv: bool,
    sync_code_gen: bool,
    gen_perf_summary: bool,
    write_model_card: bool,
) -> tuple[ResultsSpreadsheet, QAIHMModelPerf | None, QAIHMModelPerf | None]:
    """
    Process results for a model with an end-to-end pyTorch recipe.

    Parameters
    ----------
    model_id
        Model identifier.
    quantize_job_yamls
        YAML containing quantize job information.
    compile_job_yamls
        YAML containing compile job information.
    link_job_yamls
        YAML containing link job information.
    profile_job_yamls
        YAML containing profile job information.
    inference_job_yamls
        YAML containing inference job information.
    gen_csv
        Whether to generate CSV spreadsheet.
    sync_code_gen
        Whether to sync code generation.
    gen_perf_summary
        Whether to generate performance summary.
    write_model_card
        Whether to write model card.

    Returns
    -------
    spreadsheet : ResultsSpreadsheet
        Model spreadsheet, or empty spreadsheet if not gen_csv.
    previous_perf : QAIHMModelPerf | None
        Previous (on disk) perf yaml, or None if not gen_perf_summary.
    current_perf : QAIHMModelPerf | None
        Current (from this scorecard) perf yaml, or None if not gen_perf_summary.
    """

    def print_with_id(pstr: str) -> None:
        print(f"{model_id} | {pstr}")

    # Load configs
    model_info = QAIHMModelInfo.from_model(model_id)
    cj = model_info.code_gen_config

    # Skip certain models
    if cj.is_precompiled or cj.skip_hub_tests_and_scorecard or cj.skip_scorecard:
        return ResultsSpreadsheet(), None, None

    # Get enabled test paths for this model
    test_params = ModelTestParameterizations.from_recipe_model(model_info)

    # Get summaries for this model and its components.
    print_with_id("Loading quantize summary")
    quantize_summary = (
        quantize_job_yamls.summary_from_model(
            model_id,
            test_params.precisions,
            test_params.component_names,
        )
        if any(p.has_quantized_activations for p in test_params.precisions)
        and cj.can_use_quantize_job
        else None
    )
    print_with_id("Loading compile summary")
    compile_summary = compile_job_yamls.summary_from_model(
        model_id, test_params.compile_tests, test_params.component_names
    )
    print_with_id("Loading link summary")
    link_summary = link_job_yamls.summary_from_model(
        model_id, test_params.compile_tests, test_params.component_names
    )
    print_with_id("Loading profile summary")
    profile_summary = profile_job_yamls.summary_from_model(
        model_id, test_params.profile_tests, test_params.component_names
    )
    print_with_id("Loading inference summary")
    inference_summary = inference_job_yamls.summary_from_model(
        model_id, test_params.inference_tests, test_params.component_names
    )

    entries: ResultsSpreadsheet = ResultsSpreadsheet()
    code_gen_config = QAIHMModelCodeGen.from_model(model_id)
    entries.set_model_metadata(
        model_id,
        model_info.domain,
        model_info.use_case,
        _get_pytorch_tags(model_info),
        known_failure_reasons=model_info.code_gen_config.disabled_paths,
        default_quantized_precision=code_gen_config.default_quantized_precision,
        default_device=ScorecardDevice.get(code_gen_config.default_device),
    )
    if gen_csv:
        print_with_id("Adding to Spreadsheet")
        entries.append_model_summary_entries(
            model_id,
            test_params.profile_tests,
            test_params.component_names,
            quantize_summary,
            compile_summary,
            link_summary,
            profile_summary,
            inference_summary,
        )

    if sync_code_gen and not cj.freeze_perf_yaml:
        # Enable or disable runtimes on this model depending on whether the default device has passing jobs
        update_code_gen_failure_reasons(
            test_params.enabled_paths,
            test_params.component_names,
            compile_summary,
            link_summary,
            profile_summary,
            cj,
        )
        code_gen_path = cj.to_model_yaml(model_id)
        print_with_id(f"Updated Runtime Failure Reasons in {code_gen_path}")

        # Update model status & reason, if applicable
        if update_model_publish_status(model_info):
            info_yaml_path, _ = model_info.to_model_yaml(write_code_gen=False)
            print_with_id(pstr=f"Updated publish status at {info_yaml_path}")

    model_card = QAIHMModelPerf()
    prev_model_card = QAIHMModelPerf()
    if gen_perf_summary:
        print_with_id("Writing Performance YAML")

        # Build model card
        model_card = profile_summary.get_perf_card(
            include_failed_jobs=True,
            include_unpublished_runtimes=False,
            exclude_form_factors=model_info.private_perf_form_factors or [],
            model_name=model_info.name,
        )

        # Dump model card without failed jobs
        if write_model_card:
            model_card_without_failures = profile_summary.get_perf_card(
                include_failed_jobs=False,
                include_unpublished_runtimes=False,
                exclude_form_factors=model_info.private_perf_form_factors or [],
                model_name=model_info.name,
            )
        else:
            model_card_without_failures = None

        # Load old model card and write new model card
        prev_model_card = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
        if not cj.freeze_perf_yaml and model_card_without_failures:
            card_path = model_card_without_failures.to_model_yaml(model_id)
            print_with_id(f"Wrote {card_path}")

    return entries, prev_model_card, model_card


def process_static_file_model(
    model_id: str,
    deployment: str,
    models_dir: Path,
    quantize_job_yamls: QuantizeScorecardJobYaml | None,
    compile_job_yamls: CompileScorecardJobYaml | None,
    link_job_yamls: LinkScorecardJobYaml | None,
    profile_job_yamls: ProfileScorecardJobYaml | None,
    inference_job_yamls: InferenceScorecardJobYaml | None,
) -> ResultsSpreadsheet:
    """
    Process results for a static model (uploaded onnx or traced pyTorch file).

    Returns model spreadsheet.
    """

    def print_with_id(pstr: str) -> None:
        print(f"{model_id} | {pstr}")

    # Load config
    model_info = ScorecardModelConfig.from_yaml(models_dir / (model_id + ".yaml"))
    test_params = ModelTestParameterizations.from_static_model(model_info)

    # Get summaries for this model and its components.
    with default_hub_client_as(
        get_scorecard_client_or_raise(deployment, model_info.restrict_access)
    ):
        quantize_summary = None
        if quantize_job_yamls:
            print_with_id("Loading quantize summary")
            quantize_summary = quantize_job_yamls.summary_from_model(
                model_id,
                test_params.precisions,
            )

        compile_summary = None
        if compile_job_yamls:
            print_with_id("Loading compile summary")
            compile_summary = compile_job_yamls.summary_from_model(
                model_id,
                test_params.compile_tests,
                test_params.component_names,
            )

        link_summary = None
        if link_job_yamls:
            print_with_id("Loading link summary")
            link_summary = link_job_yamls.summary_from_model(
                model_id,
                test_params.compile_tests,
                test_params.component_names,
            )

        profile_summary = None
        if profile_job_yamls:
            print_with_id("Loading profile summary")
            profile_summary = profile_job_yamls.summary_from_model(
                model_id, test_params.profile_tests, test_params.component_names
            )

        inference_summary = None
        if inference_job_yamls:
            print_with_id("Loading inference summary")
            inference_summary = inference_job_yamls.summary_from_model(
                model_id, test_params.inference_tests, test_params.component_names
            )

        print_with_id("Adding to Spreadsheet")
        entries = ResultsSpreadsheet()
        entries.set_model_metadata(
            model_id,
            model_info.domain,
            model_info.use_case,
            _get_static_tags(model_info),
            default_quantized_precision=None,
            default_device=model_info.devices[0],
        )
        entries.append_model_summary_entries(
            model_id,
            test_params.profile_tests,
            test_params.component_names,
            quantize_summary,
            compile_summary,
            link_summary,
            profile_summary,
            inference_summary,
        )
        return entries


if __name__ == "__main__":
    args = parse_args()
    static_model_dir: Path = args.static_models_dir

    # Verify args are compatible with the chosen deployment.
    using_prod_hub = deployment_is_prod(args.deployment)
    if not using_prod_hub and args.sync_code_gen:
        print("Warning: Can't sync code gen if deployment is not prod.")
        args.sync_code_gen = False

    # If job ids file isn't specified, check the artifacts dir in CI
    # Or the scorecard intermediates if being run locally.
    jobs_dir = None
    args.quantize_ids = args.quantize_ids or str(get_quantize_job_ids_file(jobs_dir))
    args.compile_ids = args.compile_ids or str(get_compile_job_ids_file(jobs_dir))
    args.link_ids = args.link_ids or str(get_link_job_ids_file(jobs_dir))
    args.profile_ids = args.profile_ids or str(get_profile_job_ids_file(jobs_dir))
    args.inference_ids = args.inference_ids or str(get_inference_job_ids_file(jobs_dir))

    os.makedirs(args.artifacts_dir, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # List of models for which to generate perf.
    pytorch_models, static_models = validate_and_split_enabled_models(
        args.models, static_model_dir
    )
    all_models = SpecialModelSetting.ALL in args.models
    model_list = sorted(pytorch_models.union(static_models))

    # Load datestr
    date = DateFormatEnvvar.parse(args.date, args.date_format)

    # Set client to use target deployment
    set_default_hub_client(get_scorecard_client_or_raise(args.deployment))

    # Load Base YAMLs
    if using_prod_hub:
        # Load previous scorecard state
        quantize_job_yamls = QuantizeScorecardJobYaml.from_file(
            QUANTIZE_YAML_BASE, create_empty_if_no_file=True
        )
        compile_job_yamls = CompileScorecardJobYaml.from_file(
            COMPILE_YAML_BASE, create_empty_if_no_file=True
        )
        link_job_yamls = LinkScorecardJobYaml.from_file(
            LINK_YAML_BASE, create_empty_if_no_file=True
        )
        profile_job_yamls = ProfileScorecardJobYaml.from_file(
            PROFILE_YAML_BASE, create_empty_if_no_file=True
        )
        inference_job_yamls = InferenceScorecardJobYaml.from_file(
            INFERENCE_YAML_BASE, create_empty_if_no_file=True
        )

        # Erase jobs for models we're collecting results for, if applicable
        if args.ignore_existing_intermediate_jobs:
            clear_jobs(quantize_job_yamls, model_list, all_models)
            clear_jobs(compile_job_yamls, model_list, all_models)
            clear_jobs(link_job_yamls, model_list, all_models)
            clear_jobs(profile_job_yamls, model_list, all_models)
            clear_jobs(inference_job_yamls, model_list, all_models)
    else:
        # Previous scorecard state is applicable only on prod
        quantize_job_yamls = QuantizeScorecardJobYaml()
        compile_job_yamls = CompileScorecardJobYaml()
        link_job_yamls = LinkScorecardJobYaml()
        profile_job_yamls = ProfileScorecardJobYaml()
        inference_job_yamls = InferenceScorecardJobYaml()

    # Append additional YAMLs
    for quantize_yaml_path in args.quantize_ids.split(",") if args.quantize_ids else []:
        quantize_job_yamls.update(
            QuantizeScorecardJobYaml.from_file(quantize_yaml_path)
        )

    for compile_yaml_path in args.compile_ids.split(",") if args.compile_ids else []:
        compile_job_yamls.update(CompileScorecardJobYaml.from_file(compile_yaml_path))
    for link_yaml_path in args.link_ids.split(",") if args.link_ids else []:
        link_job_yamls.update(LinkScorecardJobYaml.from_file(link_yaml_path))
    for profile_yaml_path in args.profile_ids.split(",") if args.profile_ids else []:
        profile_job_yamls.update(ProfileScorecardJobYaml.from_file(profile_yaml_path))
    for inference_yaml_path in (
        args.inference_ids.split(",") if args.inference_ids else []
    ):
        inference_job_yamls.update(
            InferenceScorecardJobYaml.from_file(inference_yaml_path)
        )

    # Extract Data from Models
    if len(model_list) > 1:
        # Use multiprocessing for multiple models because getting jobs from Hub is slow
        pool = multiprocessing.Pool(processes=15)
        model_summaries = pool.starmap(
            process_model,
            zip(
                model_list,
                cycle([args.deployment]),
                cycle([static_model_dir]),
                cycle([quantize_job_yamls]),
                cycle([compile_job_yamls]),
                cycle([link_job_yamls]),
                cycle([profile_job_yamls]),
                cycle([inference_job_yamls]),
                cycle([args.gen_csv]),
                cycle([args.sync_code_gen]),
                cycle([args.gen_perf_summary]),
                cycle([using_prod_hub]),
            ),
        )
        pool.close()
    else:
        # Single model option for that allows breakpoints
        model_summaries = [
            process_model(
                model_list[0],
                args.deployment,
                static_model_dir,
                quantize_job_yamls,
                compile_job_yamls,
                link_job_yamls,
                profile_job_yamls,
                inference_job_yamls,
                args.gen_csv,
                args.sync_code_gen,
                args.gen_perf_summary,
                using_prod_hub,
            )
        ]

    # Collect Model Results
    perf_report = PerformanceDiff() if args.gen_perf_summary else None
    spreadsheet = ResultsSpreadsheet() if args.gen_csv else None
    if spreadsheet is not None:
        spreadsheet.set_date(date)
        # Tableau wants to differentiate between different types of scorecards
        # So mark them as such in the branch column.
        branch = args.branch
        precisions = args.precisions
        for precision in precisions:
            if isinstance(precision, SpecialPrecisionSetting):
                branch += f" - {precision.value}"
        spreadsheet.set_branch(branch)

    for model_id, (
        model_spreadsheet,
        prev_model_card,
        curr_model_card,
    ) in zip(model_list, model_summaries, strict=False):
        # Combine model spreadsheet with group spreadsheet
        if spreadsheet is not None:
            spreadsheet.combine(model_spreadsheet)

        # Update performance report with model card diff
        if perf_report is not None:
            # Summary is made between the existing perf.yaml and the newly
            # created model card.
            perf_report.update_summary(
                model_id,
                previous_report=prev_model_card,
                new_report=curr_model_card,
            )

    # Write spreadsheet to disk
    if spreadsheet is not None:
        summary_path = os.path.join(
            args.artifacts_dir, f"scorecard-summary-{now_str}.csv"
        )
        spreadsheet.to_csv(summary_path)
        print(f"Spreadsheet written to {os.path.realpath(summary_path)}")

    # Write summary to disk
    if perf_report is not None:
        report_path = os.path.join(
            args.artifacts_dir, f"performance-summary-{now_str}.txt"
        )
        perf_report.dump_summary(report_path)

    # Write jobs and environment to intermediates folder
    if using_prod_hub:
        quantize_job_yamls.to_file(QUANTIZE_YAML_BASE)
        compile_job_yamls.to_file(COMPILE_YAML_BASE)
        link_job_yamls.to_file(LINK_YAML_BASE)
        profile_job_yamls.to_file(PROFILE_YAML_BASE)
        inference_job_yamls.to_file(INFERENCE_YAML_BASE)
        print(f"Quantize Job IDs written to {QUANTIZE_YAML_BASE}")
        print(f"Compile Job IDs written to {COMPILE_YAML_BASE}")
        print(f"Link Job IDs written to {LINK_YAML_BASE}")
        print(f"Profile Job IDs written to {PROFILE_YAML_BASE}")
        print(f"Inference Job IDs written to {INFERENCE_YAML_BASE}")

        try:
            shutil.copy(get_tool_versions_file(create=False), TOOL_VERSIONS_BASE)
            print(f"Tool versions written to {TOOL_VERSIONS_BASE}")
        except (shutil.SameFileError, FileNotFoundError):
            pass

        try:
            shutil.copy(get_environment_file(create=False), ENVIRONMENT_ENV_BASE)
            print(f"Test envvars written to to {ENVIRONMENT_ENV_BASE}")
        except (shutil.SameFileError, FileNotFoundError):
            pass

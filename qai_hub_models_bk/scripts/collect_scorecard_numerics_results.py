# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

import pandas as pd

from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.scorecard.envvars import (
    ArtifactsDirEnvvar,
    DeploymentEnvvar,
    EnabledDevicesEnvvar,
    EnabledModelsEnvvar,
    EnabledPathsEnvvar,
    EnabledPrecisionsEnvvar,
    SpecialModelSetting,
    get_default_hub_deployment,
)
from qai_hub_models.scorecard.results.code_gen import (
    remove_numerics_failures,
    remove_perf_failures,
    update_code_gen_accuracy_failure_reasons,
    update_model_publish_status,
)
from qai_hub_models.scorecard.results.numerics_diff import NumericsDiff
from qai_hub_models.scorecard.results.yaml import ACCURACY_CSV_BASE
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.utils.hub_clients import deployment_is_prod
from qai_hub_models.utils.numerics_yaml import (
    QAIHMModelNumerics,
    create_numerics_yaml,
    get_chipset_registry,
)
from qai_hub_models.utils.testing_async_utils import get_accuracy_csv_path


def _merge_existing_accuracy_data(
    new_df: pd.DataFrame, models: set[str]
) -> pd.DataFrame:
    old_df = pd.read_csv(ACCURACY_CSV_BASE)
    old_df = old_df[~old_df.model_id.isin(models)]
    return pd.concat([old_df, new_df])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    EnabledModelsEnvvar.add_arg(parser, {SpecialModelSetting.PYTORCH})
    DeploymentEnvvar.add_arg(parser, default=get_default_hub_deployment())
    EnabledPrecisionsEnvvar.add_arg(parser)
    EnabledPathsEnvvar.add_arg(parser)
    EnabledDevicesEnvvar.add_arg(parser)
    ArtifactsDirEnvvar.add_arg(parser)
    parser.add_argument(
        "--accuracy-csv-path",
        type=str,
        default=str(get_accuracy_csv_path()),
    )
    parser.add_argument(
        "--summary-path", type=str, default=os.path.join("build", "numerics_diff.txt")
    )
    parser.add_argument(
        "--sync-code-gen",
        action="store_true",
        help="Sync code generation YAML with failures & successes in the numerics YAML.",
    )
    return parser.parse_args()


def main() -> None:
    # Verify args are compatible with the chosen deployment.
    args = parse_args()
    pytorch_models, _ = validate_and_split_enabled_models(args.models)
    using_prod_hub = deployment_is_prod(args.deployment)
    if not using_prod_hub and args.sync_code_gen:
        print("Warning: Can't sync code gen if deployment is not prod.")
        args.sync_code_gen = False

    accuracy_path = Path(args.accuracy_csv_path)
    if not accuracy_path.exists() or accuracy_path.stat().st_size == 0:
        print("No accuracy CSV found. Not updating any numerics files.")
        return

    accuracy_df = pd.read_csv(args.accuracy_csv_path)
    chipset_registry = get_chipset_registry()
    global_diff = NumericsDiff()
    for model_id in sorted(pytorch_models):
        try:
            model_info = QAIHMModelInfo.from_model(model_id)
            if (
                model_info.code_gen_config.skip_hub_tests_and_scorecard
                or model_info.code_gen_config.skip_scorecard
                or model_info.code_gen_config.freeze_perf_yaml
            ):
                continue

            model_diff = NumericsDiff()
            numerics = create_numerics_yaml(
                model_id,
                accuracy_df,
                chipset_registry,
                model_diff,
                benchmark=model_info.numerics_benchmark,
            )
            global_diff.merge_from(model_diff)
            if numerics is None:
                QAIHMModelNumerics().to_model_yaml(model_id)  # deletes existing file
                continue

            if numerics.metrics:
                # Update failure reasons according to what NumericsDiff says is above the acceptable accuracy threshold.
                update_code_gen_accuracy_failure_reasons(
                    model_id, model_info.code_gen_config, model_diff
                )

                # Update numerics.yaml to remove failing paths
                numerics = remove_numerics_failures(
                    numerics, model_info.code_gen_config.disabled_paths
                )

                if args.sync_code_gen and using_prod_hub:
                    # If sync-code-gen is on, then we save the updated failure reasons to disk.
                    model_info.code_gen_config.to_model_yaml(model_id)

                    # Update perf.yaml to remove failing paths
                    perf = remove_perf_failures(
                        perf=QAIHMModelPerf.from_model(model_id, not_exists_ok=True),
                        failure_reason=model_info.code_gen_config.disabled_paths,
                    )
                    perf.to_model_yaml(model_id)

                    # Un-publish or re-publish the model if needed by updating info.yaml.
                    if update_model_publish_status(model_info):
                        model_info.to_model_yaml(write_code_gen=False)

            numerics.to_model_yaml(model_id)
            print(f"{model_id} numerics update complete")
        except Exception as e:
            raise ValueError(
                f"Failed to collect accuracy results for {model_id}"
            ) from e

    # Write diff to artifacts folder
    os.makedirs(args.artifacts_dir, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    report_path = os.path.join(args.artifacts_dir, f"numerics-summary-{now_str}.txt")
    global_diff.dump_summary(report_path)

    # Write accuracy to intermediates folder
    if args.sync_code_gen and using_prod_hub:
        if args.models not in (
            {SpecialModelSetting.PYTORCH},
            {SpecialModelSetting.ALL},
        ):
            accuracy_df = _merge_existing_accuracy_data(accuracy_df, pytorch_models)
        accuracy_df.to_csv(ACCURACY_CSV_BASE, index=False)


if __name__ == "__main__":
    main()

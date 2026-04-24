# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

from qai_hub_models.configs.info_yaml import QAIHMModelInfo
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
    remove_asset_failures,
)
from qai_hub_models.scorecard.results.yaml import ScorecardAssetYaml
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.utils.testing_async_utils import (
    get_release_assets_file,
)
from qai_hub_models.utils.testing_export_eval import QAIHMModelReleaseAssets


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
        "--release-assets-yaml", type=str, default=str(get_release_assets_file())
    )
    return parser.parse_args()


def main() -> None:
    # Verify args are compatible with the chosen deployment.
    args = parse_args()
    pytorch_models, _ = validate_and_split_enabled_models(args.models)

    assets_path = Path(args.release_assets_yaml)
    if not assets_path.exists() or assets_path.stat().st_size == 0:
        print("No scorecard release assets found. Not updating any files.")
        return

    modified_files: list[str] = []
    scorecard_assets = ScorecardAssetYaml.from_yaml(args.release_assets_yaml)
    for model_id in sorted(pytorch_models):
        try:
            model_info = QAIHMModelInfo.from_model(model_id)
            if (
                model_info.code_gen_config.skip_hub_tests_and_scorecard
                or model_info.code_gen_config.skip_scorecard
                or model_info.code_gen_config.freeze_perf_yaml
            ):
                continue

            if scorecard_model_assets := scorecard_assets.models.get(model_id):
                # Remove assets for unsupported paths
                scorecard_model_assets = remove_asset_failures(
                    scorecard_model_assets, model_info.code_gen_config.disabled_paths
                )

                # Write updated assets
                modified_files.append(
                    str(scorecard_model_assets.to_model_yaml(model_id))
                )
            else:
                QAIHMModelReleaseAssets().to_model_yaml(
                    model_id
                )  # deletes existing file
        except Exception as e:
            raise ValueError(
                f"Failed to collect accuracy results for {model_id}"
            ) from e


if __name__ == "__main__":
    main()

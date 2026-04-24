#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""Dumps the tool versions used by the current scorecard configuration to the test artifacts directory."""

import argparse

from qai_hub_models.scorecard.envvars import (
    ArtifactsDirEnvvar,
    DeploymentEnvvar,
    EnabledPathsEnvvar,
    QAIRTVersionEnvvar,
)
from qai_hub_models.scorecard.results.yaml import ToolVersionsByPathYaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry failed profile jobs that failed due to known flaky reasons"
    )
    ArtifactsDirEnvvar.add_arg(parser)
    DeploymentEnvvar.add_arg(parser)
    QAIRTVersionEnvvar.add_arg(parser)
    EnabledPathsEnvvar.add_arg(parser)
    return parser.parse_args()


def main() -> None:
    parse_args()
    ToolVersionsByPathYaml.from_profile_paths().to_dir(ArtifactsDirEnvvar.get())


if __name__ == "__main__":
    main()

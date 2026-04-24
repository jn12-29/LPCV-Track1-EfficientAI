# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.scripts.generate_global_readme import generate_global_readme

TEST_MODELS = ["resnet50", "easyocr", "litehrnet"]


def test_generate_global_readme() -> None:
    models = [QAIHMModelInfo.from_model(model_id) for model_id in TEST_MODELS]

    with TemporaryDirectory() as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "src"))
        pypi_dir = os.path.join(tmp_dir, "src")
        repo_path, pypi_path = generate_global_readme(
            models, Path(tmp_dir), Path(pypi_dir)
        )

        with open(repo_path) as f:
            repo_readme = f.read()
        with open(pypi_path) as f:
            pypi_readme = f.read()

    with open(
        os.path.join(os.path.dirname(__file__), "summary_table_expected.md")
    ) as expected_f:
        expected_table = expected_f.read()

    # Repo README has relative links
    assert expected_table in repo_readme

    # PyPI README has model directory with package names instead of relative links
    with open(
        os.path.join(os.path.dirname(__file__), "pypi_table_expected.md")
    ) as expected_f:
        expected_pypi_table = expected_f.read()

    assert expected_pypi_table in pypi_readme

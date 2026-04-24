# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import platform
import re
from pathlib import Path

from packaging import version

from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    QAIHM_MODELS_ROOT,
    QAIHM_PACKAGE_ROOT,
)


def main() -> None:
    # collect all requirements.txt files to consider
    files = get_files_to_process()

    # collect global requirements from these files
    requirements = get_requirements(files)

    # overwrite global_requirements.txt
    write_to_file(requirements)


def get_files_to_process() -> list[Path]:
    files: list[Path] = [
        QAIHM_PACKAGE_ROOT / "requirements-dev.txt",
        QAIHM_PACKAGE_ROOT / "requirements.txt",
    ]
    current_version = version.parse(platform.python_version())
    for model_name in MODEL_IDS:
        file = QAIHM_MODELS_ROOT / model_name / "requirements.txt"

        if not file.exists():
            continue

        # skip model if it has opted out via "global_requirements_incompatible: True"
        code_gen_yaml_file = QAIHM_MODELS_ROOT / model_name / "code-gen.yaml"
        if code_gen_yaml_file.exists():
            code_gen_config = load_yaml(code_gen_yaml_file)
            if code_gen_config.get("global_requirements_incompatible", False):
                continue

            # Don't include models that aren't applicable for this python version
            ge_version = code_gen_config.get(
                "python_version_greater_than_or_equal_to", None
            )
            l_version = code_gen_config.get("python_version_less_than", None)
            if ge_version and current_version < version.parse(ge_version):
                continue
            if l_version and current_version >= version.parse(l_version):
                continue

        files.append(file)

    return files


def get_requirements(files: list[Path]) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for file in files:
        with open(file) as f:
            for line in f:
                line = line.strip()
                # 'botocore>=1.34,<1.36 # comment' -> 'botocore', '>=1.34,<1.36 '
                result = re.search(
                    r"([0-9a-zA-Z\.\_\-\[\]]+)\s*((==|<=|>=|~=|[><])[^#]*)", line
                )
                if not result:
                    raise ValueError(f"Unsupported version string: {line}")
                package = result.group(1)
                version = result.group(2).strip()

                if requirements.get(package, version) != version:
                    raise ValueError(
                        f"ERROR: Version conflict for {package} in {file}: {requirements[package]} vs {version}. Use option global_requirements_incompatible: True in code-gen.yaml to bypass this check if needed."
                    )
                requirements[package] = version

    return requirements


def write_to_file(requirements: dict[str, str]) -> None:
    header = """# If you:
# - Install requirements.txt
# - Then install this requirements file
# That should create an environment that works for every single model.

# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

"""
    with open(QAIHM_PACKAGE_ROOT / "global_requirements.txt", "w") as file:
        file.write(header)
        for key in sorted(requirements.keys()):
            version = requirements[key]
            file.write(f"{key}{version}\n")


if __name__ == "__main__":
    main()

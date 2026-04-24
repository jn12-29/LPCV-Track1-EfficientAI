# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, UNPUBLISHED_MODEL_WARNING
from qai_hub_models.utils.path_helpers import MODEL_IDS

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)
README_TEMPLATE = jinja_env.get_template("model_readme_template.j2")


def main() -> None:
    """Generate a README for each model, and save it in <model_dir>/README.md."""
    for model_id in MODEL_IDS:
        generate_and_write_model_readme(model_id)


def get_shared_template_args(model_info: QAIHMModelInfo) -> dict[str, Any]:
    """
    Get template arguments shared between regular README and HF model card.

    Parameters
    ----------
    model_info
        Model info object.

    Returns
    -------
    template_args: dict[str, Any]
        Dictionary of shared template arguments.
    """
    model_code_gen = model_info.code_gen_config

    return {
        # Model info
        "model_id": model_info.id,
        "model_name": model_info.name,
        "model_description": model_info.description,
        "model_url": f"{ASSET_CONFIG.models_website_url}/models/{model_info.id}",
        "additional_readme_section": model_code_gen.additional_readme_section.replace(
            "{genie_url}", ASSET_CONFIG.genie_url
        ),
        # Source and references
        "source_repo": model_info.source_repo,
        "research_paper_title": model_info.research_paper_title,
        "research_paper_url": model_info.research_paper,
        "license_url": model_info.license,
        # Flags
        "include_gen_ai_terms": model_info.is_gen_ai_model,
    }


def generate_and_write_model_readme(model_id: str) -> Path:
    """Generate a README for this model from the jinja template and write it."""
    model_info = QAIHMModelInfo.from_model(model_id)
    model_code_gen = model_info.code_gen_config

    # Convert precisions to strings for Jinja template
    supported_precisions = None
    if model_code_gen.can_use_quantize_job and model_code_gen.supported_precisions:
        supported_precisions = [str(p) for p in model_code_gen.supported_precisions]

    template_vars = get_shared_template_args(model_info)
    template_vars.update(
        {
            # Model info
            "model_status": model_info.status.value,
            "unpublished_model_warning": UNPUBLISHED_MODEL_WARNING,
            "model_headline": model_info.headline.strip("."),
            "model_package": model_info.get_package_name(),
            "supported_precisions": supported_precisions,
            # Package installation
            "has_model_requirements": model_info.has_model_requirements(),
            "pip_pre_build_reqs": model_code_gen.pip_pre_build_reqs,
            "pip_install_flags": model_code_gen.pip_install_flags,
            "pip_install_flags_gpu": model_code_gen.pip_install_flags_gpu,
            "python_version_gte": model_code_gen.python_version_greater_than_or_equal_to,
            "python_version_lt": model_code_gen.python_version_less_than,
            # Flags
            "include_example_and_usage": not model_code_gen.skip_example_usage,
            "has_on_target_demo": model_code_gen.has_on_target_demo,
            # llama.cpp commands for LLM models
            "llama_cpp_cpu_command": model_code_gen.llama_cpp_cpu_command,
            "llama_cpp_gpu_command": model_code_gen.llama_cpp_gpu_command,
            "llama_cpp_npu_command": model_code_gen.llama_cpp_npu_command,
            "llama_cpp_model_url": model_info.llm_details.llama_cpp_model_url
            if model_info.llm_details
            else None,
            # System-level dependencies installation instructions
            "readme_install_system_deps": model_code_gen.readme_install_system_deps,
        }
    )

    readme_path = model_info.get_readme_path()
    with open(readme_path, "w") as readme_file:
        readme_file.write(README_TEMPLATE.module.render(**template_vars))  # type: ignore[attr-defined]
    return readme_path


if __name__ == "__main__":
    main()

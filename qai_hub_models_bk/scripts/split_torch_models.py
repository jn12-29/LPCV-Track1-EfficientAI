# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.scorecard.envvars import EnabledModelsEnvvar
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)

# Default max number of PyTorch models per split when auto-splitting
MAX_PT_MODELS_PER_SPLIT = 30


def split_torch_models(
    models: set,
    max_pt_splits: int | None = None,
    max_pt_models_per_split: int = MAX_PT_MODELS_PER_SPLIT,
) -> list[dict[str, str]]:
    """
    Split models into chunks for parallel processing.

    Static models are all grouped into one split named "static".
    Torch models are split into multiple chunks.

    Parameters
    ----------
    models
        Set of model IDs or special settings (from EnabledModelsEnvvar.get())
    max_pt_splits
        Maximum number of splits to create for torch models. If None, automatically
        calculate based on max_pt_models_per_split.
    max_pt_models_per_split
        Maximum number of models per split when auto-calculating num_splits.
        If num_models / max_pt_splits >> max_pt_models_per_split, max_pt_splits will be used instead to avoid creating too many splits.

    Returns
    -------
    list[dict[str, str]]
        List of dicts with 'split_name' and 'models' keys for each split.
    """
    torch_models, static_models = validate_and_split_enabled_models(models)

    splits = []

    # Add all static models as one split
    if static_models:
        splits.append(
            {
                "split_name": "static",
                "models": ",".join(sorted(static_models)),
            }
        )

    # Split torch models into chunks
    all_torch_models = sorted(torch_models)
    if all_torch_models:
        num_splits = math.ceil(len(all_torch_models) / max_pt_models_per_split)
        if max_pt_splits is not None:
            num_splits = min(num_splits, max_pt_splits)

        # Divide the AOT and JIT models separately to ensure they are distributed as evenly as possible across splits,
        # since AOT models take much longer to compile and we want to avoid having one split with mostly AOT models and another with mostly JIT models
        all_models_jit = []
        all_models_aot = []
        for model in all_torch_models:
            if QAIHMModelCodeGen.from_model(model).requires_aot_prepare:
                all_models_aot.append(model)
            else:
                all_models_jit.append(model)
        jit_split_size = math.ceil(len(all_models_jit) / num_splits)
        aot_split_size = math.ceil(len(all_models_aot) / num_splits)

        # Create splits by taking chunks of the JIT and AOT models separately, then combining them
        for i in range(num_splits):
            jit_start_idx = i * jit_split_size
            jit_end_idx = min((i + 1) * jit_split_size, len(all_models_jit))

            aot_start_idx = i * aot_split_size
            aot_end_idx = min((i + 1) * aot_split_size, len(all_models_aot))

            jit_models_in_split = all_models_jit[jit_start_idx:jit_end_idx]
            aot_models_in_split = all_models_aot[aot_start_idx:aot_end_idx]
            models_in_split = (
                aot_models_in_split + jit_models_in_split
            )  # AOT models first because they take longer to compile

            if models_in_split:
                splits.append(
                    {
                        "split_name": f"torch_{i + 1}_of_{num_splits}",
                        "models": ",".join(models_in_split),
                    }
                )

    # If there's only one split and it's not static, set the name to torch
    if len(splits) == 1 and splits[0]["split_name"] != "static":
        splits[0]["split_name"] = "torch"

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split models into chunks for parallel scorecard runs"
    )
    EnabledModelsEnvvar.add_arg(parser)
    parser.add_argument(
        "--max-num-pt-splits",
        type=int,
        default=None,
        help="Maximum of PyTorch model splits to create.",
    )
    parser.add_argument(
        "--max-models-per-pt-split",
        type=int,
        default=MAX_PT_MODELS_PER_SPLIT,
        help=f"Maximum PyTorch models per split (default: {MAX_PT_MODELS_PER_SPLIT})",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "github"],
        default="json",
        help="Output format: 'json' for pretty JSON, 'github' for GitHub Actions matrix format",
    )

    args = parser.parse_args()
    splits = split_torch_models(
        args.models, args.max_num_pt_splits, args.max_models_per_pt_split
    )
    if args.output_format == "github":
        # Output as a single line JSON for GitHub Actions
        print(json.dumps(splits))
    else:
        # Pretty print JSON
        print(json.dumps(splits, indent=2))


if __name__ == "__main__":
    main()

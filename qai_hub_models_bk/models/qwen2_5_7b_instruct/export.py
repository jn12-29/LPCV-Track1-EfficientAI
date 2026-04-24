# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import warnings

from qai_hub_models.models._shared.llm.export import export_model, get_llm_parser
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.models.qwen2_5_7b_instruct import MODEL_ID, Model
from qai_hub_models.models.qwen2_5_7b_instruct.model import (
    DEFAULT_PRECISION,
    MODEL_ASSET_VERSION,
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
)

DEFAULT_EXPORT_DEVICE = "Snapdragon 8 Elite QRD"


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = get_llm_parser(
        supported_precision_runtimes={DEFAULT_PRECISION: [TargetRuntime.GENIE]},
        model_cls=Model,  # type: ignore[arg-type]
        default_export_device=DEFAULT_EXPORT_DEVICE,
        default_precision=DEFAULT_PRECISION,
    )
    args = parser.parse_args()
    export_model(
        model_cls=Model,  # type: ignore[arg-type]
        model_id=MODEL_ID,
        model_asset_version=MODEL_ASSET_VERSION,
        num_splits=NUM_SPLITS,
        num_layers_per_split=NUM_LAYERS_PER_SPLIT,
        **vars(args),
    )


if __name__ == "__main__":
    main()

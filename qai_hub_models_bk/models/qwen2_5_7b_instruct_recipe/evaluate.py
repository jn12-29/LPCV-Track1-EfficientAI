# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.llm.evaluate import llm_evaluate
from qai_hub_models.models.qwen2_5_7b_instruct_recipe import (
    FP_Model,
    Model,
    QNN_Model,
)
from qai_hub_models.models.qwen2_5_7b_instruct_recipe.model import SUPPORTED_PRECISIONS

if __name__ == "__main__":
    llm_evaluate(
        quantized_model_cls=Model,
        fp_model_cls=FP_Model,
        qnn_model_cls=QNN_Model,
        supported_precisions=SUPPORTED_PRECISIONS,
    )

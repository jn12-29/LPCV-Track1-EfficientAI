# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.llama3.model import END_TOKENS
from qai_hub_models.models._shared.llm.demo import llm_chat_demo
from qai_hub_models.models._shared.llm.model import LLM_QNN, LLM_AIMETOnnx, LLMBase
from qai_hub_models.models.llama_v3_2_1b_instruct2 import MODEL_ID
from qai_hub_models.models.llama_v3_2_1b_instruct2.model import (
    HF_REPO_NAME,
    SUPPORTED_PRECISIONS,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec

HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"


def llama_3_2_1b_chat_demo(
    model_cls: type[LLM_AIMETOnnx] = QuantizedSplitModelWrapper,
    fp_model_cls: type[LLMBase] = FPSplitModelWrapper,
    qnn_model_cls: type[LLM_QNN] = LLM_QNN,
    model_id: str = MODEL_ID,
    end_tokens: set = END_TOKENS,
    hf_repo_name: str = HF_REPO_NAME,
    hf_repo_url: str = HF_REPO_URL,
    default_prompt: str | None = None,
    test_checkpoint: CheckpointSpec | None = None,
) -> None:
    llm_chat_demo(
        model_cls=model_cls,
        fp_model_cls=fp_model_cls,
        qnn_model_cls=qnn_model_cls,
        model_id=model_id,
        end_tokens=end_tokens,
        hf_repo_name=hf_repo_name,
        hf_repo_url=hf_repo_url,
        supported_precisions=SUPPORTED_PRECISIONS,
        default_prompt=default_prompt,
        test_checkpoint=test_checkpoint,
    )


if __name__ == "__main__":
    llama_3_2_1b_chat_demo()

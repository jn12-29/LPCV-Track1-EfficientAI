# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Llama 3.2 1B Instruct2 - PreSplit-Part architecture for LLM deployment.

This module provides:
- PreSplit classes (FP and Quantizable) with class-level caching for model + ONNX splitting
- Unified Part classes that handle both FP and Quantizable modes based on precision
- Collection class for deploying the model as 3 splits
"""

from .model import (  # noqa: F401
    HF_REPO_NAME,
    HIDDEN_SIZE,
    MIN_MEMORY_RECOMMENDED,
    MODEL_ID,
    NUM_ATTN_HEADS,
    NUM_KEY_VALUE_HEADS,
    NUM_LAYERS,
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
    FPSplitModelWrapper,
    Llama3_2_1B_Collection,
    Llama3_2_1B_Part1_Of_3,
    Llama3_2_1B_Part2_Of_3,
    Llama3_2_1B_Part3_Of_3,
    Llama3_2_1B_PartBase,
    Llama3_2_1B_PreSplit,
    Llama3_2_1B_QuantizablePreSplit,
    QuantizedSplitModelWrapper,
    SplitForwardMixin,
)

Model = Llama3_2_1B_Collection

__all__ = [
    "HF_REPO_NAME",
    "HIDDEN_SIZE",
    "MIN_MEMORY_RECOMMENDED",
    "NUM_ATTN_HEADS",
    "NUM_KEY_VALUE_HEADS",
    "NUM_LAYERS",
    "NUM_LAYERS_PER_SPLIT",
    "NUM_SPLITS",
    "FPSplitModelWrapper",
    "Llama3_2_1B_Collection",
    "Llama3_2_1B_Part1_Of_3",
    "Llama3_2_1B_Part2_Of_3",
    "Llama3_2_1B_Part3_Of_3",
    "Llama3_2_1B_PartBase",
    "Llama3_2_1B_PreSplit",
    "Llama3_2_1B_QuantizablePreSplit",
    "QuantizedSplitModelWrapper",
    "SplitForwardMixin",
]

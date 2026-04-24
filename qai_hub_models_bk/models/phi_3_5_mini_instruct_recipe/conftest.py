# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from qai_hub_models.models._shared.llm.perf_collection import LLMPerfConfig


@pytest.fixture(scope="session")
def llm_perf_config() -> LLMPerfConfig:
    from qai_hub_models.models._shared.llm.perf_collection import LLMPerfConfig

    return LLMPerfConfig.from_environment()

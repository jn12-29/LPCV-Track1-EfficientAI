# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.llm.app import ChatApp as App  # noqa: F401
from qai_hub_models.models._shared.phi.model import (  # noqa: F401
    Phi3PositionProcessor as PositionProcessor,
)

from .model import MODEL_ID  # noqa: F401
from .model import Phi4Mini as FP_Model  # noqa: F401
from .model import Phi4Mini_AIMETOnnx as Model  # noqa: F401
from .model import Phi4Mini_QNN as QNN_Model  # noqa: F401

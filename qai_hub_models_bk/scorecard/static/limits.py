# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.scorecard.device import ScorecardDevice

# Number of AI Hub Workbench device jobs for static models that can be submitted per-BU.
BU_DEVICE_JOB_LIMIT = 100

# Number of AI Hub Workbench device jobs for static models that can be submitted for each device form factor per-BU.
BU_DEVICE_JOB_LIMIT_BY_FORM_FACTOR: dict[ScorecardDevice.FormFactor, int] = {
    ScorecardDevice.FormFactor.AUTO: 50
}

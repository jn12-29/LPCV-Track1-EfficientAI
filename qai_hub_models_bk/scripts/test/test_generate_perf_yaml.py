# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os

import pytest

from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.yaml import ProfileScorecardJobYaml
from qai_hub_models.utils.collection_model_helpers import get_components
from qai_hub_models.utils.hub_clients import (
    deployment_is_prod,
    get_default_hub_deployment,
)

EXPECTED_MODEL_CARD = QAIHMModelPerf.from_yaml(
    os.path.join(os.path.dirname(__file__), "perf_gt.yaml")
)


def test_generate_perf() -> None:
    # Verify the model card is made correctly given the info
    if not deployment_is_prod(get_default_hub_deployment() or ""):
        pytest.skip(
            f"This test uses jobs only accessible on production AI Hub Workbench. Enabled deployment: {get_default_hub_deployment()}"
        )

    job_ids = ProfileScorecardJobYaml.from_file(
        os.path.join(os.path.dirname(__file__), "profile_job_ids.yaml")
    )
    model_info = QAIHMModelInfo.from_model("trocr")
    component_names = get_components(model_info.id)

    paramaterizations: list[
        tuple[Precision, ScorecardProfilePath, ScorecardDevice]
    ] = []
    for device in ScorecardDevice.all_devices():
        for path in ScorecardProfilePath:
            for precision in model_info.code_gen_config.supported_precisions:
                if path.supports_precision(precision) and device.npu_supports_precision(
                    precision
                ):
                    paramaterizations.append((precision, path, device))  # noqa: PERF401

    model_perf = job_ids.summary_from_model(
        model_info.id,
        paramaterizations,
        component_names,
    )
    model_card = model_perf.get_perf_card()
    assert model_card == EXPECTED_MODEL_CARD

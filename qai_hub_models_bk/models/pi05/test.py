# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os

# Disable dynamo in lerobot forward
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from qai_hub_models.models.pi05 import Model
from qai_hub_models.models.pi05.demo import (
    DATASET_REPO_ID,
    HF_MODEL_ID,
    build_action_seq_from_first_episode,
    compute_sequence_metrics,
    load_one_batch,
    run_pi05_on_sample,
)


def test_one_sample() -> None:
    """
    Run a single forward pass and assert the predicted actions are close to

    1. Original pytorch impl (in element-wise comparison)
    2. Ground truth (in terms of rmse, sqnr)
    """
    torch.manual_seed(1234)
    # Load pretrained model and policy.
    model = Model.from_pretrained(HF_MODEL_ID)
    device = "cpu"
    policy = PI05Policy.from_pretrained(HF_MODEL_ID).to(device).eval()

    # Prepare one batch.
    batch, postprocessor, dataset = load_one_batch(
        dataset_repo_id=DATASET_REPO_ID,
        pi05_config=policy.config,
        batch_size=1,
        device=device,
    )

    # Prepare a deterministic noise tensor shared by both paths.
    cfg = policy.config
    bsize = 1
    tcfg = cfg.chunk_size
    dcfg = cfg.max_action_dim
    noise_shape = (bsize, tcfg, dcfg)

    noise_fixed = torch.normal(
        mean=0.0,
        std=1.0,
        size=noise_shape,
        dtype=torch.float32,
        device=device,
    )

    # Predict actions via the demo helper (app-like path) with fixed noise.
    pred_actions_app = run_pi05_on_sample(
        model=model,
        policy=policy,
        batch=batch,
        post_processor=postprocessor,
        noise=noise_fixed,
    )

    # Direct policy path with the identical noise returned by a patched
    # sampler. Count calls and validate shape and device.
    call_counter: dict[str, int] = {"n": 0}

    def _fixed_sample_noise(
        self: object, shape: tuple[int, int, int], dev: torch.device
    ) -> torch.Tensor:
        call_counter["n"] += 1
        assert tuple(shape) == noise_shape
        assert dev.type == torch.device(device).type
        return noise_fixed

    policy.model.sample_noise = _fixed_sample_noise.__get__(
        policy.model, type(policy.model)
    )

    pred_actions_policy = policy.predict_action_chunk(batch)
    pred_actions_policy = postprocessor(pred_actions_policy)

    # Ensure the sampler was called exactly once by the policy path.
    assert call_counter["n"] == 1

    # Parity checks after postprocessing.
    assert tuple(pred_actions_policy.shape) == tuple(pred_actions_app.shape)
    torch.testing.assert_close(
        pred_actions_policy, pred_actions_app, rtol=1e-2, atol=1e-2
    )

    # Build ground-truth sequence after prediction to avoid leakage.
    gt_actions = build_action_seq_from_first_episode(
        dataset=dataset,
        device=torch.device(device),
        seq_len=50,
    )

    # Compute metrics against ground truth using the demo prediction.
    mse, rmse, sqnr = compute_sequence_metrics(
        pred=pred_actions_app,
        gt=gt_actions,
    )

    # Sanity and threshold checks.
    assert pred_actions_app.dim() == 3
    assert gt_actions.dim() == 3
    assert pred_actions_app.shape == gt_actions.shape == (1, 50, 7)
    assert float(mse) < 0.001
    assert float(rmse) < 0.02
    assert float(sqnr) > 25

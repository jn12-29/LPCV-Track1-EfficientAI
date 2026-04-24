# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch

from qai_hub_models.models.dncnn.demo import main as demo_main
from qai_hub_models.models.dncnn.demo import make_noisy_image
from qai_hub_models.models.dncnn.model import DnCNN
from qai_hub_models.utils.testing import skip_clone_repo_check


@skip_clone_repo_check
def test_task() -> None:
    """Verify DnCNN reduces noise (PSNR improvement)."""
    model = DnCNN.from_pretrained()
    noisy_img, clean_img = make_noisy_image()

    # Convert PIL images to tensors [1, 1, H, W]
    noisy = (
        torch.from_numpy(np.array(noisy_img, dtype=np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    clean = (
        torch.from_numpy(np.array(clean_img, dtype=np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    denoised = model(noisy)

    # Compute PSNR before and after
    mse_before = torch.mean((noisy - clean) ** 2).item()
    mse_after = torch.mean((denoised - clean) ** 2).item()
    psnr_before = 10 * np.log10(1.0 / mse_before) if mse_before > 0 else float("inf")
    psnr_after = 10 * np.log10(1.0 / mse_after) if mse_after > 0 else float("inf")

    # DnCNN should improve PSNR by at least 2 dB
    assert psnr_after > psnr_before + 2.0, (
        f"PSNR did not improve enough: before={psnr_before:.2f} dB, after={psnr_after:.2f} dB"
    )


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)

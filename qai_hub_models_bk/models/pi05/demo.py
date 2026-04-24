# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import make_pi05_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from torch.utils.data import DataLoader

from qai_hub_models.models.common import Precision
from qai_hub_models.models.pi05 import MODEL_ID, Model
from qai_hub_models.models.pi05.app import Pi05App, Pi05AppConfig
from qai_hub_models.utils.args import (
    get_model_cli_parser,
    get_model_kwargs,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.evaluate import EvalMode

DATASET_REPO_ID: str = "HuggingFaceVLA/libero"
HF_MODEL_ID: str = "lerobot/pi05_libero_finetuned"


def _to_device_tree(x: Any, device: str) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device_tree(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device_tree(v, device) for v in x)
    return x


def _build_preprocessed_batch(
    cfg: PI05Config,
    raw_batch: dict[str, Any],
    batch_size: int,
    dataset_stats: dict[str, Any] | None,
) -> tuple[dict[str, Any], PolicyProcessorPipeline]:
    """
    Use the official PI05 preprocessor to create a model-ready batch.

    Returns a tuple of:
      - Preprocessed dict compatible with PI05Policy APIs,
      - postprocessor to apply to predicted actions for parity.
    """
    preprocessor, postprocessor = make_pi05_pre_post_processors(
        config=cfg,
        dataset_stats=dataset_stats,
    )
    batch = preprocessor(raw_batch)
    return batch, postprocessor


def build_action_seq_from_first_episode(
    dataset: LeRobotDataset,
    device: torch.device | str,
    seq_len: int = 50,
) -> torch.Tensor:
    """
    Try to build a [1, seq_len, dof] action sequence from the first episode
    in the dataset.

    Parameters
    ----------
    dataset
        LeRobotDataset instance to extract actions from.
    device
        Target device for the returned tensor.
    seq_len
        Number of action steps to include in the sequence.

    Returns
    -------
    action_seq : torch.Tensor
        Tensor of shape [1, seq_len, dof] on the requested device.
    """
    loader_50 = DataLoader(
        dataset,
        batch_size=seq_len,
        shuffle=False,
        num_workers=0,
    )
    raw_many = next(iter(loader_50))

    actions = raw_many[ACTION]

    epi_key = "episode_index"
    frm_key = "frame_index"
    epi_ids = raw_many[epi_key]
    frm_ids = raw_many[frm_key]
    mask = epi_ids == epi_ids[0]
    idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    _, order = torch.sort(frm_ids[idxs])
    idxs = idxs[order]

    idxs = idxs[:seq_len]
    if idxs.numel() < seq_len:
        pad_last = idxs[-1].repeat(seq_len - idxs.numel())
        idxs = torch.cat([idxs, pad_last], dim=0)

    seq = actions[idxs]
    seq = seq.to(device=device, dtype=torch.float32)
    return seq.unsqueeze(0)


def load_one_batch(
    dataset_repo_id: str,
    pi05_config: PI05Config,
    batch_size: int,
    device: str = "cpu",
) -> tuple[dict[str, Any], PolicyProcessorPipeline, LeRobotDataset]:
    """
    Load one batch from a LeRobot dataset repo and preprocess it for
    PI05.

    Parameters
    ----------
    dataset_repo_id
        HuggingFace dataset repository identifier.
    pi05_config
        PI05 configuration for preprocessing.
    batch_size
        Number of samples in the batch.
    device
        Device to place tensors on.

    Returns
    -------
    batch : dict[str, Any]
        Model-ready batch dict on the requested device.
    postprocessor : PolicyProcessorPipeline
        Postprocessor pipeline to denormalize actions.
    dataset : LeRobotDataset
        The dataset instance for later GT construction.
    """
    dataset = LeRobotDataset(dataset_repo_id)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    raw_batch = next(iter(loader))
    raw_batch = _to_device_tree(raw_batch, device)

    batch, postprocessor = _build_preprocessed_batch(
        cfg=pi05_config,
        raw_batch=raw_batch,
        batch_size=batch_size,
        dataset_stats=dataset.meta.stats,
    )
    batch = _to_device_tree(batch, device)

    return batch, postprocessor, dataset


def run_pi05_on_sample(
    model: Model,
    policy: PI05Policy,
    batch: dict[str, Any],
    post_processor: PolicyProcessorPipeline,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Run Pi05App.predict_action_chunk on a processed batch.

    Parameters
    ----------
    model
        QAI Hub Model collection wrapper for flow backbone.
    policy
        PI05Policy instance providing tokenizer/config linkage.
    batch
        Preprocessed batch ready for the policy/app.
    post_processor
        PolicyProcessorPipeline to denormalize actions.
    noise
        Optional fixed noise tensor for deterministic sampling.

    Returns
    -------
    pred_actions : torch.Tensor
        [B, Tcfg, Dcfg] denormalized predicted actions.
    """
    app_cfg = Pi05AppConfig.from_policy(policy)
    app = Pi05App(config=app_cfg, **model.components).eval()

    pred_actions = app.predict_action_chunk(batch=batch, noise=noise)
    return post_processor(pred_actions)


def compute_sequence_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute MSE, RMSE and SQNR (in dB) over the entire sequence.

    Both pred and gt are expected to be [B, T, D]. The metrics are
    computed elementwise over all B * T * D elements.
    """
    if pred.dim() != 3 or gt.dim() != 3:
        raise ValueError("pred and gt must be [B, T, D]")

    t_max = min(pred.shape[1], gt.shape[1])
    d_max = min(pred.shape[2], gt.shape[2])

    pred_use = pred[:, :t_max, :d_max]
    gt_use = gt[:, :t_max, :d_max]

    err = pred_use - gt_use
    mse = torch.mean(err * err)
    rmse = torch.sqrt(mse + 1e-12)

    sig_pow = torch.mean(gt_use * gt_use) + 1e-12
    noise_pow = mse + 1e-12
    sqnr = 10.0 * torch.log10(sig_pow / noise_pow)

    return mse, rmse, sqnr


def plot_action_chunk(
    pred_actions: torch.Tensor,
    gt_actions: torch.Tensor,
    out_png: str,
    n_steps: int = 50,
) -> None:
    """
    Plot predicted vs ground-truth actions for the first item in batch.
    Produces D-panel PNG (one row per DoF) showing trajectories over
    timesteps. Only the first B item is shown.
    """
    b0 = 0
    t_max = min(n_steps, pred_actions.shape[1], gt_actions.shape[1])
    d_max = min(pred_actions.shape[2], gt_actions.shape[2])

    x_t = list(range(t_max))
    pred_ = pred_actions[b0, :t_max, :d_max].detach().cpu()
    gt_ = gt_actions[b0, :t_max, :d_max].detach().cpu()

    fig, axes = plt.subplots(nrows=d_max, ncols=1, figsize=(10, 2 * d_max))
    if d_max == 1:
        axes = [axes]

    for i in range(d_max):
        ax = axes[i]
        ax.plot(x_t, gt_[:, i], label="gt")
        ax.plot(x_t, pred_[:, i], label="pred")
        ax.set_title(f"DoF {i + 1}")
        ax.set_xlabel("timestep")
        ax.set_ylabel("action")
        if i == 0:
            ax.legend(loc="best")
        ax.grid(True, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(is_test: bool = False) -> None:
    """
    Run a simple FP demo for PI05 on one LIBERO batch. This prints the
    predicted and ground-truth tensor shapes, saves a comparison plot,
    and reports sequence-wide metrics.
    """
    parser = get_model_cli_parser(Model)
    parser = get_on_device_demo_parser(
        parser=parser,
        supported_eval_modes=[EvalMode.FP, EvalMode.ON_DEVICE],
        supported_precisions={Precision.float},
        available_target_runtimes=[TargetRuntime.QNN_CONTEXT_BINARY],
        default_device="Samsung Galaxy S25 (Family)",
        add_output_dir=True,
    )
    args = parser.parse_args([] if is_test else None)

    validate_on_device_demo_args(args, MODEL_ID)
    assert args.eval_mode == EvalMode.FP

    model_kwargs = get_model_kwargs(Model, vars(args))
    model = Model.from_pretrained(**model_kwargs)

    # Always use CPU and load policy from the HF model id.
    device = "cpu"
    torch.manual_seed(1234)
    policy = PI05Policy.from_pretrained(HF_MODEL_ID).to(device).eval()

    batch, postprocessor, dataset = load_one_batch(
        dataset_repo_id=DATASET_REPO_ID,
        pi05_config=policy.config,
        batch_size=1,
        device=device,
    )

    pred_actions = run_pi05_on_sample(
        model=model,
        policy=policy,
        batch=batch,
        post_processor=postprocessor,
        noise=None,
    )

    # Build a [1, 50, dof] ground-truth sequence for eval after prediction.
    # This avoids any lookahead data leakage.
    gt_actions = build_action_seq_from_first_episode(
        dataset=dataset,
        device=torch.device(device),
        seq_len=50,
    )

    print(
        f"pred_actions.shape={tuple(pred_actions.shape)}, "
        f"gt_actions.shape={tuple(gt_actions.shape)}"
    )

    # Compute sequence-wide metrics over all B * T * D.
    mse, rmse, sqnr = compute_sequence_metrics(pred=pred_actions, gt=gt_actions)
    print(
        f"sequence: mse={float(mse):.6f}, "
        f"rmse={float(rmse):.6f}, sqnr={float(sqnr):.3f} dB"
    )

    # Plot first-sample trajectories across DoF over T=50.
    output_dir = args.output_dir or f"build/demo_{args.eval_mode}"
    out_png = Path(output_dir) / "pi05_actions_vs_gt.png"
    plot_action_chunk(pred_actions, gt_actions, str(out_png))
    print(f"Saved action comparison plot to: {out_png}")


if __name__ == "__main__":
    main()

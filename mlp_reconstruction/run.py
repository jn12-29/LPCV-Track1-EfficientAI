from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlp_reconstruction.runner import cmd_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP Reconstruction for MobileCLIP2")
    p.add_argument("--model-name", default="MobileCLIP2-S0")
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument(
        "--calib-jsonl", default="build_datasets/data/vg_llm_contrastive.jsonl"
    )
    p.add_argument("--project-root", default=".")
    p.add_argument("--n-calib", type=int, default=1024)
    p.add_argument("--calib-batch-size", type=int, default=256)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument(
        "--output",
        default=None,
        help="Full output path. If omitted, auto-generated as "
        "{output-dir}/{model}__{config}__{timestamp}/mlp_relu.pt",
    )
    p.add_argument(
        "--gpu-id", type=int, default=None, metavar="N", help="GPU index (e.g. 2)"
    )
    p.add_argument(
        "--no-relu-image",
        action="store_true",
        help="Skip visual-encoder reconstruction (default: reconstruct both)",
    )
    p.add_argument(
        "--no-relu-text",
        action="store_true",
        help="Skip text-encoder reconstruction (default: reconstruct both)",
    )
    p.add_argument(
        "--no-relu-stem",
        action="store_true",
        help="Keep GELU in ConvStem (MobileCLIP2-B visual[stem]); "
        "still reconstructs all other visual blocks",
    )
    p.add_argument(
        "--keep-gelu-blocks",
        nargs="+",
        default=[],
        metavar="LABEL",
        help="Block labels to keep as GELU (skip ReLU distillation). "
        "E.g. --keep-gelu-blocks text[0] visual[stem]",
    )
    p.add_argument(
        "--keep-gelu-distill",
        action="store_true",
        help="For --keep-gelu-blocks: unconditionally run GELU drift distillation "
        "(repair upstream-induced drift without changing the activation). "
        "Without this flag, drift distillation only triggers when "
        "gelu_cos_sim < --gelu-threshold (if set).",
    )
    p.add_argument(
        "--gelu-threshold",
        type=float,
        default=None,
        metavar="T",
        help="Greedy mode only: revert to GELU if relu_cos_sim < T. "
        "In non-greedy mode the 3-way weighted comparison handles selection.",
    )
    p.add_argument(
        "--relu-bonus",
        type=float,
        default=0.0,
        metavar="B",
        help="Bonus added to relu_cos_sim when comparing against gelu_cos_sim and "
        "gelu_recon_cos_sim (reflects hardware speedup of ReLU over GELU). "
        "E.g. --relu-bonus 0.01 makes ReLU preferred when within 0.01 of GELU quality.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="Linear warmup steps before cosine decay (0 = disabled).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-iters", type=int, default=20000)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="L_Clamp loss weight (0=disable). Set 2.0 to match APHQ-ViT original "
        "(only useful when targeting quantization).",
    )
    p.add_argument("--aph-mode", default="uniform", choices=["uniform", "magnitude"])
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Collect (X, O) just-in-time from the partially-replaced model instead of "
        "pre-collecting from the original all-GELU model.",
    )
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping (default: disabled; runs all n_iters)",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        metavar="N",
        help="(requires --early-stop) Stop after N log-intervals without improvement",
    )
    p.add_argument(
        "--early-stop-delta",
        type=float,
        default=1e-4,
        help="(requires --early-stop) Min relative EMA-loss improvement to reset patience counter",
    )
    p.add_argument("--skip-to", default=None)
    p.add_argument("--resume-from", default=None)
    p.add_argument(
        "--skip-eval", action="store_true", help="Skip auto eval after training"
    )
    p.add_argument("--eval-root-dir", default="sample_data")
    p.add_argument("--eval-image-to-text-csv", default="sample_data/img_list.csv")
    p.add_argument("--eval-textnums-to-texts-csv", default="sample_data/txt_list.csv")
    p.add_argument("--eval-k", type=int, default=10)
    p.add_argument("--eval-batch-size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    cmd_train(parse_args())


if __name__ == "__main__":
    main()

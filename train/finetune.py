from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileCLIP2 on LPCV contrastive jsonl data"
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="./build_datasets/data/dataset_raw_contrastive.jsonl",
    )
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids. Single-GPU example: 2. Multi-GPU example with torchrun: 0,1,2,3",
    )
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument(
        "--init-lr", type=float, default=0.0, help="LR at warmup step 0 (default 0)"
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="LR floor after cosine decay (default 0)",
    )
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--accum-freq", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (default: AMP enabled on CUDA)",
    )
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=5,
        help="Save a numbered checkpoint every N epochs (0 = disabled).",
    )

    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-positives-per-image", type=int, default=None)
    parser.add_argument("--max-hard-negatives-per-image", type=int, default=None)
    parser.add_argument("--shuffle-texts-on-load", action="store_true")
    parser.add_argument(
        "--text-sampling",
        type=str,
        default="random",
        choices=["random", "first"],
    )

    parser.add_argument(
        "--num-hard-negatives",
        type=int,
        default=0,
        help="Hard negatives per image per iter. 0 = use all (dynamic per batch).",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="clip",
        choices=["clip", "siglip"],
        help="clip: InfoNCE + separate hard-negative loss; siglip: sigmoid pairwise loss "
        "(hard negatives absorbed into the main loss matrix, --hard-negative-weight and "
        "--hard-negative-margin are ignored).",
    )
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.2)
    parser.add_argument(
        "--hard-negative-loss-type",
        type=str,
        default="hinge",
        choices=["hinge", "logsigmoid"],
    )

    # --- Freeze ---
    parser.add_argument(
        "--freeze-modules",
        type=str,
        default=None,
        help=(
            "Comma-separated top-level module names to freeze (requires_grad=False). "
            "Example: --freeze-modules visual  or  --freeze-modules visual,transformer"
        ),
    )

    # --- Validation split ---
    parser.add_argument(
        "--val-split-size",
        type=int,
        default=1024,
        help="Hold out the N records as a validation set (0 = disabled).",
    )
    parser.add_argument(
        "--val-split-seed",
        type=int,
        default=1,
        help="If set, use a random split with this seed.",
    )

    # --- Val metrics ---
    parser.add_argument(
        "--val-every-n-epochs",
        type=int,
        default=1,
        help="Run validation every N epochs (0 = disabled).",
    )
    parser.add_argument(
        "--no-val-loss",
        action="store_true",
        help="Skip val loss computation (compute val recall only).",
    )
    parser.add_argument(
        "--no-val-recall",
        action="store_true",
        help="Skip val Recall@K computation (compute val loss only).",
    )
    parser.add_argument("--val-k", type=int, default=10, help="K for val Recall@K.")
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=32,
        help="Batch size for val recall encoding.",
    )

    # --- Sample-set eval ---
    parser.add_argument(
        "--sample-eval-root",
        type=str,
        default="./sample_data",
        help="Root dir for sample-set evaluation (contains images/, img_list.csv, txt_list.csv).",
    )
    parser.add_argument(
        "--sample-eval-image-csv",
        type=str,
        default="./sample_data/img_list.csv",
    )
    parser.add_argument(
        "--sample-eval-text-csv",
        type=str,
        default="./sample_data/txt_list.csv",
    )
    parser.add_argument(
        "--sample-eval-every-n-epochs",
        type=int,
        default=1,
        help="Run sample-set Recall@K eval every N epochs (0 = disabled).",
    )
    parser.add_argument(
        "--sample-eval-k", type=int, default=10, help="K for sample-set Recall@K."
    )
    parser.add_argument(
        "--sample-eval-batch-size",
        type=int,
        default=64,
        help="Batch size for sample-set recall encoding.",
    )

    # --- Resume ---
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) to resume from. Supports regular and QAT checkpoints (auto-detected).",
    )

    # --- ReLU MLP ---
    parser.add_argument(
        "--relu-image",
        action="store_true",
        help="Replace GELU with ReLU in all visual-encoder MLP blocks.",
    )
    parser.add_argument(
        "--relu-text",
        action="store_true",
        help="Replace GELU with ReLU in all text-encoder MLP blocks.",
    )

    # --- QAT ---
    parser.add_argument(
        "--qat-enabled",
        action="store_true",
        help="Enable Quantization-Aware Training via AIMET.",
    )
    parser.add_argument(
        "--qat-weight-bw",
        type=int,
        default=8,
        choices=[8],
        help="Weight bit-width for QAT.",
    )
    parser.add_argument(
        "--qat-act-bw",
        type=int,
        default=8,
        choices=[8, 16],
        help="Activation bit-width for QAT.",
    )
    parser.add_argument(
        "--qat-calib-samples",
        type=int,
        default=1024,
        help="Number of samples for QAT calibration.",
    )
    parser.add_argument(
        "--qat-quant-scheme",
        type=str,
        default="tf_enhanced",
        choices=["tf_enhanced", "percentile"],
        help="AIMET quantization scheme.",
    )

    parser.add_argument(
        "--qat-exclude-group-conv",
        action="store_true",
        help="Exclude all grouped convolutions (groups > 1) from QAT quantization.",
    )
    parser.add_argument(
        "--qat-exclude-names",
        type=str,
        default=None,
        help=(
            "Comma-separated regex patterns. Modules whose name matches any pattern "
            "are excluded from QAT quantization. Example: --qat-exclude-names 'stem\\.conv,stages\\.3'"
        ),
    )

    # --- Export ---
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export ONNX (image + text encoder) at each numbered checkpoint and at training end. "
        ".pt checkpoints are always saved.",
    )

    # --- Compilation ---
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="torch.compile(model, mode='reduce-overhead'). Incompatible with --qat-enabled.",
    )

    # --- Profiling ---
    parser.add_argument(
        "--enable-step-timing",
        action="store_true",
        default=False,
        help="Profile per-step timing (data transfer, forward, loss, backward, optimizer). "
        "Adds CUDA sync overhead; use for profiling only.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())

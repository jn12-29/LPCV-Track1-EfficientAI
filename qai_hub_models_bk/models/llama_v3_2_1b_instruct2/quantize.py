# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""
Quantization script for Llama 3.2 1B Instruct2.

GPU-accelerated calibration (seq-mse, ada-scale) requires a two-stage workflow
because ONNX export must use CPU torch for consistent op names, while AIMET-ONNX
calibration requires CUDA torch + onnxruntime-gpu.

Setup
-----
Create two virtual environments (Python 3.10 required):

    # CPU venv -- for ONNX export only
    python3.10 -m venv qaihm-cpu
    qaihm-cpu/bin/pip install -r requirements.txt

    # GPU venv -- for calibration (needs CUDA-capable GPU)
    python3.10 -m venv qaihm-gpu
    qaihm-gpu/bin/pip install -r requirements-gpu.txt

Both venvs need access to qai_hub_models. Either install the package or add
the repo's src/ directory to the venv's site-packages via a .pth file:

    echo "/path/to/repo/src" > qaihm-cpu/lib/python3.10/site-packages/qai_hub_models.pth
    echo "/path/to/repo/src" > qaihm-gpu/lib/python3.10/site-packages/qai_hub_models.pth

Two-stage workflow
------------------
Stage 1 -- Export ONNX (CPU venv):

    qaihm-cpu/bin/python quantize.py --export-only -o onnx_export/

Stage 2 -- Calibrate from pre-exported ONNX (GPU venv):

    qaihm-gpu/bin/python quantize.py --precision w4 -o checkpoint/w4/ \
        --onnx-dir onnx_export/ --use-seq-mse

    qaihm-gpu/bin/python quantize.py --precision w4a16 -o checkpoint/w4a16/ \
        --onnx-dir onnx_export/ --use-seq-mse

Single-stage (when one env has both CPU torch and CUDA ORT):

    python quantize.py -o checkpoint/w4a16/ --precision w4a16
    python quantize.py -o checkpoint/w4a16/ --num-samples 1
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import onnxruntime
import torch

from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CONTEXT_LENGTH,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.models.llama_v3_2_1b_instruct2.model import (
    HF_REPO_NAME,
    MODEL_ID,
    SUPPORTED_PRECISIONS,
    Llama3_2_1B_PreSplit,
    Llama3_2_1B_QuantizablePreSplit,
)
from qai_hub_models.utils.args import get_quantize_action_with_default
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader
from qai_hub_models.utils.printing import print_with_box


def export_onnx(
    context_length: int,
    seq_len: int,
    output_dir: str,
    checkpoint: str | None = None,
) -> None:
    """
    Export ONNX model with dynamic shapes (Stage 1).

    Run this with a CPU-torch venv to produce consistent ONNX op names.
    The output directory will contain model_dynamic.onnx, model.data,
    tokenizer files, and config.json.
    """
    host_device = torch.device("cpu")
    source_checkpoint = checkpoint or HF_REPO_NAME

    print("Stage 1: Export ONNX only")
    print(f"  Context length: {context_length}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Output directory: {output_dir}")
    print(f"  Loading from: {source_checkpoint}")
    print()

    print("Creating FP model...")
    fp_model = Llama3_2_1B_PreSplit.from_pretrained(
        checkpoint=source_checkpoint,
        sequence_length=seq_len,
        context_length=context_length,
        host_device=host_device,
    )

    print_with_box(
        [
            "Exporting ONNX model with dynamic shapes.",
            "This may take around 30 minutes.",
        ]
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Llama3_2_1B_QuantizablePreSplit.create_onnx_models(
        checkpoint=output_dir,
        fp_model=fp_model,
        context_length=context_length,
        host_device=host_device,
        llm_io_type=fp_model.llm_io_type,
        use_dynamic_shapes=True,
    )
    Llama3_2_1B_QuantizablePreSplit.save_tokenizer_and_config(
        checkpoint=output_dir, fp_model=fp_model
    )

    print(f"ONNX export saved to: {output_dir}")
    print("Contents:")
    for f in sorted(Path(output_dir).iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    Llama3_2_1B_PreSplit.clear_cache()


def quantize(
    context_length: int,
    seq_len: int,
    precision: Precision,
    output_dir: str,
    num_samples: int = 0,
    checkpoint: str | None = None,
    onnx_dir: str | None = None,
    use_seq_mse: bool = False,
    use_ada_scale: bool = False,
    seq_mse_num_samples: int | None = None,
    ada_scale_num_samples: int | None = None,
    ada_scale_num_iterations: int | None = None,
) -> None:
    """
    Quantize the Llama 3.2 1B model.

    Parameters
    ----------
    context_length
        Context length for the model.
    seq_len
        Sequence length for calibration.
    precision
        Quantization precision (w4a16 or w4).
    output_dir
        Output directory for the checkpoint.
    num_samples
        Number of calibration samples (0 = auto).
    checkpoint
        HuggingFace repo or local path for FP weights.
        Defaults to "meta-llama/Llama-3.2-1B-Instruct".
    onnx_dir
        Directory with pre-exported ONNX (model_dynamic.onnx + model.data).
        When provided, skips ONNX export and loads from this directory.
    use_seq_mse
        Whether to use sequential MSE for calibration.
    use_ada_scale
        Whether to use AdaScale for calibration.
    seq_mse_num_samples
        Number of samples for sequential MSE.
    ada_scale_num_samples
        Number of samples for AdaScale.
    ada_scale_num_iterations
        Number of iterations for AdaScale.
    """
    # Check GPU via onnxruntime, not torch -- torch is CPU-only for consistent ONNX op names.
    has_cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    if not has_cuda and (use_seq_mse or use_ada_scale):
        raise ValueError(
            "This quantization technique requires a CUDA GPU (V100/A100). Please re-try with GPU machine."
        )
    host_device = torch.device("cpu")

    print("Starting quantization for Llama 3.2 1B")
    print(f"  Context length: {context_length}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Precision: {precision}")
    print(f"  Output directory: {output_dir}")
    if onnx_dir:
        print(f"  ONNX source: {onnx_dir} (pre-exported)")
    print()

    source_checkpoint = checkpoint or HF_REPO_NAME
    print(f"Loading from: {source_checkpoint}")
    print(f"Using device: {host_device}")
    if has_cuda:
        print("  GPU available via onnxruntime CUDAExecutionProvider")
    print(f"  ORT available providers: {onnxruntime.get_available_providers()}")
    print()

    # Create FP model (needed for tokenizer/config even when loading pre-exported ONNX)
    print("Creating FP model...")
    fp_model = Llama3_2_1B_PreSplit.from_pretrained(
        checkpoint=source_checkpoint,
        sequence_length=seq_len,
        context_length=context_length,
        host_device=host_device,
    )

    if onnx_dir:
        # Load pre-exported ONNX from onnx_dir (skips ONNX export)
        onnx_path = Path(onnx_dir) / "model_dynamic.onnx"
        data_path = Path(onnx_dir) / "model.data"
        if not onnx_path.exists() or not data_path.exists():
            raise FileNotFoundError(
                f"Pre-exported ONNX not found in {onnx_dir}. "
                f"Expected model_dynamic.onnx and model.data. "
                f"Run with --export-only first."
            )
        print(f"Loading pre-exported ONNX from: {onnx_dir}")
        print("Creating QuantizablePreSplit...")
        model_quant = Llama3_2_1B_QuantizablePreSplit.from_pretrained(
            checkpoint=onnx_dir,
            fp_model=fp_model,
            sequence_length=seq_len,
            context_length=context_length,
            precision=precision,
            host_device=host_device,
        )
    else:
        print_with_box(
            [
                "Exporting ONNX model with dynamic shapes.",
                "This may take around 30 minutes.",
            ]
        )
        print("Creating QuantizablePreSplit...")
        model_quant = Llama3_2_1B_QuantizablePreSplit.from_pretrained(
            checkpoint=source_checkpoint,
            fp_model=fp_model,
            sequence_length=seq_len,
            context_length=context_length,
            precision=precision,
            host_device=host_device,
        )

    # Determine how many samples we need
    num_max_samples = num_samples or 0
    if use_seq_mse and seq_mse_num_samples is not None:
        num_max_samples = max(num_max_samples, seq_mse_num_samples)
    if use_ada_scale and ada_scale_num_samples is not None:
        num_max_samples = max(num_max_samples, ada_scale_num_samples)

    # Get calibration data (uses cache if available)
    print("Getting calibration data...")
    calib_data = model_quant.get_calibration_data(
        num_samples=num_max_samples,
    )

    if calib_data is not None:
        dataloader = dataset_entries_to_dataloader(calib_data)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if use_seq_mse or use_ada_scale:
            print()
            print("NOTE: This quantization technique can take hours to complete.")

        # Run quantization (calibration)
        print("Running calibration...")
        model_quant.quantize(
            data=dataloader,
            num_samples=num_samples,
            use_seq_mse=use_seq_mse,
            use_ada_scale=use_ada_scale,
            seq_mse_num_samples=seq_mse_num_samples,
            ada_scale_num_samples=ada_scale_num_samples,
            ada_scale_num_iterations=ada_scale_num_iterations,
        )
    else:
        print("No calibration data needed for this precision.")

    # Save checkpoint
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Saving checkpoint to: {output_dir}")
    model_quant.save_calibrated_checkpoint(output_dir)

    # Cleanup
    print("Cleaning up...")
    Llama3_2_1B_QuantizablePreSplit.clear_cache()
    Llama3_2_1B_PreSplit.clear_cache()

    print("Quantization completed successfully.")


def main() -> None:
    """Main entry point for the quantization script."""
    parser = argparse.ArgumentParser(
        description="Quantize Llama 3.2 1B Instruct2 model",
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context length for the model",
    )

    parser.add_argument(
        "--calibration-sequence-length",
        type=int,
        default=DEFAULT_CALIBRATION_SEQ_LEN,
        help="Sequence length to be used during calibration (does not need to match deployment sequence length).",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Output directory to export the ONNX model and encodings.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Input directory with custom weights.",
    )

    parser.add_argument(
        "--export-only",
        action="store_true",
        default=False,
        help="Export ONNX to --output-dir and exit (no calibration). "
        "Use with CPU-torch venv for consistent ONNX op names.",
    )

    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=None,
        help="Directory with pre-exported ONNX (model_dynamic.onnx + model.data). "
        "Skips ONNX export and loads from this directory for calibration.",
    )

    parser.add_argument(
        "--use-seq-mse",
        action="store_true",
        default=False,
        help="Add to apply Sequential MSE.",
    )

    parser.add_argument(
        "--use-ada-scale",
        action="store_true",
        default=False,
        help="Add to apply AdaScale.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to be used for calibration.",
    )

    parser.add_argument(
        "--seq-mse-num-samples",
        type=int,
        default=None,
        help="Number of samples for sequential MSE. Defaults to --num-samples.",
    )

    parser.add_argument(
        "--ada-scale-num-samples",
        type=int,
        default=None,
        help="Number of samples for AdaScale.",
    )

    parser.add_argument(
        "--ada-scale-num-iterations",
        type=int,
        default=None,
        help="Number of iterations for AdaScale.",
    )

    parser.add_argument(
        "--precision",
        default=Precision.parse(SUPPORTED_PRECISIONS[0]),
        action=get_quantize_action_with_default(SUPPORTED_PRECISIONS[0]),
        choices=[str(p) for p in SUPPORTED_PRECISIONS],
        help="Pick the precision with which the model must be quantized.",
    )

    args = parser.parse_args()

    if args.export_only:
        export_onnx(
            context_length=args.context_length,
            seq_len=args.calibration_sequence_length,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
        )
        return

    quantize(
        context_length=args.context_length,
        seq_len=args.calibration_sequence_length,
        precision=args.precision,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        checkpoint=args.checkpoint,
        onnx_dir=args.onnx_dir,
        use_seq_mse=args.use_seq_mse,
        use_ada_scale=args.use_ada_scale,
        seq_mse_num_samples=args.seq_mse_num_samples,
        ada_scale_num_samples=args.ada_scale_num_samples,
        ada_scale_num_iterations=args.ada_scale_num_iterations,
    )

    print()
    print(
        "    If you are using custom weights via checkpoint folder, please add a copy of the model config to the output checkpoint folder. This will help run the demo and evaluation correctly for your model."
    )
    print()
    print("Evaluate:")
    print(
        f"    python -m qai_hub_models.models.{MODEL_ID}.evaluate --checkpoint {args.output_dir} --task wikitext"
    )
    print()
    print("Demo:")
    print(
        f"    python -m qai_hub_models.models.{MODEL_ID}.demo --checkpoint {args.output_dir} --prompt 'What is gravity?'"
    )
    print()
    print("Export:")
    print(
        f"    python -m qai_hub_models.models.{MODEL_ID}.export --checkpoint {args.output_dir} --device 'Snapdragon 8 Elite QRD' --skip-profiling --skip-inferencing --output-dir output"
    )


if __name__ == "__main__":
    main()

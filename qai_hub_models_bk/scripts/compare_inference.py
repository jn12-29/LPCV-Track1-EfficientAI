# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Standalone script to compare local CPU inference outputs with on-device
inference outputs from AI Hub.

This script provides detailed numerical comparison metrics for debugging
model accuracy issues after export/compilation.

Usage:
    python -m qai_hub_models.scripts.compare_inference --model resnet50

"""

from __future__ import annotations

import importlib
import sys
import warnings
from typing import Any

import qai_hub as hub

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.args import (
    _add_device_args,
    add_precision_arg,
    add_target_runtime_arg,
    get_input_spec_kwargs,
    get_model_kwargs,
    get_parser,
)
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.compare import METRICS_FUNCTIONS, torch_inference
from qai_hub_models.utils.path_helpers import MODEL_IDS
from qai_hub_models.utils.printing import print_inference_metrics
from qai_hub_models.utils.transpose_channel import transpose_channel_last_to_first

ALL_METRICS = ",".join(k for k in METRICS_FUNCTIONS if k not in ("top1", "top5"))


def load_model_module(model_id: str) -> Any:
    """Dynamically load a model module by model ID."""
    if model_id not in MODEL_IDS:
        raise ValueError(
            f"Unknown model ID: {model_id}. "
            f"Available models: {', '.join(MODEL_IDS[:10])}..."
        )
    return importlib.import_module(f"qai_hub_models.models.{model_id}")


def compare_inference(
    model_id: str,
    device: hub.Device,
    target_runtime: TargetRuntime = TargetRuntime.TFLITE,
    precision: Precision = Precision.float,
    metrics: str = ALL_METRICS,
    outputs_to_skip: list[int] | None = None,
    component: str | None = None,
    channel_last_outputs: list[str] | None = None,
    **additional_model_kwargs: Any,
) -> None:
    """
    Compare local CPU inference with on-device inference from AI Hub.

    Parameters
    ----------
    model_id
        Model ID (e.g., resnet50).
    device
        Device for which to export the model.
    target_runtime
        Which on-device runtime to target. Default is TFLite.
    precision
        The precision to which this model should be quantized.
    metrics
        Comma-separated list of metrics to compute.
    outputs_to_skip
        List of output indices to skip.
    component
        For collection models, specify which component to compare.
    channel_last_outputs
        List of output names that are in channel-last format on device.
    **additional_model_kwargs
        Additional kwargs for model.from_pretrained and model.get_input_spec.
    """
    model_module = load_model_module(model_id)
    export_module = importlib.import_module(f"qai_hub_models.models.{model_id}.export")
    model_cls = model_module.Model

    # Validate component argument
    if issubclass(model_cls, CollectionModel):
        if component and component not in model_cls.component_class_names:
            print(
                f"Unknown component: {component}. "
                f"Available: {model_cls.component_class_names}"
            )
            sys.exit(1)
        if not component:
            print(
                f"Model {model_id} is a collection model with components: "
                f"{model_cls.component_class_names}. Use --component to specify one."
            )
            sys.exit(1)
    elif component:
        print(
            f"Model {model_id} is not a collection model, "
            "--component is not applicable."
        )
        sys.exit(1)

    inference_job: hub.InferenceJob
    model: BaseModel

    # Run export with inference
    if issubclass(model_cls, CollectionModel):
        export_result = export_module.export_model(
            device=device,
            target_runtime=target_runtime,
            precision=precision,
            skip_profiling=True,
            skip_inferencing=False,
            skip_downloading=True,
            skip_summary=True,
            components=[component],
            **additional_model_kwargs,
        )
        inference_job_opt = export_result.components[component].inference_job
        model_kwargs = get_model_kwargs(
            model_cls, dict(**additional_model_kwargs, precision=precision)
        )
        model_instance = model_cls.from_pretrained(**model_kwargs)
        model = model_instance.components[component]
        if not isinstance(model, BaseModel):
            raise TypeError(f"Component {component} is not a BaseModel")
    else:
        export_result = export_module.export_model(
            device=device,
            target_runtime=target_runtime,
            precision=precision,
            skip_profiling=True,
            skip_inferencing=False,
            skip_downloading=True,
            skip_summary=True,
            **additional_model_kwargs,
        )
        inference_job_opt = export_result.inference_job
        model_kwargs = get_model_kwargs(
            model_cls, dict(**additional_model_kwargs, precision=precision)
        )
        model = model_cls.from_pretrained(**model_kwargs)

    if inference_job_opt is None:
        print("Error: Export did not produce an inference job.")
        sys.exit(1)
    inference_job = inference_job_opt

    # Download on-device results
    device_outputs = inference_job.download_output_data()
    if device_outputs is None:
        print("Error: Failed to download inference results.")
        sys.exit(1)

    output_names = model.get_output_names()

    # Transpose channel-last outputs if needed
    if channel_last_outputs:
        device_outputs = transpose_channel_last_to_first(
            channel_last_outputs, device_outputs
        )

    # Run local inference
    input_spec = model.get_input_spec(
        **get_input_spec_kwargs(model, additional_model_kwargs)
    )
    sample_inputs = model.sample_inputs(input_spec, use_channel_last_format=False)

    torch_out = torch_inference(model, sample_inputs, return_channel_last_output=False)

    print_inference_metrics(
        inference_job,
        device_outputs,
        torch_out,
        output_names,
        outputs_to_skip=outputs_to_skip,
        metrics=metrics,
    )


def main() -> None:
    warnings.filterwarnings("ignore")

    parser = get_parser()
    parser.description = (
        "Compare local CPU inference with on-device inference from AI Hub."
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help=f"Model ID (e.g., resnet50). Available: {', '.join(MODEL_IDS[:5])}...",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default=ALL_METRICS,
        help="Comma-separated list of metrics. Available: "
        + ALL_METRICS
        + f". Default: {ALL_METRICS}",
    )

    parser.add_argument(
        "--outputs-to-skip",
        type=str,
        default=None,
        help="Comma-separated list of output indices to skip (e.g., '0,2').",
    )

    parser.add_argument(
        "--component",
        "-c",
        type=str,
        default=None,
        help="For collection models, specify which component to compare.",
    )

    parser.add_argument(
        "--channel-last-outputs",
        type=str,
        default=None,
        help="Comma-separated list of output names in channel-last format on device.",
    )

    _add_device_args(parser, default_device="Samsung Galaxy S25 (Family)")
    add_target_runtime_arg(
        parser,
        helpmsg="Target runtime for compilation.",
        default=TargetRuntime.TFLITE,
    )
    add_precision_arg(
        parser,
        supported_precisions={Precision.float, Precision.w8a8, Precision.w8a16},
        default_if_arg_explicitly_passed=Precision.w8a8,
        default=Precision.float,
    )

    args = parser.parse_args()

    # Validate metrics
    for m in args.metrics.split(","):
        if m not in METRICS_FUNCTIONS:
            parser.error(
                f"Unknown metric: {m}. Available: {', '.join(METRICS_FUNCTIONS.keys())}"
            )

    # Parse list arguments
    outputs_to_skip: list[int] | None = None
    if args.outputs_to_skip:
        outputs_to_skip = [int(x.strip()) for x in args.outputs_to_skip.split(",")]

    channel_last_outputs: list[str] | None = None
    if args.channel_last_outputs:
        channel_last_outputs = [x.strip() for x in args.channel_last_outputs.split(",")]

    compare_inference(
        model_id=args.model,
        device=args.device,
        target_runtime=args.target_runtime,
        precision=args.precision,
        metrics=args.metrics,
        outputs_to_skip=outputs_to_skip,
        component=args.component,
        channel_last_outputs=channel_last_outputs,
    )


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Generator

import onnx
from aimet_onnx.common.defs import (
    QuantizationDataType,
)
from aimet_onnx.qc_quantize_op import (
    GroupedBlockQuantizeDequantize,
    QcQuantizeOp,
)
from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx


def _tie_quantizers_for_kv_cache(
    quantsim_model: QuantSimOnnx, kv_io_map: dict[str, str]
) -> None:
    quantizer_mapping = {}

    for input_name, output_name in kv_io_map.items():
        quantizer = quantsim_model._get_enabled_quantizer(output_name)  # pylint: disable=protected-access
        if quantizer:
            quantizer_mapping[input_name] = quantizer

    quantsim_model.set_quantizers(quantizer_mapping)


def _set_tensors_to_output_8b_sym(
    quantsim_model: QuantSimOnnx, out_tensors: list[str]
) -> None:
    for out_tensor in out_tensors:
        _set_tensor_to_8_bit_symmetric(quantsim_model, out_tensor)


def _get_kv_io_map(quantsim_model: QuantSimOnnx) -> dict[str, str]:
    inputs = [
        t.name
        for t in quantsim_model.model.graph().input
        if "past_key" in t.name or "past_value" in t.name
    ]
    outputs = [
        t.name.replace("_updated", "")
        for t in quantsim_model.model.graph().output
        if "past_key" in t.name or "past_value" in t.name
    ]
    return dict(zip(inputs, outputs, strict=False))


def _set_4bit_weights_to_lpbq(quantsim_model: QuantSimOnnx) -> None:
    # This is largely a copy-paste of
    # set_grouped_blockwise_quantization_for_weights, except adds an op
    # selection criterion based on all ops that already have the target
    # bitwidth. Can be simplified once that function accepts a function
    # argument.
    block_size = 64
    decompressed_bw = 8
    strict = False
    bitwidth = 4
    for op in quantsim_model.connected_graph.ordered_ops:
        _, _, param_quantizers = quantsim_model.get_op_quantizers(op)

        weight_quantizer: QcQuantizeOp = param_quantizers.get("weight")
        bias_quantizer: QcQuantizeOp = param_quantizers.get("bias")

        if not weight_quantizer:
            continue

        if weight_quantizer.bitwidth != bitwidth:
            continue

        try:
            grouped_quantizer = GroupedBlockQuantizeDequantize(
                weight_quantizer.quant_info,
                bitwidth,
                decompressed_bw,
                block_size,
                weight_quantizer.quant_scheme,
                weight_quantizer.op_mode,
                weight_quantizer.tensor_quantizer_params,
            )
        except ValueError:
            if strict:
                raise
        else:
            if bias_quantizer:
                bias_quantizer.enable_per_channel_quantization()
                bias_quantizer.use_symmetric_encodings = True
                bias_quantizer.data_type = QuantizationDataType.int

            for name, quantizer in quantsim_model.qc_quantize_op_dict.items():
                if quantizer is weight_quantizer:
                    quantsim_model.qc_quantize_op_dict[name] = grouped_quantizer


def _set_tensor_to_8_bit_symmetric(
    quantsim_model: QuantSimOnnx, tensor_name: str
) -> None:
    quantizer = quantsim_model._get_enabled_quantizer(tensor_name)  # pylint: disable=protected-access
    if quantizer:
        quantizer.set_bitwidth(8)
        quantizer.use_symmetric_encodings = True


def _set_lm_head_to_8b(quantsim_model: QuantSimOnnx) -> None:
    for weight in _get_lm_head_weights(quantsim_model.model.model):
        quantizer = quantsim_model.qc_quantize_op_dict[weight.name]
        quantizer.set_bitwidth(8)
        quantizer.quant_info.blockSize = 0
        quantizer.quant_info.blockAxis = -1
        quantizer.enable_per_channel_quantization()


def _get_lm_head_weights(
    model: onnx.ModelProto,
) -> Generator[onnx.TensorProto, None, None]:
    vocab_size = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    for weight in model.graph.initializer:
        if any(dim == vocab_size for dim in weight.dims):
            for node in model.graph.node:
                if node.op_type in ("Gemm", "MatMul", "Conv") and node.input[1] in {
                    weight.name,
                    weight.name + "_updated",
                    weight.name + "_qdq",
                }:
                    yield weight


def _get_quantizer_no_split_slice(
    quantsim_model: QuantSimOnnx, tensor_name: str
) -> QcQuantizeOp | None:
    """
    Returns closest enabled quantizer to tensor traversing upwards only
    through invariant ops and no Split/Slice.
    """
    from aimet_onnx.common.onnx._utils import _is_grid_preserving_op

    quantizer = quantsim_model.qc_quantize_op_dict.get(tensor_name, None)
    if quantizer and quantizer.enabled:
        return quantizer

    prod_dict = quantsim_model.connected_graph.get_all_products()
    product = prod_dict.get(tensor_name, None)

    if product is None:
        if tensor_name.endswith(("_updated", "_qdq")):
            raise KeyError(
                f"Could not find quantizer for tensor {tensor_name}. "
                "Input tensor_name must be the name of a tensor in the "
                "original (unquantized) graph"
            )
        raise KeyError(
            f"Could not find quantizer for tensor {tensor_name}. "
            "Tensor name does not exist in the graph"
        )

    producer = product.producer

    if producer is None:
        return None

    if not (_is_grid_preserving_op(producer.type)) or producer.type in {
        "Slice",
        "Split",
        "SplitToSequence",
    }:
        return None

    if len(producer.inputs) == 0:
        return None

    upstream_tensor = producer.inputs[0]
    return _get_quantizer_no_split_slice(quantsim_model, upstream_tensor.name)


_PROPAGATE_8B_OPS = frozenset({"Concat", "Transpose", "Reshape", "Slice", "Div"})


def _propagate_8b_upstream(
    quantsim_model: QuantSimOnnx, tensor_name: str, visited: set[str] | None = None
) -> None:
    """
    Recursively propagate 8-bit quantization upstream through Concat,
    Transpose, Reshape, Slice, and Div ops. For each such op, set its data
    inputs' quantizers to 8-bit and continue recursing.

    Div is also included to accommodate when we swap the order of MatMul and
    Div. We do this for Qwen3. When we do this, the Div is upstream from the
    MatMul and need to be included if we want to get 16x8 MatMuls.

    Concat: all inputs are data inputs.
    Transpose, Reshape, Slice: only input[0] is data.
    """
    if visited is None:
        visited = set()
    if tensor_name in visited:
        return
    visited.add(tensor_name)

    prod_dict = quantsim_model.connected_graph.get_all_products()
    product = prod_dict.get(tensor_name, None)
    if product is None:
        return

    producer = product.producer
    if producer is None:
        return

    if producer.type not in _PROPAGATE_8B_OPS:
        return

    if producer.type == "Concat":
        data_inputs = list(producer.inputs)
    else:
        # Transpose, Reshape, Slice: only first input is data
        data_inputs = [producer.inputs[0]] if producer.inputs else []

    for inp in data_inputs:
        quantizer = quantsim_model.qc_quantize_op_dict.get(inp.name, None)
        if quantizer is not None and quantizer.enabled and quantizer.bitwidth > 8:
            quantizer.set_bitwidth(8)
            quantizer.use_symmetric_encodings = True
        _propagate_8b_upstream(quantsim_model, inp.name, visited)


def _set_matmul_second_input_to_8b(quantsim_model: QuantSimOnnx) -> None:
    cg = quantsim_model.connected_graph

    for op in reversed(cg.ordered_ops):
        if op.type != "MatMul":
            continue

        upper_quantizer = quantsim_model._get_enabled_quantizer(op.inputs[1].name)  # pylint: disable=protected-access

        enabled_quantizer = _get_quantizer_no_split_slice(
            quantsim_model, op.inputs[1].name
        )

        if enabled_quantizer and enabled_quantizer.bitwidth <= 8:
            pass
        elif enabled_quantizer:
            enabled_quantizer.set_bitwidth(8)
            enabled_quantizer.use_symmetric_encodings = True
        elif upper_quantizer:
            if op.inputs[1].name in quantsim_model.qc_quantize_op_dict:
                quantizer = quantsim_model.qc_quantize_op_dict[op.inputs[1].name]
                quantizer.enabled = True
                quantizer.set_bitwidth(8)
                quantizer.use_symmetric_encodings = True
            else:
                quantsim_model._insert_quantizer(op.inputs[1].name, is_param=False)  # pylint: disable=protected-access
                quantsim_model._rebuild_session()  # pylint: disable=protected-access
                quantizer = quantsim_model.qc_quantize_op_dict[op.inputs[1].name]
                quantizer.enabled = True
                quantizer.set_bitwidth(8)
                quantizer.use_symmetric_encodings = True
        else:
            continue

        # Propagate 8-bit upstream through select op types to ensure MatMul
        # 8-bit input is respected during conversion.
        _propagate_8b_upstream(quantsim_model, op.inputs[1].name)


def _apply_int8_kv_cache_tying_and_lm_head(
    sim: QuantSimOnnx, kv_io_map: dict[str, str]
) -> QuantSimOnnx:
    sim._tie_quantizers_for_op_types(["Concat"])  # pylint: disable=protected-access
    sim._rebuild_session()  # pylint: disable=protected-access

    # Setting kv_cache and some other layers to 8-bit
    kv_io_list = list(kv_io_map.keys()) + list(kv_io_map.values())
    _set_tensors_to_output_8b_sym(sim, kv_io_list)

    # Setting the LM head weights to 8-bit.
    _set_lm_head_to_8b(sim)

    # Tie kv_cache
    _tie_quantizers_for_kv_cache(sim, kv_io_map)

    # Setting Matmul second input to 8b
    _set_matmul_second_input_to_8b(sim)

    return sim

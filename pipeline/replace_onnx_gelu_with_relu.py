from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, helper, numpy_helper


def _const_map(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for init in graph.initializer:
        values[init.name] = numpy_helper.to_array(init)

    for node in graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.type == AttributeProto.TENSOR:
                values[node.output[0]] = numpy_helper.to_array(attr.t)
            elif attr.name == "value_float":
                values[node.output[0]] = np.asarray(attr.f, dtype=np.float32)
            elif attr.name == "value_int":
                values[node.output[0]] = np.asarray(attr.i, dtype=np.int64)
    return values


def _is_scalar_close(values: dict[str, np.ndarray], name: str, target: float) -> bool:
    arr = values.get(name)
    if arr is None:
        return False
    arr = np.asarray(arr)
    if arr.size != 1:
        return False
    return bool(np.isclose(float(arr.reshape(-1)[0]), target, rtol=1e-4, atol=1e-5))


def _producer_map(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {out: node for node in graph.node for out in node.output}


def _consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            consumers[inp].append(node)
    return consumers


def _other_input(node: onnx.NodeProto, known: str) -> str | None:
    others = [inp for inp in node.input if inp != known]
    return others[0] if len(others) == 1 else None


def _has_input(node: onnx.NodeProto, name: str) -> bool:
    return any(inp == name for inp in node.input)


def _has_const(node: onnx.NodeProto, values: dict[str, np.ndarray], target: float) -> bool:
    return any(_is_scalar_close(values, inp, target) for inp in node.input)


def _gelu_source_from_erf_arg(
    erf_arg: str,
    producers: dict[str, onnx.NodeProto],
    values: dict[str, np.ndarray],
) -> tuple[str | None, onnx.NodeProto | None]:
    scale_node = producers.get(erf_arg)
    if scale_node is None or len(scale_node.input) != 2:
        return None, None

    a, b = scale_node.input
    sqrt2 = float(np.sqrt(2.0))
    inv_sqrt2 = float(1.0 / np.sqrt(2.0))

    if scale_node.op_type == "Div":
        if _is_scalar_close(values, b, sqrt2):
            return a, scale_node
        return None, None

    if scale_node.op_type == "Mul":
        if _is_scalar_close(values, a, inv_sqrt2):
            return b, scale_node
        if _is_scalar_close(values, b, inv_sqrt2):
            return a, scale_node

    return None, None


def _match_mul_chain(
    add_out: str,
    x_name: str,
    consumers: dict[str, list[onnx.NodeProto]],
    values: dict[str, np.ndarray],
) -> tuple[list[onnx.NodeProto], onnx.NodeProto, str] | None:
    # Common form: Mul(x, add_out) -> Mul(..., 0.5)
    for first_mul in consumers.get(add_out, []):
        if first_mul.op_type != "Mul" or len(first_mul.input) != 2 or len(first_mul.output) != 1:
            continue

        first_out = first_mul.output[0]
        other = _other_input(first_mul, add_out)

        if other == x_name:
            for second_mul in consumers.get(first_out, []):
                if second_mul.op_type == "Mul" and _has_const(second_mul, values, 0.5):
                    return [first_mul, second_mul], second_mul, second_mul.output[0]

        if other is not None and _is_scalar_close(values, other, 0.5):
            for second_mul in consumers.get(first_out, []):
                if second_mul.op_type == "Mul" and _has_input(second_mul, x_name):
                    return [first_mul, second_mul], second_mul, second_mul.output[0]

    # Alternate form: Mul(x, 0.5) -> Mul(..., add_out)
    for first_mul in consumers.get(x_name, []):
        if first_mul.op_type != "Mul" or len(first_mul.input) != 2 or len(first_mul.output) != 1:
            continue
        if not _has_const(first_mul, values, 0.5):
            continue
        first_out = first_mul.output[0]
        for second_mul in consumers.get(add_out, []):
            if second_mul.op_type == "Mul" and _has_input(second_mul, first_out):
                return [first_mul, second_mul], second_mul, second_mul.output[0]

    return None


def _safe_to_remove(
    nodes: list[onnx.NodeProto],
    final_output: str,
    consumers: dict[str, list[onnx.NodeProto]],
) -> bool:
    remove_ids = {id(node) for node in nodes}
    for node in nodes:
        for out in node.output:
            if out == final_output:
                continue
            for consumer in consumers.get(out, []):
                if id(consumer) not in remove_ids:
                    return False
    return True


def replace_exact_gelu_with_relu(model: onnx.ModelProto) -> int:
    graph = model.graph
    producers = _producer_map(graph)
    consumers = _consumer_map(graph)
    values = _const_map(graph)

    replacements: dict[int, onnx.NodeProto] = {}
    remove_ids: set[int] = set()
    removed_value_names: set[str] = set()

    for erf_node in list(graph.node):
        if erf_node.op_type != "Erf" or len(erf_node.input) != 1 or len(erf_node.output) != 1:
            continue
        if id(erf_node) in remove_ids:
            continue

        x_name, scale_node = _gelu_source_from_erf_arg(erf_node.input[0], producers, values)
        if x_name is None or scale_node is None:
            continue

        add_nodes = [
            node
            for node in consumers.get(erf_node.output[0], [])
            if node.op_type == "Add"
            and len(node.input) == 2
            and len(node.output) == 1
            and _has_const(node, values, 1.0)
        ]
        for add_node in add_nodes:
            match = _match_mul_chain(add_node.output[0], x_name, consumers, values)
            if match is None:
                continue

            mul_nodes, final_node, final_output = match
            nodes_to_remove = [scale_node, erf_node, add_node, *mul_nodes]
            if not _safe_to_remove(nodes_to_remove, final_output, consumers):
                continue

            relu_node = helper.make_node(
                "Relu",
                inputs=[x_name],
                outputs=[final_output],
                name=f"{final_node.name or final_output}_gelu_to_relu",
            )
            replacements[id(final_node)] = relu_node
            remove_ids.update(id(node) for node in nodes_to_remove)
            for node in nodes_to_remove:
                removed_value_names.update(node.output)
            removed_value_names.discard(final_output)
            break

    if not replacements:
        return 0

    new_nodes = []
    for node in graph.node:
        node_id = id(node)
        if node_id in replacements:
            new_nodes.append(replacements[node_id])
        elif node_id not in remove_ids:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    if removed_value_names:
        keep_value_info = [vi for vi in graph.value_info if vi.name not in removed_value_names]
        del graph.value_info[:]
        graph.value_info.extend(keep_value_info)

    return len(replacements)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace exact ONNX GELU subgraphs with Relu.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to in-place overwrite.")
    parser.add_argument("--require-match", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input

    model = onnx.load(str(args.input))
    before = Counter(node.op_type for node in model.graph.node)
    count = replace_exact_gelu_with_relu(model)
    after = Counter(node.op_type for node in model.graph.node)

    if args.require_match and count == 0:
        raise RuntimeError(f"No exact GELU subgraphs were found in {args.input}")

    onnx.checker.check_model(model)
    onnx.save(model, str(output))

    print(f"[gelu-to-relu] input:  {args.input}")
    print(f"[gelu-to-relu] output: {output}")
    print(f"[gelu-to-relu] replaced GELU subgraphs: {count}")
    print(f"[gelu-to-relu] ops before: {dict(before)}")
    print(f"[gelu-to-relu] ops after:  {dict(after)}")


if __name__ == "__main__":
    main()

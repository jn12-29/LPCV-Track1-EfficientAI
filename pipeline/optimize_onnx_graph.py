from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def _rewire_tensor_uses(old_tensor: gs.Tensor, new_tensor: gs.Tensor) -> int:
    """Rewire all consumers that use old_tensor to new_tensor."""
    rewired = 0
    for consumer in list(old_tensor.outputs):
        for idx, inp in enumerate(consumer.inputs):
            if inp is old_tensor:
                consumer.inputs[idx] = new_tensor
                rewired += 1
    return rewired


def _bypass_single_io_node(node: gs.Node) -> bool:
    """Bypass a node via tensor rewiring; cleanup() collects it later."""
    if len(node.inputs) < 1 or len(node.outputs) != 1:
        return False
    src = node.inputs[0]
    dst = node.outputs[0]
    _rewire_tensor_uses(dst, src)
    node.outputs.clear()
    return True


def _is_static_int_shape(shape: object) -> bool:
    if not isinstance(shape, (list, tuple)):
        return False
    return all(isinstance(x, int) for x in shape)


def _same_tensor_shape(a: gs.Tensor, b: gs.Tensor) -> bool:
    a_shape = getattr(a, "shape", None)
    b_shape = getattr(b, "shape", None)
    if not _is_static_int_shape(a_shape) or not _is_static_int_shape(b_shape):
        return False
    return list(a_shape) == list(b_shape)


def _same_tensor_dtype(a: gs.Tensor, b: gs.Tensor) -> bool:
    return getattr(a, "dtype", None) == getattr(b, "dtype", None)


def _inverse_perm(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for i, v in enumerate(perm):
        inv[v] = i
    return inv


def _is_identity_perm(perm: list[int]) -> bool:
    return perm == list(range(len(perm)))


def _get_perm(node: gs.Node) -> list[int] | None:
    perm = node.attrs.get("perm")
    return perm if isinstance(perm, list) else None


def _remove_generic_bypass_ops(graph: gs.Graph) -> int:
    """Topology-based bypass for shape-preserving no-op style operators."""
    removed = 0
    for node in list(graph.nodes):
        if node.op not in {"Identity", "Dropout", "Cast"}:
            continue
        if len(node.inputs) < 1 or len(node.outputs) != 1:
            continue
        inp = node.inputs[0]
        out = node.outputs[0]
        # For safety, only bypass when dtype/shape are preserved.
        if not _same_tensor_dtype(inp, out):
            continue
        if not _same_tensor_shape(inp, out):
            continue
        if _bypass_single_io_node(node):
            removed += 1
    return removed


def _remove_transpose_noops(graph: gs.Graph) -> int:
    """Remove single identity transpose and pairwise canceling transposes."""
    removed = 0
    for node in list(graph.nodes):
        if node.op != "Transpose" or len(node.outputs) != 1:
            continue

        perm_a = _get_perm(node)
        if perm_a is None:
            continue

        # Case 1: single identity transpose -> bypass.
        if _is_identity_perm(perm_a):
            if _bypass_single_io_node(node):
                removed += 1
            continue

        # Case 2: transpose followed by inverse transpose -> bypass both.
        mid_tensor = node.outputs[0]
        if len(mid_tensor.outputs) != 1:
            continue
        next_node = mid_tensor.outputs[0]
        if next_node.op != "Transpose" or len(next_node.inputs) != 1 or len(next_node.outputs) != 1:
            continue
        perm_b = _get_perm(next_node)
        if perm_b is None or _inverse_perm(perm_a) != perm_b:
            continue
        src = node.inputs[0]
        dst = next_node.outputs[0]
        _rewire_tensor_uses(dst, src)
        node.outputs.clear()
        next_node.outputs.clear()
        removed += 2
    return removed


def _remove_identity_reshape(graph: gs.Graph) -> int:
    """Bypass Reshape nodes that keep exactly the same static shape."""
    removed = 0
    for node in list(graph.nodes):
        if node.op != "Reshape" or len(node.inputs) < 1 or len(node.outputs) != 1:
            continue
        inp = node.inputs[0]
        out = node.outputs[0]
        if not _same_tensor_shape(inp, out):
            continue
        if _bypass_single_io_node(node):
            removed += 1
    return removed


def _run_rewrite_passes(graph: gs.Graph) -> dict[str, int]:
    """Run multiple rounds because cleanup can expose new simplification opportunities."""
    stats = {
        "transpose_removed": 0,
        "reshape_removed": 0,
        "bypass_removed": 0,
        "add_chain_folded": 0,
        "mul_chain_folded": 0,
    }
    for _ in range(3):
        graph.toposort()
        t = _remove_transpose_noops(graph)
        r = _remove_identity_reshape(graph)
        b = _remove_generic_bypass_ops(graph)
        c = _collapse_const_binary_chains(graph)
        if (
            t == 0
            and r == 0
            and b == 0
            and c["add_chain_folded"] == 0
            and c["mul_chain_folded"] == 0
        ):
            break
        stats["transpose_removed"] += t
        stats["reshape_removed"] += r
        stats["bypass_removed"] += b
        stats["add_chain_folded"] += c["add_chain_folded"]
        stats["mul_chain_folded"] += c["mul_chain_folded"]
        graph.cleanup().toposort()
    return stats


def _tensor_rank(tensor: gs.Tensor) -> int | None:
    shape = getattr(tensor, "shape", None)
    if isinstance(shape, (list, tuple)):
        return len(shape)
    return None


def _as_constant(tensor: gs.Tensor) -> gs.Constant | None:
    return tensor if isinstance(tensor, gs.Constant) else None


def _get_single_const_binary_inputs(
    node: gs.Node,
) -> tuple[gs.Tensor, gs.Constant] | None:
    if len(node.inputs) != 2:
        return None
    a, b = node.inputs
    ca = _as_constant(a)
    cb = _as_constant(b)
    if ca is not None and cb is None:
        return b, ca
    if cb is not None and ca is None:
        return a, cb
    return None


def _collapse_const_binary_chains(graph: gs.Graph) -> dict[str, int]:
    """Fold (x op c1) op c2 into x op c for Add/Mul."""
    folded_add = 0
    folded_mul = 0
    for node in list(graph.nodes):
        if node.op not in {"Add", "Mul"} or len(node.outputs) != 1:
            continue
        parsed = _get_single_const_binary_inputs(node)
        if parsed is None:
            continue
        base_tensor, const_1 = parsed

        mid = node.outputs[0]
        if len(mid.outputs) != 1:
            continue
        next_node = mid.outputs[0]
        if next_node.op != node.op or len(next_node.outputs) != 1:
            continue
        parsed_next = _get_single_const_binary_inputs(next_node)
        if parsed_next is None:
            continue
        next_base_tensor, const_2 = parsed_next
        if next_base_tensor is not mid:
            continue

        v1 = np.asarray(const_1.values)
        v2 = np.asarray(const_2.values)
        merged = v1 + v2 if node.op == "Add" else v1 * v2
        merged_const = gs.Constant(
            name=f"{next_node.name}_{node.op.lower()}_merged_const",
            values=merged,
        )

        # Rewrite the second node to consume original base directly.
        next_node.inputs = [base_tensor, merged_const]

        # First node becomes dead and is collected by cleanup().
        node.outputs.clear()
        if node.op == "Add":
            folded_add += 1
        else:
            folded_mul += 1
    return {"add_chain_folded": folded_add, "mul_chain_folded": folded_mul}


def _replace_matmul_add_with_gemm(graph: gs.Graph) -> int:
    """ViT-friendly rewrite: MatMul + bias Add -> Gemm for 2D linear layers."""
    replaced = 0
    for add_node in list(graph.nodes):
        if add_node.op != "Add" or len(add_node.inputs) != 2 or len(add_node.outputs) != 1:
            continue
        lhs, rhs = add_node.inputs
        lhs_prod = lhs.inputs[0] if lhs.inputs else None
        rhs_prod = rhs.inputs[0] if rhs.inputs else None
        mm_node = lhs_prod if lhs_prod and lhs_prod.op == "MatMul" else rhs_prod
        if mm_node is None or len(mm_node.inputs) != 2 or len(mm_node.outputs) != 1:
            continue

        # Add must be the only consumer of MatMul result to safely fuse.
        mm_out = mm_node.outputs[0]
        if len(mm_out.outputs) != 1 or mm_out.outputs[0] is not add_node:
            continue

        # Identify bias tensor (must be constant), and keep MatMul input order.
        bias_tensor = rhs if lhs_prod and lhs_prod.op == "MatMul" else lhs
        bias_const = _as_constant(bias_tensor)
        weight_const = _as_constant(mm_node.inputs[1])
        if bias_const is None or weight_const is None:
            continue

        # Conservative safety checks: only 2D MatMul and 1D bias.
        if _tensor_rank(mm_node.inputs[0]) != 2 or _tensor_rank(mm_node.inputs[1]) != 2:
            continue
        if not isinstance(bias_const.values, np.ndarray) or bias_const.values.ndim != 1:
            continue
        if not isinstance(weight_const.values, np.ndarray) or weight_const.values.ndim != 2:
            continue
        if weight_const.values.shape[1] != bias_const.values.shape[0]:
            continue

        gemm_out = gs.Variable(
            name=f"{mm_out.name}_gemm",
            dtype=add_node.outputs[0].dtype,
            shape=add_node.outputs[0].shape,
        )
        gemm_node = gs.Node(
            op="Gemm",
            name=f"{mm_node.name}_fused_gemm",
            inputs=[mm_node.inputs[0], mm_node.inputs[1], bias_const],
            outputs=[gemm_out],
            attrs={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
        )
        graph.nodes.append(gemm_node)

        _rewire_tensor_uses(add_node.outputs[0], gemm_out)
        add_node.outputs.clear()
        mm_node.outputs.clear()
        replaced += 1
    return replaced


def _replace_rank3_matmul_add_with_reshape_gemm(graph: gs.Graph) -> int:
    """Aggressive ViT rewrite for linear projections: [B,S,K]x[K,N]+b -> reshape+Gemm+reshape."""
    replaced = 0
    for add_node in list(graph.nodes):
        if add_node.op != "Add" or len(add_node.inputs) != 2 or len(add_node.outputs) != 1:
            continue
        lhs, rhs = add_node.inputs
        lhs_prod = lhs.inputs[0] if lhs.inputs else None
        rhs_prod = rhs.inputs[0] if rhs.inputs else None
        mm_node = lhs_prod if lhs_prod and lhs_prod.op == "MatMul" else rhs_prod
        if mm_node is None or len(mm_node.inputs) != 2 or len(mm_node.outputs) != 1:
            continue

        mm_out = mm_node.outputs[0]
        if len(mm_out.outputs) != 1 or mm_out.outputs[0] is not add_node:
            continue

        bias_tensor = rhs if lhs_prod and lhs_prod.op == "MatMul" else lhs
        bias_const = _as_constant(bias_tensor)
        weight_const = _as_constant(mm_node.inputs[1])
        x = mm_node.inputs[0]
        y = add_node.outputs[0]
        if bias_const is None or weight_const is None:
            continue

        x_shape = getattr(x, "shape", None)
        y_shape = getattr(y, "shape", None)
        if not _is_static_int_shape(x_shape) or not _is_static_int_shape(y_shape):
            continue
        if len(x_shape) != 3 or len(y_shape) != 3:
            continue
        if not isinstance(weight_const.values, np.ndarray) or weight_const.values.ndim != 2:
            continue
        if not isinstance(bias_const.values, np.ndarray) or bias_const.values.ndim != 1:
            continue

        k = int(weight_const.values.shape[0])
        n = int(weight_const.values.shape[1])
        if int(x_shape[-1]) != k or int(y_shape[-1]) != n or int(bias_const.values.shape[0]) != n:
            continue

        shape_flat = gs.Constant(
            name=f"{mm_node.name}_flat_shape",
            values=np.array([-1, k], dtype=np.int64),
        )
        x_flat = gs.Variable(
            name=f"{mm_node.name}_flat",
            dtype=x.dtype,
            shape=[None, k],
        )
        reshape_in = gs.Node(
            op="Reshape",
            name=f"{mm_node.name}_pre_reshape",
            inputs=[x, shape_flat],
            outputs=[x_flat],
        )

        gemm_out = gs.Variable(
            name=f"{mm_node.name}_gemm_flat_out",
            dtype=y.dtype,
            shape=[None, n],
        )
        gemm = gs.Node(
            op="Gemm",
            name=f"{mm_node.name}_reshape_gemm",
            inputs=[x_flat, weight_const, bias_const],
            outputs=[gemm_out],
            attrs={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
        )

        shape_restore = gs.Constant(
            name=f"{mm_node.name}_restore_shape",
            values=np.array(y_shape, dtype=np.int64),
        )
        restored = gs.Variable(name=f"{mm_node.name}_restored", dtype=y.dtype, shape=y_shape)
        reshape_out = gs.Node(
            op="Reshape",
            name=f"{mm_node.name}_post_reshape",
            inputs=[gemm_out, shape_restore],
            outputs=[restored],
        )

        graph.nodes.extend([reshape_in, gemm, reshape_out])
        _rewire_tensor_uses(y, restored)
        mm_node.outputs.clear()
        add_node.outputs.clear()
        replaced += 1
    return replaced


def _count_vit_attention_signatures(graph: gs.Graph) -> int:
    """Count common ViT attention topology signatures for visibility."""
    count = 0
    for node in graph.nodes:
        if node.op != "MatMul" or len(node.inputs) != 2:
            continue
        in0, in1 = node.inputs
        src0 = in0.inputs[0].op if in0.inputs else None
        src1 = in1.inputs[0].op if in1.inputs else None
        # Typical attention score/value matmuls involve Softmax on one side and
        # layout/value prep ops (Transpose/Squeeze/Mul) on the other side.
        softmax_on_side = src0 == "Softmax" or src1 == "Softmax"
        prep_on_other_side = (src0 in {"Transpose", "Squeeze", "Mul"}) or (
            src1 in {"Transpose", "Squeeze", "Mul"}
        )
        if softmax_on_side and prep_on_other_side:
            count += 1
    return count


def optimize(
    input_path: Path,
    output_path: Path,
    enable_vit_pass: bool = True,
    enable_vit_linear_gemm3d: bool = False,
) -> None:
    model = onnx.load(str(input_path))
    graph = gs.import_onnx(model)
    stats = _run_rewrite_passes(graph)
    vit_stats = {"gemm_fused": 0, "attn_signatures": 0, "gemm3d_fused": 0}
    if enable_vit_pass:
        vit_stats["attn_signatures"] = _count_vit_attention_signatures(graph)
        vit_stats["gemm_fused"] = _replace_matmul_add_with_gemm(graph)
        if enable_vit_linear_gemm3d:
            vit_stats["gemm3d_fused"] = _replace_rank3_matmul_add_with_reshape_gemm(graph)

    graph.cleanup().toposort()
    optimized = gs.export_onnx(graph)
    onnx.checker.check_model(optimized)
    onnx.save(optimized, str(output_path))

    print(f"[graph-opt] input:  {input_path}")
    print(f"[graph-opt] output: {output_path}")
    print(f"[graph-opt] removed transpose nodes: {stats['transpose_removed']}")
    print(f"[graph-opt] removed reshape nodes:   {stats['reshape_removed']}")
    print(f"[graph-opt] removed bypass nodes:    {stats['bypass_removed']}")
    print(f"[graph-opt] folded add chains:       {stats['add_chain_folded']}")
    print(f"[graph-opt] folded mul chains:       {stats['mul_chain_folded']}")
    print(f"[graph-opt] vit attn signatures:      {vit_stats['attn_signatures']}")
    print(f"[graph-opt] vit gemm fused:           {vit_stats['gemm_fused']}")
    print(f"[graph-opt] vit gemm3d fused:         {vit_stats['gemm3d_fused']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove redundant Transpose/Reshape patterns in ONNX.")
    parser.add_argument("--input", type=Path, required=True, help="Path to input ONNX.")
    parser.add_argument("--output", type=Path, default=None, help="Path to output ONNX. Defaults to in-place overwrite.")
    parser.add_argument(
        "--disable-vit-pass",
        action="store_true",
        help="Disable ViT-specific rewrites and diagnostics.",
    )
    parser.add_argument(
        "--enable-vit-linear-gemm3d",
        action="store_true",
        help="Aggressive: replace rank-3 MatMul+Add with reshape+Gemm+reshape.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input
    optimize(
        args.input,
        output,
        enable_vit_pass=not args.disable_vit_pass,
        enable_vit_linear_gemm3d=args.enable_vit_linear_gemm3d,
    )


if __name__ == "__main__":
    main()

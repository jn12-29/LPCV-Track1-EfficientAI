"""Small ONNX helpers shared by PTQ and compile scripts."""

from __future__ import annotations

import onnx
from onnx import GraphProto, ModelProto


def _sanitize_graph_value_info(graph: GraphProto) -> None:
    io_names = {x.name for x in graph.input} | {x.name for x in graph.output}
    keep = [vi for vi in graph.value_info if vi.name not in io_names]
    del graph.value_info[:]
    graph.value_info.extend(keep)
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _sanitize_graph_value_info(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    _sanitize_graph_value_info(g)


def sanitize_value_info_clashing_with_io(model: ModelProto) -> ModelProto:
    """Remove value_info entries whose names duplicate graph inputs/outputs."""
    _sanitize_graph_value_info(model.graph)
    return model

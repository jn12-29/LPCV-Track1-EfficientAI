# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import ast

from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


def get_components(model_id: str) -> list[str] | None:
    """
    Parse the <model_id>/model.py to extract component names from any decorator
    that calls:

    @CollectionModel.add_component(<component_class>, <component_name>)

    Returns a list of component names (as strings), or None if
    no CollectionModel.add_component call is found.
    """
    model_path = QAIHM_MODELS_ROOT / model_id / "model.py"
    with open(model_path) as f:
        source = f.read()

    tree = ast.parse(source, filename=model_path)
    components: list[str] = []

    # Iterate top-level nodes in source order (not ast.walk, which has no ordering guarantee)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id == "CollectionModel"
                    and decorator.func.attr == "add_component"
                    and len(decorator.args) >= 2
                ):
                    # component_name is the required second positional arg
                    component_arg = decorator.args[1]
                    if isinstance(component_arg, ast.Constant):
                        components.append(str(component_arg.value))
            if components:
                break  # only check first class defined in the file with added components
    return components if components else None

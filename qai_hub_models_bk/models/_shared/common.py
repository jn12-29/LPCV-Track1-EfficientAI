# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch

ReplaceT = TypeVar("ReplaceT", bound=torch.nn.Module)


def apply_module_function_recursively(
    module: torch.nn.Module,
    tgt_cls: type[ReplaceT],
    apply_fn: Callable[[ReplaceT, torch.nn.Module, str], None],
    parent_module: type[torch.nn.Module] | None = None,
) -> int:
    """
    Recursively calls a function on all modules of a given type.

    The function `apply_fn` passes in the module, the parent module, and the
    name of the module inside the parent module.

    Returns the number of modules the function was applied to.
    """
    count = 0
    for name, child in module.named_children():
        if isinstance(child, tgt_cls):
            if parent_module is None or isinstance(module, parent_module):
                apply_fn(child, module, name)
                count += 1
        else:
            count += apply_module_function_recursively(
                child, tgt_cls, apply_fn, parent_module
            )
    return count


def replace_module_recursively(
    module: torch.nn.Module,
    tgt_cls: type[torch.nn.Module],
    new_cls: type[torch.nn.Module],
    parent_module: type[torch.nn.Module] | None = None,
) -> None:
    """
    Replace all instances of `tgt_cls` with `new_cls`. If `parent_module` is
    specified, `tgt_cls` instance must be an immediate member of
    `parent_module` (useful for limiting replacement scope)
    """

    def apply_fn(child: torch.nn.Module, pmodule: torch.nn.Module, name: str) -> None:
        setattr(pmodule, name, new_cls(child))

    num_replaced = apply_module_function_recursively(
        module, tgt_cls, apply_fn, parent_module
    )
    assert num_replaced > 0, (
        f"replace_module_recursively: no instances of {tgt_cls.__name__} found "
        f"in {type(module).__name__}. The upstream model may have changed."
    )

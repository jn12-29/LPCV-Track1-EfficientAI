from __future__ import annotations

import torch
import torch.nn as nn

from mlp_reconstruction.mlp_blocks import MLPBlockInfo


@torch.no_grad()
def collect_all_O_orig(
    model: nn.Module,
    blocks: list[MLPBlockInfo],
    image_batches: list[torch.Tensor] | None,
    text_batches: list[torch.Tensor] | None,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, float | None]]:
    """Collect (O_all, gelu_ub) for all blocks in one forward pass per encoder.

    Returns dict: label -> (O_all, gelu_ub).
    Runs image batches once for all visual blocks, text batches once for all text blocks.
    """
    result: dict[str, tuple[torch.Tensor, float | None]] = {}

    visual_blocks = [b for b in blocks if b.encoder == 'visual']
    text_blocks = [b for b in blocks if b.encoder == 'text']

    for encoder_blocks, batches, encode_fn in [
        (visual_blocks, image_batches, model.encode_image),
        (text_blocks, text_batches, model.encode_text),
    ]:
        if not encoder_blocks or not batches:
            continue

        O_lists: dict[str, list[torch.Tensor]] = {b.label: [] for b in encoder_blocks}
        A_lists: dict[str, list[torch.Tensor]] = {}
        handles = []

        for info in encoder_blocks:
            lbl = info.label
            handles.append(info.mlp.register_forward_hook(
                lambda m, inp, out, lbl=lbl: O_lists[lbl].append(out.detach().cpu())
            ))
            if info.fc2 is not None:
                A_lists[lbl] = []
                handles.append(info.fc2.register_forward_pre_hook(
                    lambda m, inp, lbl=lbl: A_lists[lbl].append(inp[0].detach().cpu())
                ))

        model.eval()
        try:
            for batch in batches:
                encode_fn(batch.to(device))
        finally:
            for h in handles:
                h.remove()

        for info in encoder_blocks:
            lbl = info.label
            O_all = torch.cat(O_lists[lbl], dim=0)
            gelu_ub: float | None = None
            if lbl in A_lists:
                A_all = torch.cat(A_lists[lbl], dim=0)
                pos = A_all[A_all > 0].float()
                if pos.numel() > 1_000_000:
                    pos = pos[torch.randperm(pos.numel())[:1_000_000]]
                if pos.numel() > 0:
                    gelu_ub = float(torch.quantile(pos, 0.99))
            result[lbl] = (O_all, gelu_ub)

    return result


@torch.no_grad()
def collect_mlp_io(
    model: nn.Module,
    info: MLPBlockInfo,
    image_batches: list[torch.Tensor] | None,
    text_batches: list[torch.Tensor] | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    """Collect (X_all, O_all, gelu_ub) for one MLP block via forward hooks.

    X_all: block input (or conv output for fastvit_convmlp).
    O_all: block output.
    gelu_ub: 99th-percentile of fc2 pre-activation positives; None for conv_stem.
    All tensors returned on CPU.
    """
    X_list: list[torch.Tensor] = []
    O_list: list[torch.Tensor] = []
    A_list: list[torch.Tensor] = []

    h_x = info.register_x_hook(lambda t: X_list.append(t.detach().cpu()))
    h_o = info.mlp.register_forward_hook(
        lambda m, inp, out: O_list.append(out.detach().cpu())
    )
    h_a = None
    if info.fc2 is not None:
        h_a = info.fc2.register_forward_pre_hook(
            lambda m, inp: A_list.append(inp[0].detach().cpu())
        )

    model.eval()
    batches = text_batches if info.encoder == 'text' else image_batches
    try:
        for batch in batches:  # type: ignore[union-attr]
            if info.encoder == 'text':
                model.encode_text(batch.to(device))
            else:
                model.encode_image(batch.to(device))
    finally:
        h_x.remove()
        h_o.remove()
        if h_a is not None:
            h_a.remove()

    gelu_ub: float | None = None
    if A_list:
        A_all = torch.cat(A_list, dim=0)
        pos = A_all[A_all > 0].float()
        if pos.numel() > 1_000_000:
            pos = pos[torch.randperm(pos.numel())[:1_000_000]]
        if pos.numel() > 0:
            gelu_ub = float(torch.quantile(pos, 0.99))

    return torch.cat(X_list, dim=0), torch.cat(O_list, dim=0), gelu_ub

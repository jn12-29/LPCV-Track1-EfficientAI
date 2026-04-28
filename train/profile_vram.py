"""Measure peak VRAM for one forward+backward step at various batch sizes.

Usage:
    python train/profile_vram.py --model-name MobileCLIP2-B \
        --gpu-id 0 --loss-type siglip --num-hard-negatives 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import open_clip

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.clip_utils import _load_clip
from train.loss import SigLipLoss


def make_fake_batch(batch_size: int, num_hard_negatives: int, device: torch.device):
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    pos_tokens = torch.randint(0, 49408, (batch_size, 77), device=device)
    if num_hard_negatives > 0:
        neg_tokens = torch.randint(
            0, 49408, (batch_size, num_hard_negatives, 77), device=device
        )
        neg_mask = torch.ones(batch_size, num_hard_negatives, device=device)
    else:
        neg_tokens = torch.zeros(0, device=device)
        neg_mask = torch.zeros(batch_size, 0, device=device)
    return images, pos_tokens, neg_tokens, neg_mask


def measure_one(
    model,
    loss_fn,
    optimizer,
    batch_size: int,
    num_hard_negatives: int,
    loss_type: str,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    from torch.amp import GradScaler, autocast

    scaler = GradScaler(device=device.type, enabled=use_amp)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    images, pos_tokens, neg_tokens, neg_mask = make_fake_batch(
        batch_size, num_hard_negatives, device
    )

    with autocast(device_type=device.type, enabled=use_amp):
        image_features, pos_features, logit_scale = model(images, pos_tokens)

        if loss_type == "siglip":
            neg_feats_3d = None
            if neg_tokens.numel() > 0:
                flat = neg_tokens.view(-1, 77)
                neg_feats = model.encode_text(flat)
                neg_feats_3d = neg_feats.view(batch_size, num_hard_negatives, -1)
            loss = loss_fn(
                image_features,
                pos_features,
                logit_scale,
                extra_text_features=neg_feats_3d,
                extra_text_mask=neg_mask if neg_feats_3d is not None else None,
            )
        else:
            loss = loss_fn(image_features, pos_features, logit_scale)
            if num_hard_negatives > 0:
                img_n = F.normalize(image_features, dim=-1)
                pos_n = F.normalize(pos_features, dim=-1)
                flat = neg_tokens.view(-1, 77)
                neg_feats = model.encode_text(flat)
                neg_n = F.normalize(
                    neg_feats.view(batch_size, num_hard_negatives, -1), dim=-1
                )
                sims = torch.einsum("bd,bkd->bk", img_n, neg_n)
                pos_sim = (img_n * pos_n).sum(-1, keepdim=True)
                hn_loss = F.relu(sims - pos_sim + 0.2).mean()
                loss = loss + 0.5 * hn_loss

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize(device)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    return peak_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="MobileCLIP2-B")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--loss-type", default="siglip", choices=["clip", "siglip"])
    parser.add_argument("--num-hard-negatives", type=int, default=4)
    parser.add_argument(
        "--batch-sizes",
        default="32,64,96,128,192,256,320,384,512",
        help="Comma-separated list of batch sizes to probe",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable automatic mixed precision"
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    total_mb = torch.cuda.get_device_properties(device).total_memory / 1024**2
    amp_str = "AMP fp16" if args.amp else "fp32"
    print(f"Total VRAM: {total_mb:.0f} MB ({total_mb/1024:.1f} GB) | {amp_str}\n")

    model, _, _ = _load_clip(args.model_name, device)
    model.train()

    if args.loss_type == "siglip":
        loss_fn = SigLipLoss(rank=0, world_size=1).to(device)
        trainable = list(model.parameters()) + list(loss_fn.parameters())
    else:
        loss_fn = open_clip.ClipLoss(cache_labels=True)
        trainable = list(model.parameters())

    optimizer = optim.AdamW(trainable, lr=1e-6)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(
        f"{'BatchSize':>10} {'PeakVRAM(MB)':>14} {'PeakVRAM(GB)':>14} {'Util%':>8} {'Status':>8}"
    )
    print("-" * 60)

    results = []
    for bs in batch_sizes:
        try:
            peak_mb = measure_one(
                model,
                loss_fn,
                optimizer,
                bs,
                args.num_hard_negatives,
                args.loss_type,
                device,
                use_amp=args.amp,
            )
            util_pct = peak_mb / total_mb * 100
            status = "OOM" if peak_mb > total_mb else "OK"
            print(
                f"{bs:>10} {peak_mb:>14.0f} {peak_mb/1024:>14.2f} {util_pct:>8.1f} {status:>8}"
            )
            results.append((bs, peak_mb))
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>10} {'OOM':>14} {'OOM':>14} {'---':>8} {'OOM':>8}")
            torch.cuda.empty_cache()

    if len(results) >= 2:
        xs = np.array([r[0] for r in results], dtype=float)
        ys = np.array([r[1] for r in results], dtype=float)
        # linear fit: vram = a * batch + b
        coeffs = np.polyfit(xs, ys, 1)
        a, b = coeffs
        print(f"\nLinear fit: VRAM(MB) ≈ {a:.1f} × batch_size + {b:.0f}")
        print(f"  → per-sample cost: {a:.1f} MB")
        print(f"  → fixed overhead:  {b:.0f} MB ({b/1024:.2f} GB)")
        print()
        print("Estimated max batch size by VRAM:")
        for vram_gb in [24, 40, 48, 80]:
            vram_mb = vram_gb * 1024
            max_bs = int((vram_mb - b) / a)
            print(f"  {vram_gb:>3} GB → batch_size ≤ {max(max_bs, 0)}")


if __name__ == "__main__":
    main()

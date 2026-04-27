from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.loss import compute_hard_negative_loss
from train.train_utils import unwrap_model
from utils.data_utils import recall_at_k
from utils.preprocess import preprocess_image


@torch.no_grad()
def eval_val_loss(
    model,
    dataloader,
    loss_fn,
    loss_type: str,
    device: torch.device,
    hard_negative_weight: float,
    hard_negative_margin: float,
    hard_negative_loss_type: str,
    amp_enabled: bool,
) -> Dict[str, float]:
    """Compute validation loss over a DataLoader. Returns averaged loss metrics."""
    raw_model = unwrap_model(model)
    raw_model.eval()

    running_total = 0.0
    running_clip = 0.0
    running_hn = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        positive_tokens = batch["positive_tokens"].to(device, non_blocking=True)
        negative_tokens = batch["negative_tokens"].to(device, non_blocking=True)
        negative_mask = batch["negative_mask"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            image_features, positive_features, logit_scale = raw_model(
                images, positive_tokens
            )

            if loss_type == "siglip":
                negative_features_3d: Optional[torch.Tensor] = None
                if negative_tokens.numel() > 0:
                    flat_neg = negative_tokens.view(-1, negative_tokens.shape[-1])
                    _, flat_feats, _ = raw_model(text=flat_neg)
                    negative_features_3d = flat_feats.view(
                        *negative_tokens.shape[:2], -1
                    )
                main_loss = loss_fn(
                    image_features,
                    positive_features,
                    logit_scale,
                    extra_text_features=negative_features_3d,
                    extra_text_mask=(
                        negative_mask if negative_features_3d is not None else None
                    ),
                )
                clip_loss = main_loss
                hard_negative_loss = main_loss.new_zeros(())
                total_loss = main_loss
            else:
                # Local InfoNCE (no cross-GPU gather; val is always single-process)
                image_f = F.normalize(image_features, dim=-1)
                positive_f = F.normalize(positive_features, dim=-1)
                scale = logit_scale.exp()
                sims = image_f @ positive_f.T * scale
                labels = torch.arange(len(sims), device=device)
                clip_loss = (
                    F.cross_entropy(sims, labels) + F.cross_entropy(sims.T, labels)
                ) / 2

                positive_scores = (image_f * positive_f).sum(dim=-1)
                if negative_tokens.numel() > 0:
                    flat_neg = negative_tokens.view(-1, negative_tokens.shape[-1])
                    _, flat_feats, _ = raw_model(text=flat_neg)
                    flat_feats = F.normalize(flat_feats, dim=-1)
                    neg_feats = flat_feats.view(*negative_tokens.shape[:2], -1)
                    hard_negative_loss = compute_hard_negative_loss(
                        image_features=image_f,
                        negative_features=neg_feats,
                        negative_mask=negative_mask,
                        positive_scores=positive_scores,
                        margin=hard_negative_margin,
                        strategy=hard_negative_loss_type,
                    )
                else:
                    hard_negative_loss = clip_loss.new_zeros(())
                total_loss = clip_loss + hard_negative_weight * hard_negative_loss

        running_total += total_loss.item()
        running_clip += clip_loss.item()
        running_hn += hard_negative_loss.item()
        num_batches += 1

    raw_model.train()
    n = max(num_batches, 1)
    return {
        "val_total_loss": running_total / n,
        "val_loss": running_clip / n,
        "val_hard_negative_loss": running_hn / n,
    }


@torch.no_grad()
def eval_val_recall(
    model,
    tokenizer,
    records: List[Dict],
    device: torch.device,
    batch_size: int = 32,
    k: int = 10,
) -> Dict[str, float]:
    """Compute Recall@k on JSONL-format records using all positives as the text corpus."""
    raw_model = unwrap_model(model)
    raw_model.eval()

    # Build unified text corpus from all positives across all records
    all_texts: List[str] = []
    text_to_idx: Dict[str, int] = {}
    positive_indices: List[List[int]] = []

    for record in records:
        pos_idxs: List[int] = []
        for text in record["positives"]:
            if text not in text_to_idx:
                text_to_idx[text] = len(all_texts)
                all_texts.append(text)
            pos_idxs.append(text_to_idx[text])
        positive_indices.append(pos_idxs)

    # Encode images
    image_embeds_list: List[torch.Tensor] = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        imgs = [
            preprocess_image(Image.open(r["image_path"]).convert("RGB")) for r in batch
        ]
        pixel_values = torch.stack(imgs, dim=0).to(device)
        feats = raw_model.encode_image(pixel_values)
        image_embeds_list.append(F.normalize(feats, dim=-1).cpu())
    image_embeds = torch.cat(image_embeds_list, dim=0)

    # Encode texts
    text_embeds_list: List[torch.Tensor] = []
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i : i + batch_size]
        tokens = tokenizer(batch_texts).to(device)
        feats = raw_model.encode_text(tokens)
        text_embeds_list.append(F.normalize(feats, dim=-1).cpu())
    text_embeds = torch.cat(text_embeds_list, dim=0)

    recall = recall_at_k(
        image_embeds.numpy(), text_embeds.numpy(), positive_indices, k=k
    )
    raw_model.train()
    return {f"val_recall@{k}": recall}

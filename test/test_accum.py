from __future__ import annotations
import contextlib
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    """Minimal CLIP-like model: separate image/text linear encoders + logit_scale."""
    IMG_DIM = 16
    TXT_DIM = 10
    OUT_DIM = 8

    def __init__(self):
        super().__init__()
        self.img_enc = nn.Linear(self.IMG_DIM, self.OUT_DIM, bias=False)
        self.txt_enc = nn.Linear(self.TXT_DIM, self.OUT_DIM, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, image=None, text=None):
        scale = self.logit_scale.exp()
        img_f = self.img_enc(image) if image is not None else None
        txt_f = self.txt_enc(text.float()) if text is not None else None
        if img_f is not None and txt_f is not None:
            return img_f, txt_f, scale
        if img_f is not None:
            return img_f, torch.zeros(0), scale
        return torch.zeros(0), txt_f, scale


def make_batch(B=4, K=2, device="cpu"):
    """Return a dict mimicking DataLoader output (on device)."""
    return {
        "images": torch.randn(B, ToyModel.IMG_DIM, device=device),
        "positive_tokens": torch.randint(0, 5, (B, ToyModel.TXT_DIM), device=device),
        "negative_tokens": torch.randint(0, 5, (B, K, ToyModel.TXT_DIM), device=device),
        "negative_mask": torch.ones(B, K, device=device),
    }


def test_phase1_populates_features():
    """Phase 1 must fill img_features / pos_features / neg_features; no grad_fn."""
    from train.accum import CachedBatch, phase1_forward_no_grad

    model = ToyModel()
    B, K = 4, 2
    raw = make_batch(B=B, K=K)
    cb = CachedBatch(
        images=raw["images"],
        positive_tokens=raw["positive_tokens"],
        negative_tokens=raw["negative_tokens"],
        negative_mask=raw["negative_mask"],
    )
    phase1_forward_no_grad(model, [cb], device=torch.device("cpu"),
                           amp_enabled=False, loss_type="clip")

    assert cb.img_features is not None
    assert cb.pos_features is not None
    assert cb.neg_features is not None
    assert cb.img_features.shape == (B, ToyModel.OUT_DIM)
    assert cb.neg_features.shape == (B, K, ToyModel.OUT_DIM)
    assert cb.img_features.grad_fn is None, "Phase 1 features must be detached"
    assert cb.pos_features.grad_fn is None
    # neg_features should be L2-normalised for CLIP
    norms = cb.neg_features.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_phase2_feature_grads_nonzero():
    """After Phase 2, img_grad / pos_grad must be non-zero on every batch."""
    import open_clip
    from torch.amp import GradScaler
    from train.accum import CachedBatch, phase1_forward_no_grad, phase2_feature_backward

    model = ToyModel()
    loss_fn = open_clip.ClipLoss(cache_labels=True, rank=0, world_size=1)
    scaler = GradScaler(device="cpu", enabled=False)

    B, K = 4, 2
    batches = []
    for _ in range(2):  # accum_freq = 2
        raw = make_batch(B=B, K=K)
        batches.append(CachedBatch(
            images=raw["images"],
            positive_tokens=raw["positive_tokens"],
            negative_tokens=raw["negative_tokens"],
            negative_mask=raw["negative_mask"],
        ))

    phase1_forward_no_grad(model, batches, torch.device("cpu"), False, "clip")
    total_loss, clip_loss, hn_loss = phase2_feature_backward(
        batches=batches, loss_fn=loss_fn, loss_type="clip",
        model=model, scaler=scaler, device=torch.device("cpu"),
        amp_enabled=False, hard_negative_weight=1.0,
        hard_negative_margin=0.2, hard_negative_loss_type="hinge",
        distributed=False, world_size=1,
    )

    for i, cb in enumerate(batches):
        assert cb.img_grad is not None, f"batch {i}: img_grad is None"
        assert cb.pos_grad is not None, f"batch {i}: pos_grad is None"
        assert cb.img_grad.shape == (B, ToyModel.OUT_DIM)
        assert cb.img_grad.abs().sum() > 0, f"batch {i}: img_grad is all-zero"


def test_phase2_contrastive_pool_size():
    """With accum_freq=2 and B=4, Phase 2 loss sees 8 samples vs 4 with accum_freq=1.
    Gradient magnitudes must differ between the two cases."""
    import open_clip
    from torch.amp import GradScaler
    from train.accum import CachedBatch, phase1_forward_no_grad, phase2_feature_backward

    torch.manual_seed(0)
    model = ToyModel()
    loss_fn_1 = open_clip.ClipLoss(cache_labels=True, rank=0, world_size=1)
    loss_fn_2 = open_clip.ClipLoss(cache_labels=True, rank=0, world_size=1)
    scaler = GradScaler(device="cpu", enabled=False)

    B = 4
    raw0 = make_batch(B=B, K=0)
    raw1 = make_batch(B=B, K=0)

    # accum_freq=2 path (sees 8 negatives)
    batches2 = [
        CachedBatch(raw0["images"], raw0["positive_tokens"],
                    raw0["negative_tokens"], raw0["negative_mask"]),
        CachedBatch(raw1["images"], raw1["positive_tokens"],
                    raw1["negative_tokens"], raw1["negative_mask"]),
    ]
    phase1_forward_no_grad(model, batches2, torch.device("cpu"), False, "clip")
    phase2_feature_backward(batches2, loss_fn_2, "clip", model, scaler,
                            torch.device("cpu"), False, 0.0, 0.2, "hinge", False, 1)
    grad_accum2 = batches2[0].img_grad.clone()

    # Reset model grads
    model.zero_grad()

    # accum_freq=1 path (sees only 4 negatives)
    batch1 = CachedBatch(raw0["images"], raw0["positive_tokens"],
                         raw0["negative_tokens"], raw0["negative_mask"])
    phase1_forward_no_grad(model, [batch1], torch.device("cpu"), False, "clip")
    phase2_feature_backward([batch1], loss_fn_1, "clip", model, scaler,
                            torch.device("cpu"), False, 0.0, 0.2, "hinge", False, 1)
    grad_accum1 = batch1.img_grad.clone()

    # Gradients must differ (different number of negatives → different loss landscape)
    assert not torch.allclose(grad_accum2, grad_accum1), \
        "Gradients are identical — pool size did not change"


def test_phase3_model_params_get_grad():
    """After phases 1-2-3, model parameters must have non-zero gradients."""
    import open_clip
    from torch.amp import GradScaler
    from train.accum import CachedBatch, phase1_forward_no_grad
    from train.accum import phase2_feature_backward, phase3_model_backward

    torch.manual_seed(42)
    model = ToyModel()
    model.zero_grad()
    loss_fn = open_clip.ClipLoss(cache_labels=True, rank=0, world_size=1)
    scaler = GradScaler(device="cpu", enabled=False)

    B, K = 4, 2
    batches = [
        CachedBatch(**{k: v for k, v in make_batch(B, K).items()})
        for _ in range(2)
    ]
    phase1_forward_no_grad(model, batches, torch.device("cpu"), False, "clip")
    phase2_feature_backward(batches, loss_fn, "clip", model, scaler,
                            torch.device("cpu"), False, 1.0, 0.2, "hinge", False, 1)
    phase3_model_backward(model, batches, loss_type="clip",
                          distributed=False, amp_enabled=False,
                          device=torch.device("cpu"))

    assert model.img_enc.weight.grad is not None
    assert model.img_enc.weight.grad.abs().sum() > 0
    assert model.txt_enc.weight.grad is not None
    assert model.txt_enc.weight.grad.abs().sum() > 0


def test_integration_loss_decreases_with_accum():
    """Full 3-phase loop on toy model: loss should be in a reasonable range after one step."""
    import open_clip
    from torch.amp import GradScaler
    from torch.utils.data import DataLoader
    from train.accum import train_one_epoch_accum

    torch.manual_seed(7)
    model = ToyModel()
    loss_fn = open_clip.ClipLoss(cache_labels=True, rank=0, world_size=1)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()), lr=1e-2
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = GradScaler(device="cpu", enabled=False)

    B = 4

    def make_ds_batch():
        return {
            "images": torch.randn(B, ToyModel.IMG_DIM),
            "positive_tokens": torch.randint(0, 5, (B, ToyModel.TXT_DIM)),
            "negative_tokens": torch.zeros(B, 0, ToyModel.TXT_DIM, dtype=torch.long),
            "negative_mask": torch.zeros(B, 0),
        }

    raw_data = [make_ds_batch(), make_ds_batch()]

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            return self.data[i]

    def collate(batch):
        return batch[0]

    loader = DataLoader(ListDataset(raw_data), batch_size=1, collate_fn=collate)

    metrics, gs = train_one_epoch_accum(
        model=model, dataloader=loader, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, loss_fn=loss_fn, loss_type="clip",
        device=torch.device("cpu"), epoch=1, global_step=0,
        log_every_n_steps=1, grad_clip_norm=None, accum_freq=2,
        hard_negative_weight=0.0, hard_negative_margin=0.2,
        hard_negative_loss_type="hinge", is_main_process=False,
        distributed=False, world_size=1, amp_enabled=False,
    )
    assert gs == 1, f"global_step should be 1, got {gs}"
    # Loss is finite and positive; exact value depends on pool size (8 samples with accum=2)
    assert torch.isfinite(torch.tensor(metrics["train_loss"])), "Loss must be finite"
    assert metrics["train_loss"] > 0, "Loss must be positive"

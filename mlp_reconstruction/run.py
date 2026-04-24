from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: nn.Module,
    model_name: str,
    relu_labels: list[str],
    path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'relu_blocks': relu_labels,
    }, path)


def load_reconstructed_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Module, object]:
    """Load a reconstructed checkpoint. Returns (model, tokenizer)."""
    from utils.clip_utils import _load_clip
    from mlp_reconstruction.mlp_blocks import apply_relu_blocks

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_name = ckpt['model_name']
    relu_labels = ckpt['relu_blocks']

    model, _, tokenizer = _load_clip(model_name, torch.device('cpu'))
    apply_relu_blocks(model, relu_labels)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device).eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from utils.clip_utils import _load_clip
    from mlp_reconstruction.mlp_blocks import iter_mlp_blocks, replace_gelu_with_relu, apply_relu_blocks
    from mlp_reconstruction.calibrate import VGCalibrationLoader, collect_mlp_io
    from mlp_reconstruction.distill import distill_mlp, verify_reconstruction

    device = torch.device(args.device)

    model, _, tokenizer = _load_clip(args.model_name, device, checkpoint_path=args.checkpoint_path)
    model.eval()

    relu_labels: list[str] = []
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        apply_relu_blocks(model, ckpt['relu_blocks'])
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        relu_labels = list(ckpt['relu_blocks'])
        print(f'Resumed from {args.resume_from}, {len(relu_labels)} blocks already done')

    loader = VGCalibrationLoader(
        args.calib_jsonl, args.project_root, tokenizer,
        n_calib=args.n_calib, batch_size=args.calib_batch_size,
    )
    img_batches = loader.get_image_batches()
    txt_batches = loader.get_text_batches()
    print(f'Calibration: {len(img_batches) * args.calib_batch_size} images, '
          f'{len(txt_batches) * args.calib_batch_size} texts')

    blocks = iter_mlp_blocks(model)
    done_set = set(relu_labels)
    skip_mode = args.skip_to is not None and args.skip_to not in done_set

    for info in tqdm(blocks, desc='Reconstructing MLP blocks'):
        if skip_mode:
            if info.label == args.skip_to:
                skip_mode = False
            else:
                if info.label not in done_set:
                    replace_gelu_with_relu(info)
                    relu_labels.append(info.label)
                    done_set.add(info.label)
                continue

        print(f'\n=== {info.label} ({info.kind}) ===')

        X_all, O_all = collect_mlp_io(model, info, img_batches, txt_batches, device)
        print(f'  X={tuple(X_all.shape)}, O={tuple(O_all.shape)}')

        final_loss = distill_mlp(
            model, info, X_all, O_all,
            lr=args.lr, batch_size=args.batch_size, n_iters=args.n_iters,
            alpha=args.alpha, device=device, log_every=args.log_every,
            aph_mode=args.aph_mode,
        )
        relu_labels.append(info.label)
        done_set.add(info.label)

        cos_sim = verify_reconstruction(info, X_all, O_all, device)
        status = 'OK' if cos_sim >= 0.99 else 'WARN'
        print(f'  [{status}] final_loss={final_loss:.6f}, cos_sim={cos_sim:.4f}')

        _save_checkpoint(model, args.model_name, relu_labels, args.output + '.tmp')

    _save_checkpoint(model, args.model_name, relu_labels, args.output)
    print(f'\nDone. {len(relu_labels)}/{len(blocks)} blocks reconstructed.')
    print(f'Saved: {args.output}')


def cmd_eval(args: argparse.Namespace) -> None:
    from pipeline.dataset import RetrievalEvalDataset
    from pipeline.eval_local import _encode_images_torch, _encode_texts_torch
    from utils.data_utils import recall_at_k

    device = torch.device(args.device)
    model, tokenizer = load_reconstructed_model(args.checkpoint_path, device)

    image_dataset = RetrievalEvalDataset(
        root_dir=args.root_dir,
        image_to_text_csv=args.image_to_text_csv,
        textnums_to_texts_csv=args.textnums_to_texts_csv,
        mode='image',
    )
    eval_data = image_dataset.get_image_to_text_eval_data()

    image_embeds = _encode_images_torch(model, image_dataset, device, args.batch_size)
    text_embeds = _encode_texts_torch(model, tokenizer, eval_data.texts, device, args.batch_size)

    recall = recall_at_k(
        image_embeds.numpy(), text_embeds.numpy(),
        eval_data.positive_text_indices, k=args.k,
    )
    print(f'Recall@{args.k}: {recall:.4f}')


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='MLP Reconstruction for MobileCLIP2')
    sub = p.add_subparsers(dest='command', required=True)

    t = sub.add_parser('train', help='Run MLP reconstruction distillation')
    t.add_argument('--model-name', default='MobileCLIP2-S0')
    t.add_argument('--checkpoint-path', default=None)
    t.add_argument('--calib-jsonl', default='build_datasets/data/vg_llm_contrastive.jsonl')
    t.add_argument('--project-root', default='.')
    t.add_argument('--n-calib', type=int, default=1024)
    t.add_argument('--calib-batch-size', type=int, default=32)
    t.add_argument('--output', default='checkpoints/mlp_relu.pt')
    t.add_argument('--device', default='cuda')
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--batch-size', type=int, default=32)
    t.add_argument('--n-iters', type=int, default=20000)
    t.add_argument('--alpha', type=float, default=2.0)
    t.add_argument('--aph-mode', default='uniform', choices=['uniform', 'magnitude'])
    t.add_argument('--log-every', type=int, default=500)
    t.add_argument('--skip-to', default=None)
    t.add_argument('--resume-from', default=None)

    e = sub.add_parser('eval', help='Evaluate a reconstructed checkpoint')
    e.add_argument('--checkpoint-path', required=True)
    e.add_argument('--root-dir', default='sample_data')
    e.add_argument('--image-to-text-csv', default='sample_data/img_list.csv')
    e.add_argument('--textnums-to-texts-csv', default='sample_data/txt_list.csv')
    e.add_argument('--k', type=int, default=10)
    e.add_argument('--batch-size', type=int, default=64)
    e.add_argument('--device', default='cuda')

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'eval':
        cmd_eval(args)


if __name__ == '__main__':
    main()

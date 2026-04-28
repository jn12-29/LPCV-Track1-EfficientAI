from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import open_clip
import onnxruntime as ort
import torch
import torch.nn.functional as F

from pipeline.dataset import RetrievalEvalDataset
from pipeline.export_onnx import _resize_vit_pos_embed
from train.data import load_jsonl_records, split_val_records
from train.val_step import eval_val_recall
from utils.clip_utils import _load_clip
from utils.data_utils import _batched, recall_at_k
import numpy as np


# ---------------------------------------------------------------------------
# ONNX encoding
# ---------------------------------------------------------------------------


def _encode_images_onnx(
    sess: "ort.InferenceSession",
    image_dataset: RetrievalEvalDataset,
    batch_size: int,
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for _, batch_indices in _batched(list(range(len(image_dataset))), batch_size):
        batch_images = [image_dataset[idx]["image"] for idx in batch_indices]
        pixel_values = torch.stack(batch_images, dim=0).numpy()
        out = sess.run(None, {"image": pixel_values})[0]
        features.append(F.normalize(torch.from_numpy(out), dim=-1))
    return torch.cat(features, dim=0)


def _encode_texts_onnx(
    sess: "ort.InferenceSession",
    tokenizer,
    texts: Sequence[str],
    batch_size: int,
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for _, batch_texts in _batched(list(texts), batch_size):
        tokens = (
            tokenizer(
                list(batch_texts),
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )["input_ids"]
            .numpy()
            .astype(np.int32)
        )
        out = sess.run(None, {"text": tokens})[0]
        features.append(F.normalize(torch.from_numpy(out), dim=-1))
    return torch.cat(features, dim=0)


# ---------------------------------------------------------------------------
# Torch encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def _encode_texts_torch(
    model, tokenizer, texts: Sequence[str], device: torch.device, batch_size: int
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for _, batch_texts in _batched(list(texts), batch_size):
        text_tokens = tokenizer(
            list(batch_texts),
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )["input_ids"].to(device)
        text_features = model.encode_text(text_tokens)
        features.append(F.normalize(text_features, dim=-1).cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def _encode_images_torch(
    model, image_dataset: RetrievalEvalDataset, device: torch.device, batch_size: int,
    image_size: int = 224, image_mode: str = "resize",
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for _, batch_indices in _batched(list(range(len(image_dataset))), batch_size):
        batch_images = [image_dataset[idx]["image"] for idx in batch_indices]
        pixel_values = torch.stack(batch_images, dim=0).to(device)
        if image_size != 224:
            if image_mode == "crop":
                h, w = pixel_values.shape[-2], pixel_values.shape[-1]
                top  = (h - image_size) // 2
                left = (w - image_size) // 2
                pixel_values = pixel_values[:, :, top:top + image_size, left:left + image_size]
            else:
                pixel_values = F.interpolate(pixel_values, size=(image_size, image_size), mode="bilinear", align_corners=False)
        image_features = model.encode_image(pixel_values)
        features.append(F.normalize(image_features, dim=-1).cpu())
    return torch.cat(features, dim=0)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def run_clip_retrieval_eval(
    root_dir: str | Path,
    image_to_text_csv: str | Path,
    textnums_to_texts_csv: str | Path,
    model_name: str = "MobileCLIP2-S0",
    batch_size: int = 32,
    k: int = 10,
    device: str | None = None,
    checkpoint_path: str | None = None,
    onnx_dir: str | Path | None = None,
    print_model: bool = False,
    image_size: int = 224,
    image_mode: str = "resize",
) -> Dict[str, float]:
    root_dir = Path(root_dir)
    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_str)

    image_dataset = RetrievalEvalDataset(
        root_dir=root_dir,
        image_to_text_csv=image_to_text_csv,
        textnums_to_texts_csv=textnums_to_texts_csv,
        mode="image",
    )
    eval_data = image_dataset.get_image_to_text_eval_data()

    if onnx_dir is not None:
        onnx_dir = Path(onnx_dir)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device_str.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        img_sess = ort.InferenceSession(
            str(onnx_dir / "image_encoder.onnx"), providers=providers
        )
        txt_sess = ort.InferenceSession(
            str(onnx_dir / "text_encoder.onnx"), providers=providers
        )
        # tokenizer = open_clip.get_tokenizer("ViT-B-32")
        from transformers import CLIPTokenizer

        pretrained_tokenizer = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_tokenizer, local_files_only=True
        )
        tokenizer.add_special_tokens({"cls_token": tokenizer.eos_token})
        # QAI Hub input spec requires batch=1
        image_embeds = _encode_images_onnx(img_sess, image_dataset, batch_size=1)
        text_embeds = _encode_texts_onnx(
            txt_sess, tokenizer, eval_data.texts, batch_size=1
        )
    else:
        model, _, tokenizer = _load_clip(
            model_name, device_obj, checkpoint_path=checkpoint_path
        )
        if print_model:
            print(model)
        if image_size != 224:
            _resize_vit_pos_embed(model, image_size)
        image_embeds = _encode_images_torch(
            model, image_dataset, device_obj, batch_size, image_size=image_size, image_mode=image_mode
        )
        text_embeds = _encode_texts_torch(
            model, tokenizer, eval_data.texts, device_obj, batch_size
        )

    image_to_text_recall = recall_at_k(
        image_embeds.numpy(),
        text_embeds.numpy(),
        eval_data.positive_text_indices,
        k=k,
    )
    return {f"image_to_text_recall@{k}": image_to_text_recall}


@torch.no_grad()
def eval_recall_with_model(
    model,
    tokenizer,
    root_dir: str | Path,
    image_to_text_csv: str | Path,
    textnums_to_texts_csv: str | Path,
    device: torch.device,
    batch_size: int = 32,
    k: int = 10,
) -> Dict[str, float]:
    """Compute sample-set Recall@k using a pre-loaded model (no checkpoint reload)."""
    from train.train_utils import unwrap_model

    raw_model = unwrap_model(model)
    raw_model.eval()

    root_dir = Path(root_dir)
    image_dataset = RetrievalEvalDataset(
        root_dir=root_dir,
        image_to_text_csv=image_to_text_csv,
        textnums_to_texts_csv=textnums_to_texts_csv,
        mode="image",
    )
    eval_data = image_dataset.get_image_to_text_eval_data()
    image_embeds = _encode_images_torch(raw_model, image_dataset, device, batch_size)
    text_embeds = _encode_texts_torch(
        raw_model, tokenizer, eval_data.texts, device, batch_size
    )

    raw_model.train()
    recall = recall_at_k(
        image_embeds.numpy(),
        text_embeds.numpy(),
        eval_data.positive_text_indices,
        k=k,
    )
    return {f"sample_recall@{k}": recall}


def eval_val_recall_from_jsonl(
    jsonl_path: str | Path,
    val_split_size: int,
    val_split_seed: Optional[int] = None,
    model_name: str = "MobileCLIP2-S0",
    checkpoint_path: str | None = None,
    device: str | None = None,
    batch_size: int = 32,
    k: int = 10,
    print_model: bool = False,
) -> Dict[str, float]:
    """Compute Recall@k on the val split of a training JSONL, using the same
    split logic as finetune.py (split_val_records with identical defaults)."""
    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_str)

    model, _, tokenizer = _load_clip(
        model_name, device_obj, checkpoint_path=checkpoint_path
    )
    if print_model:
        print(model)

    all_records = load_jsonl_records(jsonl_path)
    _, val_records = split_val_records(all_records, val_split_size, val_split_seed)
    if not val_records:
        raise ValueError(
            "val_split_size=0 produces an empty val set; set --val-split-size > 0."
        )

    return eval_val_recall(model, tokenizer, val_records, device_obj, batch_size, k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP image-text retrieval evaluation (torch)"
    )
    parser.add_argument("--root-dir", type=str, default="./sample_data")
    parser.add_argument(
        "--image-to-text-csv", type=str, default="./sample_data/img_list.csv"
    )
    parser.add_argument(
        "--textnums-to-texts-csv", type=str, default="./sample_data/txt_list.csv"
    )
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=None,
        help="Path to exported ONNX dir (e.g. exported_MobileCLIP2-S0_onnx). "
        "If set, uses ONNX inference instead of PyTorch.",
    )
    parser.add_argument(
        "--print-model",
        action="store_true",
        help="Print model architecture after loading (torch mode only).",
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Downsample images to this resolution inside the model (torch path). Match export_onnx.py --image-size.",
    )
    parser.add_argument(
        "--image-mode", type=str, default="resize", choices=["resize", "crop"],
        help="How to reduce image resolution: 'resize' (bilinear) or 'crop' (center crop).",
    )

    # --- Val-split mode ---
    parser.add_argument(
        "--val",
        action="store_true",
        help="Evaluate on the val split of --jsonl-path instead of the sample-set CSVs.",
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="./build_datasets/data/dataset_raw_contrastive.jsonl",
        help="Path to training JSONL (used with --val).",
    )
    parser.add_argument(
        "--val-split-size",
        type=int,
        default=1024,
        help="Number of tail records held out as the val set (must match finetune.py).",
    )
    parser.add_argument(
        "--val-split-seed",
        type=int,
        default=None,
        help="Random seed for val split (default None = deterministic tail split). Must match finetune.py.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val:
        metrics = eval_val_recall_from_jsonl(
            jsonl_path=args.jsonl_path,
            val_split_size=args.val_split_size,
            val_split_seed=args.val_split_seed,
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            batch_size=args.batch_size,
            k=args.k,
            print_model=args.print_model,
        )
    else:
        metrics = run_clip_retrieval_eval(
            root_dir=args.root_dir,
            image_to_text_csv=args.image_to_text_csv,
            textnums_to_texts_csv=args.textnums_to_texts_csv,
            model_name=args.model_name,
            batch_size=args.batch_size,
            k=args.k,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            onnx_dir=args.onnx_dir,
            print_model=args.print_model,
            image_size=args.image_size,
            image_mode=args.image_mode,
        )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

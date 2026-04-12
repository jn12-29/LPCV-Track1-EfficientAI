from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

from matplotlib.style import available
import torch
import torch.nn.functional as F
import open_clip

from dataset import RetrievalEvalDataset
from utils import calculate_recall_at_k


def _load_clip(model_name: str, device: torch.device):
    print(f"Loading CLIP model '{model_name}'...")
    available_models_tuple = open_clip.list_pretrained()
    available_models_dict = {}
    for temp in available_models_tuple:
        name, ckpt = temp
        if name not in available_models_dict:
            available_models_dict[name] = ckpt
    if model_name in available_models_dict.keys():
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=available_models_dict[model_name]
        )
    else:
        raise ValueError(f"Available models: {open_clip.list_pretrained()}")
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def _batched(iterable: Sequence, batch_size: int):
    for start in range(0, len(iterable), batch_size):
        yield start, iterable[start : start + batch_size]


@torch.no_grad()
def encode_texts(
    model, tokenizer, texts: Sequence[str], device: torch.device, batch_size: int
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for _, batch_texts in _batched(list(texts), batch_size):
        text_tokens = tokenizer(list(batch_texts)).to(device)
        text_features = model.encode_text(text_tokens)
        features.append(F.normalize(text_features, dim=-1).cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def encode_images(
    model, image_dataset: RetrievalEvalDataset, device: torch.device, batch_size: int
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for start in range(0, len(image_dataset), batch_size):
        batch_images = [
            image_dataset[idx]["image"]
            for idx in range(start, min(start + batch_size, len(image_dataset)))
        ]
        pixel_values = torch.stack(batch_images, dim=0).to(device)
        image_features = model.encode_image(pixel_values)
        features.append(F.normalize(image_features, dim=-1).cpu())
    return torch.cat(features, dim=0)


def evaluate_image_to_text_recall_at_k(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    positive_text_indices: List[List[int]],
    k: int = 10,
) -> float:
    recalls: List[float] = []
    similarity = image_embeds @ text_embeds.T
    for idx, gt_indices in enumerate(positive_text_indices):
        scores = similarity[idx].tolist()
        recalls.append(calculate_recall_at_k(scores, gt_indices, k=k))
    return float(sum(recalls) / max(len(recalls), 1))


def run_clip_retrieval_eval(
    root_dir: str | Path,
    image_to_text_csv: str | Path,
    textnums_to_texts_csv: str | Path,
    model_name: str = "ViT-B-32",
    batch_size: int = 32,
    k: int = 10,
    device: str | None = None,
) -> Dict[str, float]:
    root_dir = Path(root_dir)
    device_obj = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, _, tokenizer = _load_clip(model_name, device_obj)

    image_dataset = RetrievalEvalDataset(
        root_dir=root_dir,
        image_to_text_csv=image_to_text_csv,
        textnums_to_texts_csv=textnums_to_texts_csv,
        mode="image",
    )
    eval_data = image_dataset.get_image_to_text_eval_data()

    image_embeds = encode_images(
        model, image_dataset, device_obj, batch_size=batch_size
    )
    text_embeds = encode_texts(
        model, tokenizer, eval_data.texts, device_obj, batch_size=batch_size
    )

    image_to_text_recall = evaluate_image_to_text_recall_at_k(
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        positive_text_indices=eval_data.positive_text_indices,
        k=k,
    )
    return {f"image_to_text_recall@{k}": image_to_text_recall}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP image-text retrieval evaluation with Recall@10"
    )
    parser.add_argument("--root-dir", type=str, default="./sample_data")
    parser.add_argument(
        "--image-to-text-csv", type=str, default="./sample_data/img_list.csv"
    )
    parser.add_argument(
        "--textnums-to-texts-csv",
        type=str,
        default="./sample_data/txt_list.csv",
    )
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_clip_retrieval_eval(
        root_dir=args.root_dir,
        image_to_text_csv=args.image_to_text_csv,
        textnums_to_texts_csv=args.textnums_to_texts_csv,
        model_name=args.model_name,
        batch_size=args.batch_size,
        k=args.k,
        device=args.device,
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import open_clip
import onnxruntime as ort
import torch
import torch.nn.functional as F

from pipeline.dataset import RetrievalEvalDataset
from utils.clip_utils import _load_clip
from utils.data_utils import _batched, recall_at_k


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
        tokens = tokenizer(list(batch_texts)).numpy()
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
        text_tokens = tokenizer(list(batch_texts)).to(device)
        text_features = model.encode_text(text_tokens)
        features.append(F.normalize(text_features, dim=-1).cpu())
    return torch.cat(features, dim=0)


@torch.no_grad()
def _encode_images_torch(
    model, image_dataset: RetrievalEvalDataset, device: torch.device, batch_size: int
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for _, batch_indices in _batched(list(range(len(image_dataset))), batch_size):
        batch_images = [image_dataset[idx]["image"] for idx in batch_indices]
        pixel_values = torch.stack(batch_images, dim=0).to(device)
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
        img_sess = ort.InferenceSession(str(onnx_dir / "image_encoder.onnx"), providers=providers)
        txt_sess = ort.InferenceSession(str(onnx_dir / "text_encoder.onnx"), providers=providers)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # QAI Hub input spec requires batch=1
        image_embeds = _encode_images_onnx(img_sess, image_dataset, batch_size=1)
        text_embeds = _encode_texts_onnx(txt_sess, tokenizer, eval_data.texts, batch_size=1)
    else:
        model, _, tokenizer = _load_clip(model_name, device_obj, checkpoint_path=checkpoint_path)
        if print_model:
            print(model)
        image_embeds = _encode_images_torch(model, image_dataset, device_obj, batch_size)
        text_embeds = _encode_texts_torch(model, tokenizer, eval_data.texts, device_obj, batch_size)

    image_to_text_recall = recall_at_k(
        image_embeds.numpy(),
        text_embeds.numpy(),
        eval_data.positive_text_indices,
        k=k,
    )
    return {f"image_to_text_recall@{k}": image_to_text_recall}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP image-text retrieval evaluation (torch)"
    )
    parser.add_argument("--root-dir", type=str, default="./sample_data")
    parser.add_argument("--image-to-text-csv", type=str, default="./sample_data/img_list.csv")
    parser.add_argument("--textnums-to-texts-csv", type=str, default="./sample_data/txt_list.csv")
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--onnx-dir", type=str, default=None,
        help="Path to exported ONNX dir (e.g. exported_MobileCLIP2-S0_onnx). "
             "If set, uses ONNX inference instead of PyTorch.",
    )
    parser.add_argument(
        "--print-model", action="store_true",
        help="Print model architecture after loading (torch mode only).",
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
        checkpoint_path=args.checkpoint_path,
        onnx_dir=args.onnx_dir,
        print_model=args.print_model,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

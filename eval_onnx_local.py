from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
import open_clip
import onnxruntime as ort

from dataset import RetrievalEvalDataset
from utils import calculate_recall_at_k

import zipfile
import os
import torch
import onnxruntime as ort
import open_clip


def _extract_and_get_onnx_path(zip_path: str, extract_dir: str) -> str:
    """
    安全解压包含 model.data 的 QAI Hub zip 包，并返回 model.onnx 的路径
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"找不到压缩包: {zip_path}")

    # 如果还没解压，则进行解压
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        print(f"Extracting {zip_path} to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    # 遍历解压后的目录，寻找 .onnx 文件 (通常在 job_xxx_qdq_onnx 文件夹下)
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".onnx"):
                # 找到完整的 onnx 路径
                # 只要 model.data 在同一个 root 目录下，onnxruntime 就能自动找到它
                return os.path.join(root, file)

    raise FileNotFoundError(f"在解压目录 {extract_dir} 中没有找到 .onnx 文件")


def _load_onnx_sessions(
    model_name: str,
    device: torch.device,
    postfix: str = "",
):
    ONNX_DIR = f"exported_{model_name}_onnx"

    if postfix == "w8a8":
        # ZIP 压缩包路径
        image_zip_path = os.path.join(ONNX_DIR, "image_encoder_qai_int8.onnx.onnx.zip")
        text_zip_path = os.path.join(ONNX_DIR, "text_encoder_qai_int8.onnx.onnx.zip")

        # 为防止文件冲突，为 image 和 text 分别建立独立的解压目录
        image_extract_dir = os.path.join(ONNX_DIR, "image_qai_extracted")
        text_extract_dir = os.path.join(ONNX_DIR, "text_qai_extracted")

        # 获取真实包含外挂权重的 onnx 路径
        image_onnx_path = _extract_and_get_onnx_path(image_zip_path, image_extract_dir)
        text_onnx_path = _extract_and_get_onnx_path(text_zip_path, text_extract_dir)
    else:
        # 浮点模型直接读取
        image_onnx_path = os.path.join(ONNX_DIR, f"image_encoder{postfix}.onnx")
        text_onnx_path = os.path.join(ONNX_DIR, f"text_encoder{postfix}.onnx")

    providers = ["CPUExecutionProvider"]

    print(f"Loading Image Session from: {image_onnx_path}")
    image_session = ort.InferenceSession(image_onnx_path, providers=providers)

    print(f"Loading Text Session from: {text_onnx_path}")
    text_session = ort.InferenceSession(text_onnx_path, providers=providers)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return image_session, text_session, None, tokenizer


def _batched(iterable: Sequence, batch_size: int):
    for start in range(0, len(iterable), batch_size):
        yield start, iterable[start : start + batch_size]


def encode_texts(
    text_session: ort.InferenceSession, tokenizer, texts: Sequence[str], batch_size: int
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for _, batch_texts in _batched(list(texts), batch_size):
        text_tokens = tokenizer(list(batch_texts)).numpy()
        text_features_np = text_session.run(None, {"text": text_tokens})[0]
        text_features = torch.from_numpy(text_features_np)
        features.append(F.normalize(text_features, dim=-1).cpu())
    return torch.cat(features, dim=0)


def encode_images(
    image_session: ort.InferenceSession,
    image_dataset: RetrievalEvalDataset,
    batch_size: int,
) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for start in range(0, len(image_dataset), batch_size):
        batch_images = [
            image_dataset[idx]["image"]
            for idx in range(start, min(start + batch_size, len(image_dataset)))
        ]
        import pdb

        pdb.set_trace()
        pixel_values = torch.stack(batch_images, dim=0).numpy()
        image_features_np = image_session.run(None, {"image": pixel_values})[0]
        image_features = torch.from_numpy(image_features_np)
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
    model_name: str = "MobileCLIP2-S0",
    postfix: str = "",
    batch_size: int = 32,
    k: int = 10,
    device: str | None = None,
) -> Dict[str, float]:
    root_dir = Path(root_dir)
    device_obj = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    image_session, text_session, _, tokenizer = _load_onnx_sessions(
        model_name, device_obj, postfix
    )

    image_dataset = RetrievalEvalDataset(
        root_dir=root_dir,
        image_to_text_csv=image_to_text_csv,
        textnums_to_texts_csv=textnums_to_texts_csv,
        mode="image",
    )
    eval_data = image_dataset.get_image_to_text_eval_data()

    image_embeds = encode_images(image_session, image_dataset, batch_size=batch_size)
    text_embeds = encode_texts(
        text_session, tokenizer, eval_data.texts, batch_size=batch_size
    )

    image_to_text_recall = evaluate_image_to_text_recall_at_k(
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        positive_text_indices=eval_data.positive_text_indices,
        k=k,
    )
    return {f"image_to_text_recall@{k}": image_to_text_recall}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FP32 ONNX CLIP evaluation")
    parser.add_argument("--root-dir", type=str, default="./sample_data")
    parser.add_argument(
        "--image-to-text-csv", type=str, default="./sample_data/img_list.csv"
    )
    parser.add_argument(
        "--textnums-to-texts-csv", type=str, default="./sample_data/txt_list.csv"
    )
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_clip_retrieval_eval(
        root_dir=args.root_dir,
        image_to_text_csv=args.image_to_text_csv,
        textnums_to_texts_csv=args.textnums_to_texts_csv,
        model_name=args.model_name,
        postfix=args.postfix,
        batch_size=args.batch_size,
        k=args.k,
        device=args.device,
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()

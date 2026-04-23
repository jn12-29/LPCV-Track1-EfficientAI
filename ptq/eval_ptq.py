"""Evaluate quantized (or FP32) ONNX image/text encoders on a validation split.

Computes image-to-text Recall@K using ONNX Runtime inference on the held-out
validation set produced by ``ptq.dataset.split_calib_val``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import open_clip
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ptq.dataset import load_jsonl_records, split_calib_val
from utils.preprocess import preprocess_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PTQ ONNX encoders")
    p.add_argument("--onnx-dir", type=str, required=True)
    p.add_argument("--image-postfix", type=str, default="_ptq_u8s8")
    p.add_argument("--text-postfix", type=str, default="_ptq_u8s8")
    p.add_argument("--jsonl-path", type=str, required=True)
    p.add_argument("--image-base-dir", type=str, required=True)
    p.add_argument("--calib-size", type=int, default=1000)
    p.add_argument("--val-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model-name", type=str, default="MobileCLIP2-B")
    return p.parse_args()


def _ort_session(onnx_path: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)


def encode_images(
    session: ort.InferenceSession, records: List[Dict], batch_size: int
) -> np.ndarray:
    embeddings = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        tensors = []
        for rec in chunk:
            img = Image.open(rec["_resolved_path"]).convert("RGB")
            tensors.append(preprocess_image(img).numpy().astype(np.float32))
        batched = np.stack(tensors, axis=0)
        out = session.run(None, {"image": batched})[0]
        embeddings.append(out)
    return np.concatenate(embeddings, axis=0)


def encode_texts(
    session: ort.InferenceSession, texts: List[str], tokenizer, batch_size: int
) -> np.ndarray:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        tok = tokenizer(chunk).numpy().astype(np.int64)
        out = session.run(None, {"text": tok})[0]
        embeddings.append(out)
    return np.concatenate(embeddings, axis=0)


def recall_at_k(
    img_embeds: np.ndarray,
    txt_embeds: np.ndarray,
    positive_indices: List[List[int]],
    k: int,
) -> float:
    img_embeds = img_embeds / (np.linalg.norm(img_embeds, axis=1, keepdims=True) + 1e-8)
    txt_embeds = txt_embeds / (np.linalg.norm(txt_embeds, axis=1, keepdims=True) + 1e-8)
    sim = img_embeds @ txt_embeds.T
    recalls = []
    for i, gt in enumerate(positive_indices):
        if not gt:
            continue
        top_k = set(np.argsort(-sim[i])[:k].tolist())
        recalls.append(len(top_k & set(gt)) / len(gt))
    return float(np.mean(recalls)) if recalls else 0.0


def main() -> None:
    args = parse_args()
    onnx_dir = args.onnx_dir

    image_onnx = str(Path(onnx_dir) / f"image_encoder{args.image_postfix}.onnx")
    text_onnx = str(Path(onnx_dir) / f"text_encoder{args.text_postfix}.onnx")

    records = load_jsonl_records(args.jsonl_path, args.image_base_dir)
    _, val_records = split_calib_val(
        records, calib_size=args.calib_size, val_size=args.val_size, seed=args.seed,
    )
    print(f"Validation records: {len(val_records)}")

    # Build text corpus and per-image positive indices
    all_texts: List[str] = []
    text_to_idx: Dict[str, int] = {}
    positive_indices: List[List[int]] = []
    for rec in val_records:
        pos_texts = rec.get("positives", [])
        indices = []
        for t in pos_texts:
            if t not in text_to_idx:
                text_to_idx[t] = len(all_texts)
                all_texts.append(t)
            indices.append(text_to_idx[t])
        positive_indices.append(indices)

    print(f"Unique texts: {len(all_texts)}")

    img_sess = _ort_session(image_onnx)
    txt_sess = _ort_session(text_onnx)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    print("Encoding images …")
    img_embeds = encode_images(img_sess, val_records, batch_size=args.batch_size)
    print("Encoding texts …")
    txt_embeds = encode_texts(
        txt_sess, all_texts, tokenizer, batch_size=args.batch_size
    )

    r = recall_at_k(img_embeds, txt_embeds, positive_indices, k=args.k)
    print(f"image_to_text_recall@{args.k}: {r:.4f}")


if __name__ == "__main__":
    main()

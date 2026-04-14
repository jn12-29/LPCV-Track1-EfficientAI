from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _batched(iterable: Sequence, batch_size: int):
    """Yield (start_index, batch_slice) pairs."""
    for start in range(0, len(iterable), batch_size):
        yield start, iterable[start : start + batch_size]


def recall_at_k(
    img_embeds: np.ndarray,
    txt_embeds: np.ndarray,
    positive_indices: List[List[int]],
    k: int = 10,
) -> float:
    """Compute mean Recall@K over all image queries.

    Embeddings are L2-normalised internally so raw or pre-normalised arrays
    both work correctly.
    """
    img_embeds = img_embeds / (np.linalg.norm(img_embeds, axis=1, keepdims=True) + 1e-8)
    txt_embeds = txt_embeds / (np.linalg.norm(txt_embeds, axis=1, keepdims=True) + 1e-8)
    sim = cosine_similarity(img_embeds, txt_embeds)
    recalls = []
    for i, gt_idx in enumerate(positive_indices):
        if not gt_idx:
            continue
        top_k = set(np.argsort(-sim[i])[:k].tolist())
        matched = len(top_k & set(gt_idx))
        recalls.append(matched / len(gt_idx))
    return float(np.mean(recalls))


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_image_to_textnums(csv_path: Path) -> Dict[str, List[int]]:
    """Load image_name → [text_num, ...] mapping from img_list.csv."""
    mapping: Dict[str, List[int]] = {}
    for row in _read_csv_rows(csv_path):
        image_name = row["Image_names"].strip()
        text_nums = [int(x) for x in row["Text_nums"].split(";") if x.strip()]
        mapping[image_name] = text_nums
    return mapping


def load_textnums_to_texts(csv_path: Path) -> Dict[int, str]:
    """Load text_num → text_string mapping from txt_list.csv."""
    mapping: Dict[int, str] = {}
    for row in _read_csv_rows(csv_path):
        text_num = int(row["Text_nums"])
        mapping[text_num] = row["Unique_Texts"].strip()
    return mapping


def load_ground_truth(
    img_csv: Path = Path("./sample_data/img_list.csv"),
    txt_csv: Path = Path("./sample_data/txt_list.csv"),
) -> Tuple[List[str], List[List[int]]]:
    """Load image names and their positive text indices.

    Returns:
        image_names: ordered list of image filenames.
        positive_indices: for each image, indices into the text list.
    """
    txt_nums: List[int] = []
    with open(txt_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            txt_nums.append(int(row["Text_nums"]))
    txt_num_to_idx = {n: i for i, n in enumerate(txt_nums)}

    image_names: List[str] = []
    positive_indices: List[List[int]] = []
    with open(img_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            image_names.append(row["Image_names"].strip())
            gt = [int(x) for x in row["Text_nums"].split(";") if x.strip()]
            positive_indices.append(
                [txt_num_to_idx[n] for n in gt if n in txt_num_to_idx]
            )
    return image_names, positive_indices

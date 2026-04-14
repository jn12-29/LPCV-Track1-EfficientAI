from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset

from utils.data_utils import load_image_to_textnums, load_textnums_to_texts
from utils.preprocess import preprocess_image


@dataclass(frozen=True)
class ImageTextSample:
    image_path: Path
    image_name: str
    text_num: int
    text: str


@dataclass(frozen=True)
class ImageToTextEvalData:
    texts: List[str]
    positive_text_indices: List[List[int]]


def build_image_text_samples(
    image_to_textnums: Dict[str, Sequence[int]],
    textnums_to_texts: Dict[int, str],
    image_dir: Path,
) -> List[ImageTextSample]:
    """Create one sample for every positive image-text pair."""
    samples: List[ImageTextSample] = []
    for image_name, text_nums in image_to_textnums.items():
        image_path = image_dir / image_name
        for text_num in text_nums:
            if text_num not in textnums_to_texts:
                continue
            samples.append(
                ImageTextSample(
                    image_path=image_path,
                    image_name=image_name,
                    text_num=text_num,
                    text=textnums_to_texts[text_num],
                )
            )
    return samples


class ImageTextRetrievalDataset(Dataset):
    """PyTorch dataset for image-text retrieval.

    Each item is a positive image-text pair.
    Returns a dict with: image (3,224,224), text, image_name, text_num, image_path.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_to_text_csv: str | Path,
        textnums_to_texts_csv: str | Path,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"

        image_to_textnums = load_image_to_textnums(Path(image_to_text_csv))
        textnums_to_texts = load_textnums_to_texts(Path(textnums_to_texts_csv))
        self.samples = build_image_text_samples(
            image_to_textnums=image_to_textnums,
            textnums_to_texts=textnums_to_texts,
            image_dir=self.image_dir,
        )
        self.image_to_textnums = image_to_textnums
        self.textnums_to_texts = textnums_to_texts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        return {
            "image": preprocess_image(image),
            "text": sample.text,
            "image_name": sample.image_name,
            "text_num": sample.text_num,
            "image_path": str(sample.image_path),
        }


class RetrievalEvalDataset(Dataset):
    """Evaluation dataset for one-side retrieval.

    mode='image': each item is one image with all its positive text ids.
    mode='text':  each item is one text with all its positive image names.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_to_text_csv: str | Path,
        textnums_to_texts_csv: str | Path,
        mode: str = "image",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.mode = mode

        self.image_to_textnums = load_image_to_textnums(Path(image_to_text_csv))
        self.textnums_to_texts = load_textnums_to_texts(Path(textnums_to_texts_csv))

        self.text_to_image_names: Dict[int, List[str]] = {}
        for image_name, text_nums in self.image_to_textnums.items():
            for text_num in text_nums:
                self.text_to_image_names.setdefault(text_num, []).append(image_name)

        if mode == "image":
            self.items = list(self.image_to_textnums.items())
        elif mode == "text":
            self.items = list(self.textnums_to_texts.items())
        else:
            raise ValueError("mode must be 'image' or 'text'")

    def __len__(self) -> int:
        return len(self.items)

    def get_image_to_text_eval_data(self) -> ImageToTextEvalData:
        """Build reusable inputs for Image-to-Text retrieval evaluation."""
        image_names = list(self.image_to_textnums.keys())
        text_nums = list(self.textnums_to_texts.keys())
        texts = [self.textnums_to_texts[tn] for tn in text_nums]
        text_num_to_index = {tn: idx for idx, tn in enumerate(text_nums)}

        positive_text_indices: List[List[int]] = []
        for image_name in image_names:
            gt_text_nums = self.image_to_textnums.get(image_name, [])
            positive_text_indices.append(
                [text_num_to_index[tn] for tn in gt_text_nums if tn in text_num_to_index]
            )
        return ImageToTextEvalData(texts=texts, positive_text_indices=positive_text_indices)

    def __getitem__(self, index: int) -> Dict:
        if self.mode == "image":
            image_name, text_nums = self.items[index]
            image_path = self.image_dir / image_name
            image = Image.open(image_path).convert("RGB")
            return {
                "image": preprocess_image(image),
                "image_name": image_name,
                "image_path": str(image_path),
                "positive_text_nums": list(text_nums),
                "positive_texts": [
                    self.textnums_to_texts[t]
                    for t in text_nums
                    if t in self.textnums_to_texts
                ],
            }
        text_num, text = self.items[index]
        return {
            "text": text,
            "text_num": text_num,
            "positive_image_names": self.text_to_image_names.get(text_num, []),
        }

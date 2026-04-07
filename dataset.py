from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
import csv
import numpy as np

from PIL import Image
import torch
from torch.utils.data import Dataset


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


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_image_to_textnums(csv_path: Path) -> Dict[str, List[int]]:
    """Load the image -> text number mapping CSV."""
    mapping: Dict[str, List[int]] = {}
    for row in _read_csv_rows(csv_path):
        image_name = row["Image_names"].strip()
        text_nums = [int(x) for x in row["Text_nums"].split(";") if x.strip()]
        mapping[image_name] = text_nums
    return mapping


def load_textnums_to_texts(csv_path: Path) -> Dict[int, str]:
    """Load the text number -> text string mapping CSV."""
    mapping: Dict[int, str] = {}
    for row in _read_csv_rows(csv_path):
        text_num = int(row["Text_nums"])
        mapping[text_num] = row["Unique_Texts"].strip()
    return mapping


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


def preprocess_image_imagenet_style(image: Image.Image) -> torch.Tensor:
    """Preprocess an image to a float32 tensor of shape (1, 3, 224, 224).

    This follows the requested sample-solution style:
        1. resize to 224x224
        2. divide pixel values by 255
        3. reshape from HxWxC to CxHxW
        4. add batch dimension

    No mean/std normalization is applied.
    """
    image = image.convert("RGB").resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))[np.newaxis, :]
    return torch.from_numpy(image_array)


class ImageTextRetrievalDataset(Dataset):
    """PyTorch dataset for image-text retrieval.

    Each item is a positive image-text pair.
    Returns a dict with:
      - image: transformed image
      - text: raw text string
      - image_name: image filename
      - text_num: text id
      - image_path: image file path
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_to_text_csv: str | Path,
        textnums_to_texts_csv: str | Path,
        image_transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.image_transform = image_transform

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

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.image_transform is not None:
            raise ValueError("image_transform is not supported")
            image = self.image_transform(image)
        else:
            image = preprocess_image_imagenet_style(image)

        return {
            "image": image,
            "text": sample.text,
            "image_name": sample.image_name,
            "text_num": sample.text_num,
            "image_path": str(sample.image_path),
        }


class RetrievalEvalDataset(Dataset):
    """Evaluation dataset for one-side retrieval.

    mode="image" returns each image and all its positive text ids.
    mode="text" returns each text and all its positive image names.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_to_text_csv: str | Path,
        textnums_to_texts_csv: str | Path,
        mode: str = "image",
        image_transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.image_transform = image_transform
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
            raise ValueError("mode must be either 'image' or 'text'")

    def __len__(self) -> int:
        return len(self.items)

    def get_image_to_text_eval_data(self) -> ImageToTextEvalData:
        """Build reusable inputs for Image-to-Text retrieval evaluation."""
        image_names = list(self.image_to_textnums.keys())

        text_nums = list(self.textnums_to_texts.keys())
        texts = [self.textnums_to_texts[text_num] for text_num in text_nums]
        text_num_to_index = {text_num: idx for idx, text_num in enumerate(text_nums)}

        positive_text_indices: List[List[int]] = []
        for image_name in image_names:
            gt_text_nums = self.image_to_textnums.get(image_name, [])
            positive_text_indices.append(
                [
                    text_num_to_index[text_num]
                    for text_num in gt_text_nums
                    if text_num in text_num_to_index
                ]
            )

        return ImageToTextEvalData(
            texts=texts,
            positive_text_indices=positive_text_indices,
        )

    def __getitem__(self, index: int):
        if self.mode == "image":
            image_name, text_nums = self.items[index]
            image_path = self.image_dir / image_name
            image = Image.open(image_path).convert("RGB")
            if self.image_transform is not None:
                raise ValueError("image_transform is not supported")
                image = self.image_transform(image)
            else:
                image = preprocess_image_imagenet_style(image).squeeze(0)
            return {
                "image": image,
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


def default_image_transform(image):
    """Optional minimal transform helper for quick tests.

    Kept dependency-free except for PIL and torch users can replace it with
    torchvision transforms.
    """
    return image

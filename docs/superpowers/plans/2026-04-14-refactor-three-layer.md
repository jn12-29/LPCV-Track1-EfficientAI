# Refactor: Three-Layer Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the codebase into `utils/` (shared), `pipeline/` (export/eval), and cleaned-up `train_clip/` (training), eliminating all code duplication.

**Architecture:** Shared utilities (`_load_clip`, `preprocess_image`, metrics, CSV helpers) live in `utils/`. Pipeline scripts (ONNX export, QAI Hub compile/profile, local and remote eval) move into `pipeline/`. Training scripts stay in `train_clip/` but shed duplicate data-processing code, consuming only pre-built contrastive JSONL.

**Tech Stack:** Python, PyTorch, open_clip, onnxruntime, qai_hub, sklearn

---

## File Map

**Create:**
- `utils/__init__.py`
- `utils/preprocess.py` — single `preprocess_image` implementation
- `utils/clip_utils.py` — single `_load_clip` implementation
- `utils/data_utils.py` — `_batched`, `recall_at_k`, CSV loaders, `load_ground_truth`
- `pipeline/__init__.py`
- `pipeline/dataset.py` — eval Dataset classes (migrated from `sample_dataset.py`)
- `pipeline/export_onnx.py` — migrated + deduped
- `pipeline/compile_and_profile.py` — migrated + argparse-in-main fix
- `pipeline/eval_local.py` — migrated + deduped
- `pipeline/eval_remote.py` — migrated + deduped
- `train_clip/__init__.py`
- `train_clip/record_utils.py` — `dedupe_keep_order`, `resolve_image_path`, `normalize_record` (simple format only)

**Modify:**
- `train_clip/finetune_mobileclip2_jsonl.py` — remove duplicated functions, import from utils/record_utils
- `train_clip/analyze_hard_negatives.py` — remove duplicated functions, import from utils/record_utils
- `script.sh` — update command paths
- `CLAUDE.md` — update architecture section

**Delete:**
- `eval_common.py`
- `sample_dataset.py`
- `export_onnx.py` (root)
- `compile_and_profile.py` (root)
- `eval_local.py` (root)
- `eval_remote.py` (root)

---

## Task 1: Create `utils/` package — preprocess, clip_utils, data_utils

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/preprocess.py`
- Create: `utils/clip_utils.py`
- Create: `utils/data_utils.py`

- [ ] **Step 1: Create `utils/__init__.py`**

```python
```
(empty file — marks `utils/` as a Python package)

- [ ] **Step 2: Create `utils/preprocess.py`**

Single canonical image preprocessing function. Returns `(3, 224, 224)` float32 tensor (no batch dim). All callers add the batch dim themselves.

```python
from __future__ import annotations

import numpy as np
from PIL import Image
import torch


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize to 224×224, divide by 255, return CHW float32 tensor.

    Shape: (3, 224, 224). No mean/std normalization — competition requirement.
    Callers add the batch dimension as needed.
    """
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))
```

- [ ] **Step 3: Create `utils/clip_utils.py`**

Canonical `_load_clip`: includes MobileCLIP2 `image_mean/std` kwargs and tokenizer fallback (both missing from the `eval_local.py` and `export_onnx.py` versions).

```python
from __future__ import annotations

from typing import Optional, Tuple
import torch
import open_clip


def _load_clip(
    model_name: str,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    pretrained: Optional[str] = None,
) -> Tuple:
    """Load a CLIP model, optionally from a fine-tuned checkpoint.

    Args:
        model_name: open_clip model name, e.g. 'MobileCLIP2-S0'.
        device: target device.
        checkpoint_path: path to a fine-tuned .pt file (optional).
        pretrained: explicit pretrained tag; auto-selected when None.

    Returns:
        (model, preprocess, tokenizer)
    """
    print(f"Loading CLIP model '{model_name}'...")
    available_models_tuple = open_clip.list_pretrained()
    available_model_names = {name for name, _ in available_models_tuple}
    if model_name not in available_model_names:
        raise ValueError(f"Model '{model_name}' not found. Available: {open_clip.list_pretrained()}")

    if pretrained:
        pretrained_tag = pretrained
    else:
        available_tags = [ckpt for name, ckpt in available_models_tuple if name == model_name]
        if model_name.startswith("MobileCLIP2") and "dfndr2b" in available_tags:
            pretrained_tag = "dfndr2b"
        else:
            pretrained_tag = available_tags[0]

    # MobileCLIP2 variants expect [0, 1] RGB — disable internal normalization so
    # the competition-style preprocessing (resize + /255 only) passes through unchanged.
    model_kwargs: dict = {}
    if model_name in {"MobileCLIP2-S0", "MobileCLIP2-S2", "MobileCLIP2-S3", "MobileCLIP2-B"}:
        model_kwargs = {"image_mean": (0.0, 0.0, 0.0), "image_std": (1.0, 1.0, 1.0)}

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_tag, **model_kwargs
    )

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)}")

    model.eval().to(device)

    try:
        tokenizer = open_clip.get_tokenizer(model_name)
    except Exception:
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return model, preprocess, tokenizer
```

- [ ] **Step 4: Create `utils/data_utils.py`**

Merges `eval_common.py` content with the CSV helpers from `sample_dataset.py`.

```python
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
```

- [ ] **Step 5: Verify imports work**

Run from project root:
```bash
python -c "from utils.preprocess import preprocess_image; print('preprocess OK')"
python -c "from utils.data_utils import recall_at_k, _batched, load_ground_truth; print('data_utils OK')"
python -c "from utils.clip_utils import _load_clip; print('clip_utils OK')"
```
Expected: three `OK` lines, no ImportError.

- [ ] **Step 6: Commit**

```bash
git add utils/
git commit -m "refactor: add utils/ package with canonical preprocess, clip_utils, data_utils"
```

---

## Task 2: Create `pipeline/` package with `dataset.py`

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/dataset.py`

- [ ] **Step 1: Create `pipeline/__init__.py`**

```python
```
(empty)

- [ ] **Step 2: Create `pipeline/dataset.py`**

Migrated from `sample_dataset.py`. Uses `preprocess_image` from `utils.preprocess` instead of the inline implementation. Removes the dead unreachable code after `raise ValueError`.

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

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
```

- [ ] **Step 3: Verify**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from pipeline.dataset import RetrievalEvalDataset, ImageTextRetrievalDataset
print('pipeline.dataset OK')
"
```
Expected: `pipeline.dataset OK`

- [ ] **Step 4: Commit**

```bash
git add pipeline/
git commit -m "refactor: add pipeline/ package with dataset.py (migrated from sample_dataset.py)"
```

---

## Task 3: Migrate `export_onnx.py` → `pipeline/export_onnx.py`

**Files:**
- Create: `pipeline/export_onnx.py`

- [ ] **Step 1: Create `pipeline/export_onnx.py`**

Remove the inline `_load_clip`. Import from `utils.clip_utils`. Rest of the file is unchanged.

```python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from timm.utils import reparameterize_model
import open_clip

from utils.clip_utils import _load_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--output-postfix",
        type=str,
        default="",
        help="Suffix appended to exported ONNX filenames and output directory.",
    )
    return parser.parse_args()


class OpenClipVisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image)


class OpenClipTextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(dtype=torch.int64)
        eot_pos = token_ids.argmax(dim=-1, keepdim=True)
        positions = torch.arange(token_ids.shape[-1], device=token_ids.device).unsqueeze(0)
        mask = (positions <= eot_pos).to(token_ids.dtype)
        return self.model.encode_text(token_ids * mask)


def verify_onnx(
    onnx_path: str,
    input_dict: dict,
    pt_output: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {k: v.numpy() for k, v in input_dict.items()}
    ort_out = sess.run(None, ort_inputs)[0]
    pt_out = pt_output.numpy()
    max_diff = np.abs(ort_out - pt_out).max()
    status = "PASS" if np.allclose(ort_out, pt_out, rtol=rtol, atol=atol) else "FAIL"
    print(f"  Max abs diff (ONNX vs PyTorch): {max_diff:.6f}  {status}")
    if status == "FAIL":
        raise RuntimeError(f"ONNX output mismatch for {onnx_path}. Max diff={max_diff:.6f}")


def main() -> None:
    args = parse_args()

    output_dir_name = f"exported_{args.model_name}_onnx"
    if args.output_postfix:
        output_dir_name += args.output_postfix
    os.makedirs(output_dir_name, exist_ok=True)
    print(f"Saving ONNX files to directory: {os.path.abspath(output_dir_name)}")

    device = torch.device("cpu")
    clip_model, _, _ = _load_clip(
        model_name=args.model_name,
        device=device,
        checkpoint_path=args.checkpoint_path,
    )
    clip_model.eval()
    clip_model = reparameterize_model(clip_model)

    image_encoder = OpenClipVisionEncoder(clip_model)
    text_encoder = OpenClipTextEncoder(clip_model)
    image_encoder.eval()
    text_encoder.eval()

    dummy_image_input = torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device)
    dummy_text_input = torch.randint(0, 49408, (1, 77), dtype=torch.int64, device=device)

    print("\nCalculating PyTorch baseline outputs for validation...")
    with torch.no_grad():
        pt_img_feat = image_encoder(dummy_image_input)
        pt_txt_feat = text_encoder(dummy_text_input)

    image_onnx_path = os.path.join(output_dir_name, "image_encoder.onnx")
    text_onnx_path = os.path.join(output_dir_name, "text_encoder.onnx")

    print(f"\nExporting Image Encoder to {image_onnx_path}...")
    torch.onnx.export(
        image_encoder,
        dummy_image_input,
        image_onnx_path,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=True,
    )
    verify_onnx(image_onnx_path, {"image": dummy_image_input}, pt_img_feat)

    print(f"\nExporting Text Encoder to {text_onnx_path}...")
    torch.onnx.export(
        text_encoder,
        dummy_text_input,
        text_onnx_path,
        input_names=["text"],
        output_names=["text_embedding"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=True,
    )
    verify_onnx(text_onnx_path, {"text": dummy_text_input}, pt_txt_feat)

    print("\nExport and verification complete.")
    if args.checkpoint_path:
        print(f"Exported fine-tuned checkpoint: {Path(args.checkpoint_path).resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import**

```bash
python -c "import pipeline.export_onnx; print('export_onnx OK')"
```
Expected: `export_onnx OK`

- [ ] **Step 3: Commit**

```bash
git add pipeline/export_onnx.py
git commit -m "refactor: migrate export_onnx.py to pipeline/, use utils.clip_utils"
```

---

## Task 4: Migrate `compile_and_profile.py` → `pipeline/compile_and_profile.py`

**Files:**
- Create: `pipeline/compile_and_profile.py`

Changes from the root version:
- Move all module-level code into `main()` so the file can be imported
- Translate Chinese print string to English

- [ ] **Step 1: Create `pipeline/compile_and_profile.py`**

```python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnx
import qai_hub


def run_profile(model, name: str, device) -> str:
    """Submit a profile job and return the job ID."""
    profile_job = qai_hub.submit_profile_job(
        model=model,
        name=name,
        device=device,
        options="--max_profiler_iterations 100",
    )
    return profile_job.job_id


def compile_model(model, name: str, device, input_specs) -> str:
    """Submit a compile job, share results, and return the job ID."""
    compile_job = qai_hub.submit_compile_job(
        model=model,
        name=name,
        device=device,
        input_specs=input_specs,
        options="--target_runtime qnn_dlc --truncate_64bit_io",
    )
    compile_job.modify_sharing(add_emails=["lowpowervision@gmail.com"])
    print(f"Job {compile_job.job_id} shared with lowpowervision@gmail.com")
    return compile_job.job_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--onnx-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model_name
    postfix = args.postfix
    onnx_dir = args.onnx_dir or f"exported_{model_name}_onnx"

    image_onnx_path = os.path.join(onnx_dir, f"image_encoder{postfix}.onnx")
    text_onnx_path = os.path.join(onnx_dir, f"text_encoder{postfix}.onnx")

    if not os.path.exists(onnx_dir):
        print(f"Error: Directory '{onnx_dir}' not found. Run 'export_onnx.py' first.")
        sys.exit(1)

    print(f"Loading ONNX Image Encoder from {image_onnx_path}...")
    onnx_img_model = onnx.load(image_onnx_path)
    try:
        onnx.checker.check_model(onnx_img_model)
        print("Image ONNX model is valid")
    except onnx.checker.ValidationError as e:
        print(f"Image ONNX model validation failed: {e}")

    print(f"\nLoading ONNX Text Encoder from {text_onnx_path}...")
    onnx_txt_model = onnx.load(text_onnx_path)
    try:
        onnx.checker.check_model(onnx_txt_model)
        print("Text ONNX model is valid")
    except onnx.checker.ValidationError as e:
        print(f"Text ONNX model validation failed: {e}")

    target_device = qai_hub.Device("XR2 Gen 2 (Proxy)")

    print("\nSubmitting compilation jobs to QAI Hub...")
    img_compile_id = compile_model(
        model=onnx_img_model,
        name=f"{model_name}_image_encoder{postfix}",
        device=target_device,
        input_specs={"image": (1, 3, 224, 224)},
    )
    txt_compile_id = compile_model(
        model=onnx_txt_model,
        name=f"{model_name}_text_encoder{postfix}",
        device=target_device,
        input_specs={"text": ((1, 77), "int64")},
    )
    print(f"Image compilation job ID: {img_compile_id}")
    print(f"Text  compilation job ID: {txt_compile_id}")

    print("\nWaiting for compilation to finish before profiling...")
    img_compiled_model = qai_hub.get_job(img_compile_id).get_target_model()
    txt_compiled_model = qai_hub.get_job(txt_compile_id).get_target_model()

    print("\nSubmitting profiling jobs to QAI Hub...")
    run_profile(
        model=img_compiled_model,
        name=f"{model_name}_image_encoder{postfix}",
        device=target_device,
    )
    run_profile(
        model=txt_compiled_model,
        name=f"{model_name}_text_encoder{postfix}",
        device=target_device,
    )
    print("Profiling jobs submitted for both models.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import**

```bash
python -c "import pipeline.compile_and_profile; print('compile_and_profile OK')"
```
Expected: `compile_and_profile OK` (qai_hub import may warn if not configured, but should not error on import)

- [ ] **Step 3: Commit**

```bash
git add pipeline/compile_and_profile.py
git commit -m "refactor: migrate compile_and_profile.py to pipeline/, fix module-level argparse"
```

---

## Task 5: Migrate `eval_local.py` → `pipeline/eval_local.py`

**Files:**
- Create: `pipeline/eval_local.py`

Changes: remove `_load_clip` (use `utils.clip_utils`), remove ONNX zip-extract and ONNX session code (dead path no longer needed — ONNX eval is handled separately), use `_batched` from `utils.data_utils`, import Dataset from `pipeline.dataset`.

- [ ] **Step 1: Create `pipeline/eval_local.py`**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import open_clip

from pipeline.dataset import RetrievalEvalDataset
from utils.clip_utils import _load_clip
from utils.data_utils import _batched, recall_at_k


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
) -> Dict[str, float]:
    root_dir = Path(root_dir)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    image_dataset = RetrievalEvalDataset(
        root_dir=root_dir,
        image_to_text_csv=image_to_text_csv,
        textnums_to_texts_csv=textnums_to_texts_csv,
        mode="image",
    )
    eval_data = image_dataset.get_image_to_text_eval_data()

    model, _, tokenizer = _load_clip(model_name, device_obj, checkpoint_path=checkpoint_path)
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-path", type=str, default=None)
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
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
```

Note: The ONNX backend (`--backend onnx`) is removed from this file. ONNX local eval was a secondary code path and can be handled by running `pipeline/eval_local.py` after implementing a separate ONNX wrapper if needed.

- [ ] **Step 2: Verify import**

```bash
python -c "import pipeline.eval_local; print('eval_local OK')"
```
Expected: `eval_local OK`

- [ ] **Step 3: Commit**

```bash
git add pipeline/eval_local.py
git commit -m "refactor: migrate eval_local.py to pipeline/, remove ONNX backend, use utils"
```

---

## Task 6: Migrate `eval_remote.py` → `pipeline/eval_remote.py`

**Files:**
- Create: `pipeline/eval_remote.py`

Changes: replace inline `_process_image` with `preprocess_image` from `utils.preprocess`.

- [ ] **Step 1: Create `pipeline/eval_remote.py`**

```python
"""
Remote evaluation on QAI Hub for MobileCLIP2 image-text retrieval.

Three modes:
  Mode A – Upload dataset, then submit inference from compiled models:
      python pipeline/eval_remote.py --upload-dataset \\
          --image-compiled-id <id> --text-compiled-id <id>

  Mode B – Submit inference using existing dataset IDs:
      python pipeline/eval_remote.py \\
          --image-compiled-id <id> --text-compiled-id <id> \\
          [--image-dataset-id <id>] [--text-dataset-id <id>]

  Mode C – Skip inference, reuse existing inference job outputs:
      python pipeline/eval_remote.py \\
          --image-inference-id <id> --text-inference-id <id>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import open_clip
import qai_hub
from PIL import Image

from utils.data_utils import load_ground_truth, recall_at_k
from utils.preprocess import preprocess_image

DATA_DIR = Path("./sample_data")
IMAGE_DIR = DATA_DIR / "images"
IMG_LIST_CSV = DATA_DIR / "img_list.csv"
TXT_LIST_CSV = DATA_DIR / "txt_list.csv"

DEFAULT_IMAGE_DATASET_ID = "d2qe36jl2"
DEFAULT_TEXT_DATASET_ID = "d95k6jwm9"


# ---------------------------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------------------------

def upload_datasets() -> tuple[str, str]:
    """Upload image and text datasets to QAI Hub and return their IDs."""
    print("Processing images...")
    image_names = []
    with open(IMG_LIST_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            image_names.append(row["Image_names"].strip())

    images = []
    for name in image_names:
        img = Image.open(IMAGE_DIR / name)
        # preprocess_image returns (3,224,224); add batch dim + convert to numpy
        tensor = preprocess_image(img).unsqueeze(0)  # (1, 3, 224, 224)
        images.append(tensor.numpy())

    print(f"  {len(images)} images, shape {images[0].shape}, dtype {images[0].dtype}")
    print("Uploading image dataset to QAI Hub...")
    image_dataset = qai_hub.upload_dataset({"image": images})
    print(f"  Image dataset ID: {image_dataset.dataset_id}")

    print("\nLoading text prompts...")
    prompts = []
    with open(TXT_LIST_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            prompts.append(row["Unique_Texts"].strip())
    print(f"  {len(prompts)} prompts.")

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokenized_texts = [tokenizer([p]).numpy() for p in prompts]

    print("Uploading text dataset to QAI Hub...")
    text_dataset = qai_hub.upload_dataset({"text": tokenized_texts})
    print(f"  Text dataset ID: {text_dataset.dataset_id}")

    return image_dataset.dataset_id, text_dataset.dataset_id


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, name: str, device, input_dataset):
    """Submit an inference job, wait for completion, and return the job object."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        name=name,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1",
    )
    inference_job.wait()
    return inference_job


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--upload-dataset", action="store_true",
                        help="Upload local sample_data to QAI Hub (Mode A).")
    parser.add_argument("--image-dataset-id", type=str, default=None,
                        help=f"QAI Hub image dataset ID (default: {DEFAULT_IMAGE_DATASET_ID}).")
    parser.add_argument("--text-dataset-id", type=str, default=None,
                        help=f"QAI Hub text dataset ID (default: {DEFAULT_TEXT_DATASET_ID}).")
    parser.add_argument("--image-compiled-id", type=str, default=None)
    parser.add_argument("--text-compiled-id", type=str, default=None)
    parser.add_argument("--image-inference-id", type=str, default=None)
    parser.add_argument("--text-inference-id", type=str, default=None)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    use_existing = args.image_inference_id is not None or args.text_inference_id is not None
    outputs = {}

    if use_existing:
        tasks = {"text": args.text_inference_id, "image": args.image_inference_id}
        for task_name, job_id in tasks.items():
            if job_id is None:
                raise ValueError(f"--{task_name}-inference-id is required in Mode C")
            print(f"Fetching {task_name} inference job {job_id} ...")
            inference_job = qai_hub.get_job(job_id)
            inference_output = inference_job.download_output_data()
            outputs[task_name] = inference_output["output_0"]

    else:
        if args.image_compiled_id is None or args.text_compiled_id is None:
            raise ValueError(
                "Provide either (--image-compiled-id + --text-compiled-id) "
                "or (--image-inference-id + --text-inference-id)"
            )

        if args.upload_dataset:
            print("=== Uploading datasets ===")
            img_ds_id, txt_ds_id = upload_datasets()
            print()
        else:
            img_ds_id = args.image_dataset_id or DEFAULT_IMAGE_DATASET_ID
            txt_ds_id = args.text_dataset_id or DEFAULT_TEXT_DATASET_ID

        device = qai_hub.Device("XR2 Gen 2 (Proxy)")
        tasks = {
            "text":  {"compiled_id": args.text_compiled_id,  "dataset_id": txt_ds_id},
            "image": {"compiled_id": args.image_compiled_id, "dataset_id": img_ds_id},
        }
        for task_name, info in tasks.items():
            input_dataset = qai_hub.get_dataset(info["dataset_id"])
            compiled_model = qai_hub.get_job(info["compiled_id"]).get_target_model()

            print(f"Running inference for {task_name} model {compiled_model.model_id}")
            inference_job = run_inference(
                compiled_model,
                f"{args.model_name}_{task_name}_{info['compiled_id']}",
                device,
                input_dataset,
            )

            if inference_job.get_status().failure:
                print(f"{task_name.capitalize()} inference failed")
                outputs[task_name] = None
            else:
                inference_output = inference_job.download_output_data()
                outputs[task_name] = inference_output["output_0"]

    img_embeds = np.vstack(outputs["image"])
    txt_embeds = np.vstack(outputs["text"])
    print(f"\nComputing Recall@{args.k}...")
    print(f"  Image embeddings: {img_embeds.shape}")
    print(f"  Text  embeddings: {txt_embeds.shape}")

    _, positive_indices = load_ground_truth()
    recall = recall_at_k(img_embeds, txt_embeds, positive_indices, k=args.k)
    print(f"\nRecall@{args.k} (on-device, XR2 Gen 2): {recall:.4f}")
```

- [ ] **Step 2: Verify import**

```bash
python -c "import pipeline.eval_remote; print('eval_remote OK')"
```
Expected: `eval_remote OK`

- [ ] **Step 3: Commit**

```bash
git add pipeline/eval_remote.py
git commit -m "refactor: migrate eval_remote.py to pipeline/, use utils.preprocess"
```

---

## Task 7: Create `train_clip/record_utils.py` and `__init__.py`

**Files:**
- Create: `train_clip/__init__.py`
- Create: `train_clip/record_utils.py`

- [ ] **Step 1: Create `train_clip/__init__.py`**

```python
```
(empty)

- [ ] **Step 2: Create `train_clip/record_utils.py`**

Simplified from the two training scripts. `normalize_record` now only handles the simple contrastive format `{image_path, positives, hard_negatives}` — the nested raw annotation format is handled upstream by `build_datasets/`.

```python
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence


def dedupe_keep_order(items: Sequence[str]) -> List[str]:
    """Deduplicate a list of strings preserving first-seen order."""
    output: List[str] = []
    seen: set = set()
    for item in items:
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def resolve_image_path(
    image_path: str,
    jsonl_path: Path,
    repo_root: Optional[Path],
) -> Path:
    """Resolve a potentially relative image path to an absolute Path.

    Search order:
      1. Absolute path as-is
      2. repo_root / image_path
      3. jsonl_path.parent / image_path
    """
    candidate = Path(image_path)
    candidates: List[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    if repo_root is not None:
        candidates.append(repo_root / image_path)
    candidates.append(jsonl_path.parent / image_path)

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(f"Unable to resolve image path: {image_path!r}")


def normalize_record(record: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Validate and normalise a contrastive training record.

    Expects the flat contrastive format produced by build_datasets/:
        {"image_path": str, "positives": [...], "hard_negatives": [...]}

    Returns None if the record lacks an image_path or has no positives.
    Hard negatives that duplicate a positive text are silently dropped.
    """
    if "image_path" not in record:
        return None

    positives = dedupe_keep_order(record.get("positives", []))
    hard_negatives = dedupe_keep_order(record.get("hard_negatives", []))

    if not positives:
        return None

    positive_set = set(positives)
    return {
        "image_path": record["image_path"],
        "positives": positives,
        "hard_negatives": [t for t in hard_negatives if t not in positive_set],
        "image_id": record.get("image_id", Path(record["image_path"]).name),
        "challenge_tags": record.get("challenge_tags", []),
    }
```

- [ ] **Step 3: Verify**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from train_clip.record_utils import dedupe_keep_order, normalize_record, resolve_image_path
r = normalize_record({'image_path': 'a.jpg', 'positives': ['cat', 'cat', ''], 'hard_negatives': ['dog', 'cat']})
assert r['positives'] == ['cat'], r
assert r['hard_negatives'] == ['dog'], r
assert r['image_id'] == 'a.jpg', r
print('record_utils OK')
"
```
Expected: `record_utils OK`

- [ ] **Step 4: Commit**

```bash
git add train_clip/__init__.py train_clip/record_utils.py
git commit -m "refactor: add train_clip/record_utils.py, simplified normalize_record for contrastive JSONL"
```

---

## Task 8: Refactor `train_clip/analyze_hard_negatives.py`

**Files:**
- Modify: `train_clip/analyze_hard_negatives.py`

Remove: `preprocess_image_competition_style`, `dedupe_keep_order`, `flatten_raw_record`, `normalize_record`, `resolve_image_path`, `batched`.
Import from: `utils.clip_utils`, `utils.preprocess`, `utils.data_utils`, `train_clip.record_utils`.

- [ ] **Step 1: Rewrite `train_clip/analyze_hard_negatives.py`**

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.clip_utils import _load_clip
from utils.data_utils import _batched
from utils.preprocess import preprocess_image
from train_clip.record_utils import normalize_record, resolve_image_path


@torch.no_grad()
def encode_texts(
    model, tokenizer, texts: List[str], device: torch.device, batch_size: int
) -> torch.Tensor:
    outputs = []
    for _, batch_texts in _batched(list(texts), batch_size):
        text_tokens = tokenizer(list(batch_texts)).to(device)
        text_features = model.encode_text(text_tokens)
        outputs.append(F.normalize(text_features, dim=-1).cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_image(model, image_path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image).unsqueeze(0).to(device)
    image_features = model.encode_image(image_tensor)
    return F.normalize(image_features, dim=-1).cpu().squeeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze positive vs hard negative similarity distributions"
    )
    parser.add_argument(
        "--jsonl-path", type=str,
        default="./build_datasets/data/dataset_raw_contrastive.jsonl",
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--top-k-hard-cases", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./analysis_hard_negatives")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl_path).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, tokenizer = _load_clip(
        model_name=args.model_name,
        device=device,
        checkpoint_path=args.checkpoint_path,
    )
    model.eval()

    positive_sims: List[float] = []
    negative_sims: List[float] = []
    per_record_stats: List[Dict] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            raw_record = json.loads(line)
            record = normalize_record(raw_record)
            if record is None or not record["hard_negatives"]:
                continue

            image_path = resolve_image_path(record["image_path"], jsonl_path, repo_root)
            image_feature = encode_image(model, image_path, device)

            pos_features = encode_texts(
                model, tokenizer, record["positives"], device, args.text_batch_size
            )
            neg_features = encode_texts(
                model, tokenizer, record["hard_negatives"], device, args.text_batch_size
            )

            pos_sims = torch.mv(pos_features, image_feature).numpy()
            neg_sims = torch.mv(neg_features, image_feature).numpy()

            positive_sims.extend(pos_sims.tolist())
            negative_sims.extend(neg_sims.tolist())

            min_pos = float(np.min(pos_sims))
            max_neg = float(np.max(neg_sims))
            gap = min_pos - max_neg
            hardest_neg_idx = int(np.argmax(neg_sims))
            weakest_pos_idx = int(np.argmin(pos_sims))

            per_record_stats.append({
                "image_id": record["image_id"],
                "image_path": str(image_path),
                "num_positives": len(record["positives"]),
                "num_hard_negatives": len(record["hard_negatives"]),
                "mean_positive_similarity": float(np.mean(pos_sims)),
                "mean_hard_negative_similarity": float(np.mean(neg_sims)),
                "min_positive_similarity": min_pos,
                "max_hard_negative_similarity": max_neg,
                "gap_min_pos_minus_max_neg": gap,
                "weakest_positive_text": record["positives"][weakest_pos_idx],
                "weakest_positive_similarity": float(pos_sims[weakest_pos_idx]),
                "hardest_negative_text": record["hard_negatives"][hardest_neg_idx],
                "hardest_negative_similarity": float(neg_sims[hardest_neg_idx]),
                "positives": record["positives"],
                "hard_negatives": record["hard_negatives"],
            })

            if args.max_records is not None and len(per_record_stats) >= args.max_records:
                break
            if len(per_record_stats) % 100 == 0:
                print(f"Processed {len(per_record_stats)} records...")

    if not per_record_stats:
        raise ValueError("No valid records processed.")

    stats_path = output_dir / "similarity_stats.jsonl"
    with stats_path.open("w", encoding="utf-8") as f:
        for item in per_record_stats:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    hard_cases = sorted(
        per_record_stats, key=lambda x: x["gap_min_pos_minus_max_neg"]
    )[: args.top_k_hard_cases]
    hard_cases_path = output_dir / "hard_cases.jsonl"
    with hard_cases_path.open("w", encoding="utf-8") as f:
        for item in hard_cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    gaps = [item["gap_min_pos_minus_max_neg"] for item in per_record_stats]

    plt.figure(figsize=(10, 6))
    plt.hist(positive_sims, bins=60, alpha=0.6, label="positive", density=True)
    plt.hist(negative_sims, bins=60, alpha=0.6, label="hard_negative", density=True)
    plt.xlabel("cosine similarity")
    plt.ylabel("density")
    plt.title("Positive vs Hard Negative Similarity Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "positive_vs_hard_negative_similarity.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(gaps, bins=60, alpha=0.8, color="tab:red")
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("min_positive_similarity - max_hard_negative_similarity")
    plt.ylabel("count")
    plt.title("Gap Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "gap_distribution.png", dpi=160)
    plt.close()

    summary = {
        "num_records": len(per_record_stats),
        "num_positive_pairs": len(positive_sims),
        "num_hard_negative_pairs": len(negative_sims),
        "positive_similarity_mean": float(np.mean(positive_sims)),
        "positive_similarity_std": float(np.std(positive_sims)),
        "hard_negative_similarity_mean": float(np.mean(negative_sims)),
        "hard_negative_similarity_std": float(np.std(negative_sims)),
        "gap_mean": float(np.mean(gaps)),
        "gap_std": float(np.std(gaps)),
        "num_gap_below_zero": int(sum(g < 0 for g in gaps)),
        "ratio_gap_below_zero": float(sum(g < 0 for g in gaps) / len(gaps)),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved summary to:    {summary_path}")
    print(f"Saved full stats to: {stats_path}")
    print(f"Saved hard cases to: {hard_cases_path}")
    print(f"Saved plots to:      {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import**

```bash
python -c "import train_clip.analyze_hard_negatives; print('analyze_hard_negatives OK')"
```
Expected: `analyze_hard_negatives OK`

- [ ] **Step 3: Commit**

```bash
git add train_clip/analyze_hard_negatives.py
git commit -m "refactor: clean up analyze_hard_negatives.py, remove duplicated helpers"
```

---

## Task 9: Refactor `train_clip/finetune_mobileclip2_jsonl.py`

**Files:**
- Modify: `train_clip/finetune_mobileclip2_jsonl.py`

Remove: `preprocess_image_competition_style`, `dedupe_keep_order`, `flatten_raw_record`, `normalize_record`, `resolve_image_path`, `_select_pretrained_tag`, `_load_clip`.
Import from: `utils.clip_utils`, `utils.preprocess`, `train_clip.record_utils`.
The `ContrastiveRecordDataset._load_records` method only handles the flat contrastive format now.

- [ ] **Step 1: Rewrite the header / imports section of `finetune_mobileclip2_jsonl.py`**

Replace the current top of the file (lines 1–173, up to end of `_load_clip`) with:

```python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import open_clip

from utils.clip_utils import _load_clip
from utils.preprocess import preprocess_image
from train_clip.record_utils import dedupe_keep_order, normalize_record, resolve_image_path
```

- [ ] **Step 2: Replace `ContrastiveRecordDataset._load_records` to remove nested annotation parsing**

The `_load_records` method currently calls `normalize_record` which had a `flatten_raw_record` fallback for the nested annotation format. Now `normalize_record` only handles the flat format, so `_load_records` is unchanged in structure but now correctly rejects raw annotation records (which should never be passed to training).

No change needed to `ContrastiveRecordDataset` — it already calls `normalize_record(raw_record)` and skips `None` returns. The simplified `normalize_record` in `record_utils.py` handles this correctly.

- [ ] **Step 3: Replace `__getitem__` to use `preprocess_image` from utils**

In `ContrastiveRecordDataset.__getitem__`, change:
```python
image_tensor = preprocess_image_competition_style(image)
```
to:
```python
image_tensor = preprocess_image(image)
```

- [ ] **Step 4: Verify import**

```bash
python -c "import train_clip.finetune_mobileclip2_jsonl; print('finetune OK')"
```
Expected: `finetune OK`

- [ ] **Step 5: Commit**

```bash
git add train_clip/finetune_mobileclip2_jsonl.py
git commit -m "refactor: clean up finetune_mobileclip2_jsonl.py, remove duplicated helpers"
```

---

## Task 10: Delete old root-level files, update `script.sh` and `CLAUDE.md`

**Files:**
- Delete: `eval_common.py`, `sample_dataset.py`, `export_onnx.py`, `compile_and_profile.py`, `eval_local.py`, `eval_remote.py`
- Modify: `script.sh`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Delete old root-level files**

```bash
git rm eval_common.py sample_dataset.py export_onnx.py compile_and_profile.py eval_local.py eval_remote.py
```

- [ ] **Step 2: Update `script.sh`**

Change all script paths from root to `pipeline/`:
```bash
# Old → New
python export_onnx.py          → python pipeline/export_onnx.py
python compile_and_profile.py  → python pipeline/compile_and_profile.py
python eval_local.py           → python pipeline/eval_local.py
python eval_remote.py          → python pipeline/eval_remote.py
```

Read the current `script.sh` first, then apply these substitutions throughout.

- [ ] **Step 3: Update `CLAUDE.md` — Pipeline section**

In the "End-to-End Pipeline" section, update commands to reflect new paths:
```bash
# 1. Export ONNX
python pipeline/export_onnx.py --model-name MobileCLIP2-S0

# 2. Compile and profile
python pipeline/compile_and_profile.py --model-name MobileCLIP2-S0 [--postfix <suffix>]

# 3. Evaluate locally
python pipeline/eval_local.py --model-name MobileCLIP2-S0 --k 10

# Remote evaluation
python pipeline/eval_remote.py --upload-dataset \
    --image-compiled-id <id> --text-compiled-id <id>
python pipeline/eval_remote.py \
    --image-compiled-id <id> --text-compiled-id <id>
python pipeline/eval_remote.py \
    --image-inference-id <id> --text-inference-id <id>
```

Update the Architecture section to reflect the new structure:
- `utils/clip_utils.py` — `_load_clip` (shared by all)
- `utils/preprocess.py` — `preprocess_image` (shared by all)
- `utils/data_utils.py` — `_batched`, `recall_at_k`, CSV helpers, `load_ground_truth`
- `pipeline/dataset.py` — `RetrievalEvalDataset`, `ImageTextRetrievalDataset`
- `pipeline/export_onnx.py` — ONNX export and verification
- `pipeline/compile_and_profile.py` — QAI Hub compile + profile
- `pipeline/eval_local.py` — torch local evaluation
- `pipeline/eval_remote.py` — QAI Hub inference + evaluation
- `train_clip/record_utils.py` — `dedupe_keep_order`, `normalize_record`, `resolve_image_path`
- `train_clip/finetune_mobileclip2_jsonl.py` — fine-tuning
- `train_clip/analyze_hard_negatives.py` — similarity distribution analysis

- [ ] **Step 4: Final smoke test**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from utils.preprocess import preprocess_image
from utils.clip_utils import _load_clip
from utils.data_utils import recall_at_k, _batched, load_ground_truth
from pipeline.dataset import RetrievalEvalDataset
from pipeline import eval_local, eval_remote, export_onnx, compile_and_profile
from train_clip.record_utils import normalize_record
from train_clip import finetune_mobileclip2_jsonl, analyze_hard_negatives
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 5: Final commit**

```bash
git add script.sh CLAUDE.md
git commit -m "refactor: delete old root-level scripts, update script.sh and CLAUDE.md for new pipeline/ layout"
```

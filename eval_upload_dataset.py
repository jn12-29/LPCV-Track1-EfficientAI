"""
Upload dataset for MobileCLIP2-S0 inference on QAI Hub.

比赛统一预处理（与 CLIP sample solution 一致）：
  - Resize to 224x224, divide by 255 → float32 [0,1]
  - Shape: (1, 3, 224, 224)
  - 文本：CLIPTokenizer openai/clip-vit-base-patch32，padding=max_length，int32

Usage:
    python upload_dataset_mobileclip2.py
"""

import csv
from pathlib import Path

import numpy as np
import qai_hub
import torch
from PIL import Image
from transformers import CLIPTokenizer

# --- Configuration ---
DATA_DIR = Path("./sample_data")
IMAGE_DIR = DATA_DIR / "images"
IMG_LIST_CSV = DATA_DIR / "img_list.csv"
TXT_LIST_CSV = DATA_DIR / "txt_list.csv"
IMAGE_SIZE = 224  # 比赛统一输入尺寸
TEXT_CTX = 77
# ---------------------


def process_image(image_path):
    """比赛预处理：resize 224x224，除以 255，(1,3,H,W) float32。"""
    img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]
    return np.transpose(arr, (2, 0, 1))[np.newaxis, :]  # (1,3,224,224)


# ---- Images（按 img_list.csv 行顺序）----
print("Processing images...")
image_names = []
with open(IMG_LIST_CSV, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        image_names.append(row["Image_names"].strip())

images = []
for name in image_names:
    images.append(process_image(IMAGE_DIR / name))

print(
    f"Processed {len(images)} images, shape: {images[0].shape}, dtype: {images[0].dtype}"
)

print("Uploading image dataset to QAI Hub...")
image_dataset = qai_hub.upload_dataset({"image": images})
print(f"Image dataset ID: {image_dataset.dataset_id}")

# ---- Texts（按 txt_list.csv 行顺序）----
print("\nLoading text prompts...")
prompts = []
with open(TXT_LIST_CSV, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        prompts.append(row["Unique_Texts"].strip())

print(f"Loaded {len(prompts)} prompts.")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
tokenized_texts = []
for prompt in prompts:
    tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )["input_ids"].to(
        torch.int32
    )  # (1, 77) int32, pad=49407
    tokenized_texts.append(tokens.numpy())

print(
    f"Example tokenized shape: {tokenized_texts[0].shape}, dtype: {tokenized_texts[0].dtype}"
)

print("Uploading text dataset to QAI Hub...")
text_dataset = qai_hub.upload_dataset({"text": tokenized_texts})
print(f"Text dataset ID: {text_dataset.dataset_id}")

print("\n=== Upload complete ===")
print(f"  Image dataset ID: {image_dataset.dataset_id}")
print(f"  Text  dataset ID: {text_dataset.dataset_id}")

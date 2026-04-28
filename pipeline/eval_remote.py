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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _upload_image_dataset() -> str:
    """Preprocess and upload the image dataset; return dataset ID."""
    print("Processing images...")
    image_names = []
    with open(IMG_LIST_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            image_names.append(row["Image_names"].strip())

    images = []
    for name in image_names:
        img = Image.open(IMAGE_DIR / name)
        tensor = preprocess_image(img).unsqueeze(0)  # (1, 3, 224, 224)
        images.append(tensor.numpy())

    print(f"  {len(images)} images, shape {images[0].shape}, dtype {images[0].dtype}")
    print("Uploading image dataset to QAI Hub...")
    image_dataset = qai_hub.upload_dataset({"image": images})
    print(f"  Image dataset ID: {image_dataset.dataset_id}")
    return image_dataset.dataset_id


def _upload_text_dataset() -> str:
    """Tokenize and upload the text dataset; return dataset ID."""
    print("Loading text prompts...")
    prompts = []
    with open(TXT_LIST_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            prompts.append(row["Unique_Texts"].strip())
    print(f"  {len(prompts)} prompts.")

    # tokenizer = open_clip.get_tokenizer("ViT-B-32")
    from transformers import CLIPTokenizer

    pretrained_tokenizer = "openai/clip-vit-base-patch32"

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_tokenizer, local_files_only=True
    )

    tokenizer.add_special_tokens({"cls_token": tokenizer.eos_token})

    tokenized_texts = [
        tokenizer(
            [p],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )["input_ids"]
        .numpy()
        .astype(np.int32)
        for p in prompts
    ]

    print("Uploading text dataset to QAI Hub...")
    text_dataset = qai_hub.upload_dataset({"text": tokenized_texts})
    print(f"  Text dataset ID: {text_dataset.dataset_id}")
    return text_dataset.dataset_id


def upload_datasets() -> tuple[str, str]:
    """Upload image and text datasets to QAI Hub in parallel; return their IDs."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        img_future = executor.submit(_upload_image_dataset)
        txt_future = executor.submit(_upload_text_dataset)
        img_ds_id = img_future.result()
        txt_ds_id = txt_future.result()
    return img_ds_id, txt_ds_id


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
    parser.add_argument(
        "--upload-dataset",
        action="store_true",
        help="Upload local sample_data to QAI Hub (Mode A).",
    )
    parser.add_argument(
        "--image-dataset-id",
        type=str,
        default=None,
        help=f"QAI Hub image dataset ID (default: {DEFAULT_IMAGE_DATASET_ID}).",
    )
    parser.add_argument(
        "--text-dataset-id",
        type=str,
        default=None,
        help=f"QAI Hub text dataset ID (default: {DEFAULT_TEXT_DATASET_ID}).",
    )
    parser.add_argument("--image-compiled-id", type=str, default=None)
    parser.add_argument("--text-compiled-id", type=str, default=None)
    parser.add_argument("--image-inference-id", type=str, default=None)
    parser.add_argument("--text-inference-id", type=str, default=None)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    use_existing = (
        args.image_inference_id is not None or args.text_inference_id is not None
    )
    outputs = {}

    if use_existing:
        tasks = {"text": args.text_inference_id, "image": args.image_inference_id}
        for task_name, job_id in tasks.items():
            if job_id is None:
                raise ValueError(f"--{task_name}-inference-id is required in Mode C")

        def _fetch_output(task_name: str, job_id: str):
            print(f"Fetching {task_name} inference job {job_id} ...")
            inference_job = qai_hub.get_job(job_id)
            inference_output = inference_job.download_output_data()
            return task_name, inference_output["output_0"]

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_fetch_output, task_name, job_id)
                for task_name, job_id in tasks.items()
            ]
            for future in as_completed(futures):
                task_name, result = future.result()
                outputs[task_name] = result

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
            "text": {"compiled_id": args.text_compiled_id, "dataset_id": txt_ds_id},
            "image": {"compiled_id": args.image_compiled_id, "dataset_id": img_ds_id},
        }

        def _run_task(task_name: str, info: dict):
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
                return task_name, None
            inference_output = inference_job.download_output_data()
            return task_name, inference_output["output_0"]

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_run_task, task_name, info)
                for task_name, info in tasks.items()
            ]
            for future in as_completed(futures):
                task_name, result = future.result()
                outputs[task_name] = result

    img_embeds = np.vstack(outputs["image"])
    txt_embeds = np.vstack(outputs["text"])
    print(f"\nComputing Recall@{args.k}...")
    print(f"  Image embeddings: {img_embeds.shape}")
    print(f"  Text  embeddings: {txt_embeds.shape}")

    _, positive_indices = load_ground_truth()
    recall = recall_at_k(img_embeds, txt_embeds, positive_indices, k=args.k)
    print(f"\nRecall@{args.k} (on-device, XR2 Gen 2): {recall:.4f}")

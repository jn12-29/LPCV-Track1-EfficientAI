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
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import open_clip
import qai_hub
from PIL import Image

from utils.data_utils import (
    load_ground_truth,
    load_sample_image_names,
    load_sample_text_prompts,
    recall_at_k,
)
from utils.preprocess import preprocess_image
from ptq.dataset import load_jsonl_records, split_calib_val

DATA_DIR = Path("./sample_data")
IMAGE_DIR = DATA_DIR / "images"
IMG_LIST_CSV = DATA_DIR / "img_list.csv"
TXT_LIST_CSV = DATA_DIR / "txt_list.csv"

DEFAULT_IMAGE_DATASET_ID = "d2qe36jl2"
DEFAULT_TEXT_DATASET_ID = "d95k6jwm9"
DEFAULT_TEXT_COMPILED_ID = "jpx7wd93g"


# ---------------------------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------------------------

def _upload_image_dataset(image_channel_last: bool = False) -> str:
    """Preprocess and upload the image dataset; return dataset ID."""
    print("Processing images...")
    image_names = load_sample_image_names(IMG_LIST_CSV)

    images = []
    for name in image_names:
        img = Image.open(IMAGE_DIR / name)
        tensor = preprocess_image(img).unsqueeze(0)  # (1, 3, 224, 224)
        arr = tensor.numpy()
        if image_channel_last:
            arr = np.transpose(arr, (0, 2, 3, 1))  # (1, 224, 224, 3)
        images.append(arr)

    print(f"  {len(images)} images, shape {images[0].shape}, dtype {images[0].dtype}")
    print("Uploading image dataset to QAI Hub...")
    image_dataset = qai_hub.upload_dataset({"image": images})
    print(f"  Image dataset ID: {image_dataset.dataset_id}")
    return image_dataset.dataset_id


def _upload_text_dataset(model_name: str, text_dtype: str) -> str:
    """Tokenize and upload the text dataset; return dataset ID."""
    print("Loading text prompts...")
    prompts = load_sample_text_prompts(TXT_LIST_CSV)
    print(f"  {len(prompts)} prompts.")

    tokenizer = open_clip.get_tokenizer(model_name)
    target_dtype = np.int32 if text_dtype == "int32" else np.int64
    tokens = tokenizer(prompts).numpy().astype(target_dtype)
    tokenized_texts = [tokens[i : i + 1] for i in range(tokens.shape[0])]

    print("Uploading text dataset to QAI Hub...")
    text_dataset = qai_hub.upload_dataset({"text": tokenized_texts})
    print(f"  Text dataset ID: {text_dataset.dataset_id}")
    return text_dataset.dataset_id


def upload_datasets(
    model_name: str, text_dtype: str, image_channel_last: bool = False
) -> tuple[str, str]:
    """Upload image and text datasets to QAI Hub in parallel; return their IDs."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        img_future = executor.submit(_upload_image_dataset, image_channel_last)
        txt_future = executor.submit(_upload_text_dataset, model_name, text_dtype)
        img_ds_id = img_future.result()
        txt_ds_id = txt_future.result()
    return img_ds_id, txt_ds_id


def build_ptq_valid_eval_data(
    *,
    jsonl_path: str,
    image_base_dir: str,
    calib_size: int,
    val_size: int,
    seed: int,
    model_name: str,
    text_dtype: str,
    image_channel_last: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[int]]]:
    records = load_jsonl_records(jsonl_path=jsonl_path, image_base_dir=image_base_dir)
    _, val_records = split_calib_val(
        records,
        calib_size=calib_size,
        val_size=val_size,
        seed=seed,
    )

    images: list[np.ndarray] = []
    for rec in val_records:
        img = Image.open(rec["_resolved_path"]).convert("RGB")
        tensor = preprocess_image(img).unsqueeze(0)
        arr = tensor.numpy()
        if image_channel_last:
            arr = np.transpose(arr, (0, 2, 3, 1))  # (1, 224, 224, 3)
        images.append(arr)

    all_texts: list[str] = []
    text_to_idx: dict[str, int] = {}
    positive_indices: list[list[int]] = []
    for rec in val_records:
        indices = []
        for txt in rec.get("positives", []):
            if txt not in text_to_idx:
                text_to_idx[txt] = len(all_texts)
                all_texts.append(txt)
            indices.append(text_to_idx[txt])
        positive_indices.append(indices)

    tokenizer = open_clip.get_tokenizer(model_name)
    target_dtype = np.int32 if text_dtype == "int32" else np.int64
    tokens = tokenizer(all_texts).numpy().astype(target_dtype)
    tokenized_texts = [tokens[i : i + 1] for i in range(tokens.shape[0])]
    return images, tokenized_texts, positive_indices


def upload_ptq_valid_datasets(
    *,
    jsonl_path: str,
    image_base_dir: str,
    calib_size: int,
    val_size: int,
    seed: int,
    model_name: str,
    text_dtype: str,
    image_channel_last: bool = False,
) -> tuple[str, str, list[list[int]]]:
    images, tokenized_texts, positive_indices = build_ptq_valid_eval_data(
        jsonl_path=jsonl_path,
        image_base_dir=image_base_dir,
        calib_size=calib_size,
        val_size=val_size,
        seed=seed,
        model_name=model_name,
        text_dtype=text_dtype,
        image_channel_last=image_channel_last,
    )
    image_dataset = qai_hub.upload_dataset({"image": images})
    text_dataset = qai_hub.upload_dataset({"text": tokenized_texts})
    print(f"  Image dataset ID: {image_dataset.dataset_id}")
    print(f"  Text dataset ID: {text_dataset.dataset_id}")
    return image_dataset.dataset_id, text_dataset.dataset_id, positive_indices


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
    parser.add_argument("--ids-file", type=str, default=None)
    parser.add_argument("--image-inference-id", type=str, default=None)
    parser.add_argument("--text-inference-id", type=str, default=None)
    parser.add_argument("--jsonl-path", type=str, default=None)
    parser.add_argument("--image-base-dir", type=str, default="./")
    parser.add_argument("--calib-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dataset-ids-file", type=str, default=None)
    parser.add_argument(
        "--text-dtype",
        type=str,
        default="int32",
        choices=["int32", "int64"],
        help="Token dtype for uploaded text dataset. Use int32 with --truncate_64bit_io compile.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--image-channel-last",
        action="store_true",
        help="If set, transpose image input from NCHW to NHWC before dataset upload.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    use_existing = args.image_inference_id is not None or args.text_inference_id is not None
    outputs = {}
    positive_indices = None

    if (args.image_compiled_id is None or args.text_compiled_id is None) and args.ids_file:
        try:
            with open(args.ids_file, "r", encoding="utf-8") as f:
                ids_payload = json.load(f)
            args.image_compiled_id = ids_payload.get("image_compile_id")
            args.text_compiled_id = ids_payload.get("text_compile_id")
        except Exception as e:
            print(f"Warning: failed to read ids file '{args.ids_file}': {e}")

    if args.text_compiled_id is None:
        args.text_compiled_id = DEFAULT_TEXT_COMPILED_ID

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
            if args.jsonl_path:
                img_ds_id, txt_ds_id, positive_indices = upload_ptq_valid_datasets(
                    jsonl_path=args.jsonl_path,
                    image_base_dir=args.image_base_dir,
                    calib_size=args.calib_size,
                    val_size=args.val_size,
                    seed=args.seed,
                    model_name=args.model_name,
                    text_dtype=args.text_dtype,
                    image_channel_last=args.image_channel_last,
                )
            else:
                img_ds_id, txt_ds_id = upload_datasets(
                    model_name=args.model_name,
                    text_dtype=args.text_dtype,
                    image_channel_last=args.image_channel_last,
                )
            print()
        else:
            img_ds_id = args.image_dataset_id or DEFAULT_IMAGE_DATASET_ID
            txt_ds_id = args.text_dataset_id or DEFAULT_TEXT_DATASET_ID
            if args.jsonl_path:
                _, _, positive_indices = build_ptq_valid_eval_data(
                    jsonl_path=args.jsonl_path,
                    image_base_dir=args.image_base_dir,
                    calib_size=args.calib_size,
                    val_size=args.val_size,
                    seed=args.seed,
                    model_name=args.model_name,
                    text_dtype=args.text_dtype,
                    image_channel_last=args.image_channel_last,
                )

        if args.save_dataset_ids_file:
            with open(args.save_dataset_ids_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"image_dataset_id": img_ds_id, "text_dataset_id": txt_ds_id},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        device = qai_hub.Device("XR2 Gen 2 (Proxy)")
        tasks = {
            "text":  {"compiled_id": args.text_compiled_id,  "dataset_id": txt_ds_id},
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

    if positive_indices is None:
        _, positive_indices = load_ground_truth()
    recall = recall_at_k(img_embeds, txt_embeds, positive_indices, k=args.k)
    print(f"\nRecall@{args.k} (on-device, XR2 Gen 2): {recall:.4f}")

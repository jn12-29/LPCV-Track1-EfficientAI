import qai_hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
from pathlib import Path
import argparse

from eval_remote import load_ground_truth, compute_recall_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP image-text retrieval evaluation with Recall@k"
    )
    parser.add_argument("--image-inference-id", type=str, default="jp27mzwx5")
    parser.add_argument("--text-inference-id", type=str, default="jgklyk8y5")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Define target device
    device = qai_hub.Device("XR2 Gen 2 (Proxy)")

    text_inference_id = args.text_inference_id
    image_inference_id = args.image_inference_id
    k = args.k

    # TODO: Define tasks with their corresponding compiled job IDs and dataset IDs
    tasks = {
        "text": text_inference_id,
        "image": image_inference_id,
    }

    # Dictionary to store outputs separately
    outputs = {}

    for task_name, info in tasks.items():
        inference_job = qai_hub.get_job(info)
        inference_output = inference_job.download_output_data()
        outputs[task_name] = inference_output["output_0"]

    text_output = outputs["text"]
    image_output = outputs["image"]

    # ---- Evaluate ----
    print(f"\nComputing Recall{k}...")
    img_embeds = np.vstack(image_output)  # (N, D)
    txt_embeds = np.vstack(text_output)  # (M, D)
    print(f"  Image embeddings: {img_embeds.shape}")
    print(f"  Text  embeddings: {txt_embeds.shape}")

    image_names, positive_indices = load_ground_truth()
    recall = compute_recall_at_k(img_embeds, txt_embeds, positive_indices, k=k)
    print(f"\nRecall@{k} (on-device, XR2 Gen 2): {recall:.4f}")

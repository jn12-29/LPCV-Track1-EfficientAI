import qai_hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
from pathlib import Path
import argparse

DATA_DIR = Path("./data")
IMG_LIST_CSV = DATA_DIR / "img_list.csv"
TXT_LIST_CSV = DATA_DIR / "txt_list.csv"


def run_inference(model, name, device, input_dataset):
    """Submits an inference job for the model and returns the output data."""
    inference_job = qai_hub.submit_inference_job(
        model=model,
        name=name,
        device=device,
        inputs=input_dataset,
        options="--max_profiler_iterations 1",
    )
    # return inference_job.download_output_data()
    inference_job.wait()
    return inference_job.job_id


def load_ground_truth():
    """返回 image_names 列表和对应的 gt_text_nums (list of list of int)，
    以及 text_nums 顺序列表（与上传顺序一致）。"""
    # txt_list 顺序（上传顺序）
    txt_nums = []
    with open(TXT_LIST_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            txt_nums.append(int(row["Text_nums"]))

    txt_num_to_idx = {n: i for i, n in enumerate(txt_nums)}

    # img_list 顺序（上传顺序）
    image_names = []
    positive_indices = []  # 每张图对应的文本在 txt_nums 中的下标列表
    with open(IMG_LIST_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            image_names.append(row["Image_names"].strip())
            gt = [int(x) for x in row["Text_nums"].split(";") if x.strip()]
            positive_indices.append(
                [txt_num_to_idx[n] for n in gt if n in txt_num_to_idx]
            )

    return image_names, positive_indices


def compute_recall_at_k(img_embeds, txt_embeds, positive_indices, k=10):
    img_embeds = img_embeds / (np.linalg.norm(img_embeds, axis=1, keepdims=True))
    txt_embeds = txt_embeds / (np.linalg.norm(txt_embeds, axis=1, keepdims=True))
    sim = cosine_similarity(img_embeds, txt_embeds)
    recalls = []
    for i, gt_idx in enumerate(positive_indices):
        if not gt_idx:
            continue
        top_k = set(np.argsort(-sim[i])[:k].tolist())
        matched = len(top_k & set(gt_idx))
        recalls.append(matched / len(gt_idx))
    return float(np.mean(recalls))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP image-text retrieval evaluation with Recall@k"
    )
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--image-compiled-id", type=str, default="j5wd9zmzg")
    parser.add_argument("--text-compiled-id", type=str, default="jp1d81qkp")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Define target device
    device = qai_hub.Device("XR2 Gen 2 (Proxy)")

    text_compiled_id = args.text_compiled_id
    image_compiled_id = args.image_compiled_id
    k = args.k

    # TODO: Define tasks with their corresponding compiled job IDs and dataset IDs
    tasks = {
        "text": {"compiled_id": text_compiled_id, "dataset_id": "d91ydw4e9"},
        "image": {"compiled_id": image_compiled_id, "dataset_id": "d9k1oyzx2"},
    }

    # Dictionary to store outputs separately
    outputs = {}

    for task_name, info in tasks.items():
        compiled_id = info["compiled_id"]
        input_dataset = qai_hub.get_dataset(info["dataset_id"])

        # Retrieve the compiled model
        job = qai_hub.get_job(compiled_id)
        compiled_model = job.get_target_model()

        # Run inference
        print(
            f"Running inference for {task_name} model {compiled_model.model_id} on device {device.name}"
        )
        inference_id = run_inference(
            compiled_model,
            f"{args.model_name}_{task_name}_{compiled_id}",
            device,
            input_dataset,
        )
        inference_job = qai_hub.get_job(inference_id)

        if inference_job.get_status().failure:
            print(f"{task_name.capitalize()} inference failed")
            outputs[task_name] = None
        else:
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

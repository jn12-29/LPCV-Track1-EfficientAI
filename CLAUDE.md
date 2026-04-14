# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Personal Preferences

- Use English for code and annotations, but communicate with me in Chinese.
- Update README.md, CLAUDE.md, and any \*.sh files when modifications are needed.

## Project Overview

LPCVC 2026 Track 1 — Image-to-Text Retrieval competition. Goal: maximize Recall@K on a Qualcomm XR2 Gen 2 device. The model is MobileCLIP2 (open_clip), exported to ONNX, then compiled via QAI Hub for on-device inference.

## Setup

```bash
pip install -r requirements.txt
qai-hub configure  # requires API token from QAI Hub
```

Recommended environment variables:

```bash
export OMP_NUM_THREADS=1
export HF_HOME=/mnt/sada1/data
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONNOUSERSITE=1
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

Optional GPU visibility override:

```bash
export CUDA_VISIBLE_DEVICES=<gpu_id_or_gpu_list>
```

## End-to-End Pipeline

```bash
# 1. Export ONNX (fp32) — output goes to exported_{model_name}_onnx/
python pipeline/export_onnx.py --model-name MobileCLIP2-S0

# 2. Compile for XR2 Gen 2 and submit profiling job to QAI Hub
python pipeline/compile_and_profile.py --model-name MobileCLIP2-S0 [--postfix <suffix>]

# 3. Evaluate
python pipeline/eval_local.py --model-name MobileCLIP2-S0 --k 10        # torch, local

# Remote evaluation (three modes):
python pipeline/eval_remote.py --upload-dataset \
    --image-compiled-id <id> --text-compiled-id <id>            # Mode A: upload + infer
python pipeline/eval_remote.py \
    --image-compiled-id <id> --text-compiled-id <id>            # Mode B: infer, existing dataset
python pipeline/eval_remote.py \
    --image-inference-id <id> --text-inference-id <id>          # Mode C: reuse inference jobs
```

Available model names: `MobileCLIP2-S0`, `MobileCLIP2-S2`, `MobileCLIP2-S3`

## Architecture

### Shared utilities (`utils/`)

- **`utils/clip_utils.py`** — `_load_clip()`: canonical model loader shared by all scripts. Includes MobileCLIP2-specific `image_mean/std=(0,0,0)/(1,1,1)` kwargs and tokenizer fallback.
- **`utils/preprocess.py`** — `preprocess_image()`: resize 224×224, divide by 255, return `(3,224,224)` float32 tensor. No mean/std normalization (competition requirement).
- **`utils/data_utils.py`** — `_batched`, `recall_at_k`, CSV loaders (`load_image_to_textnums`, `load_textnums_to_texts`), `load_ground_truth`.

### Pipeline (`pipeline/`)

- **`pipeline/dataset.py`** — `RetrievalEvalDataset` (per-image or per-text iteration; `get_image_to_text_eval_data()` returns all texts + per-image positive indices) and `ImageTextRetrievalDataset` (positive pairs).
- **`pipeline/export_onnx.py`** — wraps image/text encoders in `OpenClipVisionEncoder` / `OpenClipTextEncoder`, exports to ONNX opset 18, verifies outputs against PyTorch.
- **`pipeline/compile_and_profile.py`** — submits ONNX models to QAI Hub as compile jobs (runtime: `qnn_dlc`, `--truncate_64bit_io`), then profile jobs. Auto-shares results with `lowpowervision@gmail.com`.
- **`pipeline/eval_local.py`** — torch local Recall@K evaluation.
- **`pipeline/eval_remote.py`** — dataset upload, QAI Hub inference submission, and Recall@K. Three modes: (A) upload + infer, (B) infer with existing dataset IDs, (C) reuse inference job outputs.

### Training (`train/`)

- **`train/finetune.py`** — fine-tunes MobileCLIP2 on flat contrastive JSONL from `build_datasets/`. Supports single-GPU training via `--gpu-ids 0` and multi-GPU DDP training via `torchrun --nnodes=1 --master_addr=127.0.0.1 --master_port=<port> ... --gpu-ids 0,1,...`. `--batch-size` is per-GPU. Writes each run to `checkpoints/<model>__<config>__<timestamp>/` with `train.log`, metrics files, epoch-tagged checkpoints, `run_config.json`, and TensorBoard logs under `tensorboard/`.
- **`train/analyze_hard_negatives.py`** — analyzes positive vs hard-negative similarity distributions.

### Dataset builder (`build_datasets/`)

Generates fine-grained retrieval training data from images via an OpenRouter VLM API. Outputs flat contrastive JSONL records consumed directly by `train/finetune.py` and `train/analyze_hard_negatives.py`. Config and prompts are in `DEFINE.py`; the builder script is `vlm_dataset_builder.py`.

Expected flat training record format:

```json
{"image_path": "build_datasets/data/VG_100K/107914.jpg", "positives": ["..."], "hard_negatives": ["..."]}
```

Training and analysis scripts are expected to run from the repository root so `image_path` can be opened directly.

### Samples Dataset layout expected

```
sample_data/
├── images/          # .jpg, .png, .jpeg, .webp
├── img_list.csv     # columns: Image_names, Text_nums (semicolon-separated)
└── txt_list.csv     # columns: Text_nums, Unique_Texts
```

### Preprocessing (non-standard)

Images are resized to 224×224 and divided by 255. **No ImageNet mean/std normalization** — this differs from typical CLIP usage.

### QAI Hub compile settings

- Target device: `"XR2 Gen 2 (Proxy)"`
- Runtime: `qnn_dlc`
- Image input spec: `(1, 3, 224, 224)` float32
- Text input spec: `(1, 77)` int64

### Dataset builder (`build_datasets/`)

Generates fine-grained retrieval training data from images via an OpenRouter VLM API. Config and prompts are in `DEFINE.py`; the builder script is `vlm_dataset_builder.py`.

We are working on it, we should focus on this part.

## Key Gotchas

- `reparameterize_model()` from `timm.utils` must be called on the model before ONNX export.
- The `TextEncoder` wrapper in `mobileclipv2.py` normalizes padding: tokens after the EOS position (argmax) are zeroed out to ensure consistent input regardless of tokenizer padding style.
- QAI Hub auto-converts to fp16; `mobileclipv2_fp16.py` is a dead file.
- The tokenizer is always `open_clip.get_tokenizer("ViT-B-32")` regardless of which MobileCLIP2 variant is used, infact, they share the same tokenizer.

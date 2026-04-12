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

Required environment variables:

```bash
export CUDA_VISIBLE_DEVICES=<gpu_id>
export HF_HOME=/mnt/sada1/data
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONNOUSERSITE=1
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

## End-to-End Pipeline

```bash
# 1. Export ONNX (fp32) — output goes to exported_{model_name}_onnx/
python mobileclipv2.py --model-name MobileCLIP2-S0

# 1b. (Optional) Export quantized ONNX
python mobileclipv2_quant.py --model-name MobileCLIP2-S0

# 2. Compile for XR2 Gen 2 and submit profiling job to QAI Hub
python compile_and_profile.py --model-name MobileCLIP2-S0 [--postfix <suffix>]

# 2b. (Alternative) Remote quantization via QAI Hub
python qai_quant_compile_profile.py

# 3. Evaluate
python eval_local.py --model-name MobileCLIP2-S0 --k 10        # torch, local
python eval_onnx_local.py --model-name MobileCLIP2-S0 --k 10   # ONNX, local
python eval_remote.py --image-compiled-id <id> --text-compiled-id <id> --k 10  # QAI Hub cloud
python eval_remote_eval_inference.py --inference-job-id <id>    # from existing inference job

# 4. (Optional) Fine-tune (not done infact, just demo)
python finetune.py --model-name MobileCLIP2-S0 --root-dir ./sample_data ...
```

Available model names: `MobileCLIP2-S0`, `MobileCLIP2-S2`, `MobileCLIP2-S3`

## Architecture

### Core modules

- **`eval_local.py`** — defines `_load_clip()` (shared by all scripts), runs Recall@K evaluation with PyTorch. Also exports `encode_images`, `encode_texts`, `evaluate_image_to_text_recall_at_k`.
- **`mobileclipv2.py`** — wraps image/text encoders in `OpenClipVisionEncoder` / `OpenClipTextEncoder` nn.Module subclasses, exports to ONNX opset 18, and verifies ONNX outputs against PyTorch.
- **`dataset.py`** — two Dataset classes:
  - `ImageTextRetrievalDataset` — positive (image, text) pairs for training.
  - `RetrievalEvalDataset` — per-image or per-text iteration for evaluation; `get_image_to_text_eval_data()` returns all texts + per-image positive indices.
- **`utils.py`** — `calculate_recall_at_k`.
- **`compile_and_profile.py`** — submits ONNX models to QAI Hub as compile jobs (runtime: `qnn_dlc`, `--truncate_64bit_io`), then profile jobs. Auto-shares results with `lowpowervision@gmail.com`.
- **`Token.py`** — hardcoded QAI Hub token and dataset IDs (`d2qe36jl2` images, `d95k6jwm9` texts).

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

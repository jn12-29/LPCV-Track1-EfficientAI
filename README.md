# LPCVC 2026 Track 1 — Efficient AI (Image-to-Text Retrieval)

Goal: maximize Recall@K on a Qualcomm XR2 Gen 2 device using MobileCLIP2 (open_clip), exported to ONNX and compiled via QAI Hub for on-device inference.

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
python eval_local.py --model-name MobileCLIP2-S0 --k 10   # PyTorch, local

# Remote evaluation (three modes — see below)
python eval_remote.py --upload-dataset \
    --image-compiled-id <id> --text-compiled-id <id>       # Mode A: upload + infer
python eval_remote.py \
    --image-compiled-id <id> --text-compiled-id <id>       # Mode B: infer with existing dataset
python eval_remote.py \
    --image-inference-id <id> --text-inference-id <id>     # Mode C: reuse inference jobs
```

Available model names: `MobileCLIP2-S0`, `MobileCLIP2-S2`, `MobileCLIP2-S3`

## Training

Run training and analysis from the repository root.

The builder writes flat contrastive JSONL records consumed directly by `train/`:

```json
{"image_path": "build_datasets/data/VG_100K/107914.jpg", "positives": ["..."], "hard_negatives": ["..."]}
```

```bash
# fine-tune MobileCLIP2 on builder output
python train/finetune.py \
    --jsonl-path build_datasets/data/VG_100K_GEMINI31FLASHLITE_NEW/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-S2 --batch-size 256 --epochs 20

# analyze positive vs hard-negative similarity distributions
python train/analyze_hard_negatives.py \
    --jsonl-path build_datasets/data/VG_100K_GEMINI31FLASHLITE_NEW/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-S0
```

`train/finetune.py` reads the JSONL records as-is, samples one positive text and `--num-hard-negatives` hard negatives per image, and optimizes `CLIP loss + hard negative loss`.

## `eval_remote.py` Modes

| Mode | Arguments | Description |
|------|-----------|-------------|
| **A** | `--upload-dataset` + compiled IDs | Upload local `sample_data` to QAI Hub, then submit inference |
| **B** | compiled IDs only | Submit inference using existing dataset IDs (defaults: `d2qe36jl2` / `d95k6jwm9`) |
| **C** | inference IDs | Download outputs from already-completed inference jobs |

Optional overrides for Mode B: `--image-dataset-id`, `--text-dataset-id`.

## Architecture

### Core modules

- **`eval_local.py`** — defines `_load_clip()` (shared by all scripts), runs Recall@K evaluation with PyTorch. Exports `encode_images`, `encode_texts`, `evaluate_image_to_text_recall_at_k`.
- **`eval_remote.py`** — unified remote evaluation: dataset upload, inference submission, and Recall@K computation. Replaces the former `eval_upload_dataset.py` and `eval_remote_eval_inference.py`.
- **`eval_common.py`** — shared helpers: `load_ground_truth()`, `recall_at_k()`.
- **`mobileclipv2.py`** — wraps image/text encoders in `OpenClipVisionEncoder` / `OpenClipTextEncoder` nn.Module subclasses, exports to ONNX opset 18, verifies ONNX outputs against PyTorch.
- **`mobileclipv2_quant.py`** — quantized ONNX export.
- **`compile_and_profile.py`** — submits ONNX models to QAI Hub as compile jobs (runtime: `qnn_dlc`, `--truncate_64bit_io`), then profile jobs.
- **`qai_quant_compile_profile.py`** — remote quantization + compile + profile via QAI Hub.

### Sample dataset layout

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

Generates fine-grained retrieval training data from images via an OpenRouter VLM API. Config and prompts are in `DEFINE.py`; the builder script is `vlm_dataset_builder.py`. The main training artifact is `dataset_raw_contrastive.jsonl`, which is consumed directly by `train/finetune.py` and `train/analyze_hard_negatives.py`.

## Key Gotchas

- `reparameterize_model()` from `timm.utils` must be called on the model before ONNX export.
- The `TextEncoder` wrapper in `mobileclipv2.py` normalizes padding: tokens after the EOS position (argmax) are zeroed out to ensure consistent input regardless of tokenizer padding style.
- QAI Hub auto-converts to fp16; `mobileclipv2_fp16.py` is a dead file.
- The tokenizer is always `open_clip.get_tokenizer("ViT-B-32")` regardless of which MobileCLIP2 variant is used — all variants share the same tokenizer.

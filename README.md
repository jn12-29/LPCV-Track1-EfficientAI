# LPCVC 2026 Track 1 — Efficient AI (Image-to-Text Retrieval)

Goal: maximize Recall@K on a Qualcomm XR2 Gen 2 device using MobileCLIP2 (open_clip), exported to ONNX and compiled via QAI Hub for on-device inference.

## Setup

```bash
pip install -r requirements.txt
qai-hub configure --api_token <您的TOKEN> # requires API token from QAI Hub
```

AIMET (for QAT) requires a separate wheel matched to your CUDA version — see comments in `requirements.txt`.

## End-to-End Pipeline

```bash
# 1. Export ONNX (fp32) — output goes to exported_{model_name}_onnx/
python pipeline/export_onnx.py --model-name MobileCLIP2-B

# 2. Compile for XR2 Gen 2 and submit profiling job to QAI Hub
python pipeline/compile_and_profile.py --model-name MobileCLIP2-B [--postfix <suffix>]

# 3. Evaluate
python pipeline/eval_local.py --model-name MobileCLIP2-B --k 10        # torch, local

# Remote evaluation (three modes):
python pipeline/eval_remote.py --upload-dataset \
    --image-compiled-id <id> --text-compiled-id <id>            # Mode A: upload + infer
python pipeline/eval_remote.py \
    --image-compiled-id <id> --text-compiled-id <id>            # Mode B: infer, existing dataset
python pipeline/eval_remote.py \
    --image-inference-id <id> --text-inference-id <id>          # Mode C: reuse inference jobs
```

Available model names: `MobileCLIP2-B`

## Training (Fine-tuning)

Run all training commands from the repository root. Flat contrastive JSONL format:

```json
{
  "image_path": "build_datasets/data/VG_100K/107914.jpg",
  "positives": ["..."],
  "hard_negatives": ["..."]
}
```

```bash
# Single GPU
python train/finetune.py \
    --jsonl-path build_datasets/data/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 0 --batch-size 256 --epochs 20

# Multi-GPU DDP
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 \
    train/finetune.py \
    --jsonl-path build_datasets/data/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 0,1 --batch-size 256 --epochs 20

# Resume from checkpoint
python train/finetune.py ... --resume checkpoints/.../checkpoint_latest_epoch_05.pt
```

`--batch-size` is per-GPU. Effective global batch size = `batch_size × world_size × accum_freq`.

Each run writes into `checkpoints/<model>__<config>__<timestamp>/` containing `train.log`, `metrics.csv`, `metrics.jsonl`, `training_curves.png`, epoch-tagged checkpoints, `run_config.json`, and TensorBoard logs.

```bash
tensorboard --logdir checkpoints
```

## QAT (Quantization-Aware Training)

QAT optimizes model weights under simulated int8 quantization using AIMET, reducing accuracy loss when QAI Hub compiles to QNN int8.

```bash
# QAT from pretrained weights (W8A8)
python train/finetune.py \
    --jsonl-path build_datasets/data/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 0 --batch-size 64 --epochs 5 \
    --qat-enabled --qat-weight-bw 8 --qat-act-bw 8 --qat-calib-batches 32

# QAT from a regular finetune checkpoint (W8A16)
python train/finetune.py \
    --jsonl-path build_datasets/data/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 0 --batch-size 64 --epochs 3 \
    --resume checkpoints/.../checkpoint_epoch_20.pt \
    --qat-enabled --qat-weight-bw 8 --qat-act-bw 16

# Resume QAT from a QAT checkpoint (auto-detected, calibration skipped)
python train/finetune.py ... --resume checkpoints/.../checkpoint_latest_epoch_02.pt \
    --qat-enabled

# Also export ONNX at each numbered checkpoint
python train/finetune.py ... --qat-enabled --export-onnx

# Export ONNX from an existing QAT checkpoint
python pipeline/export_onnx.py --model-name MobileCLIP2-B \
    --checkpoint-path checkpoints/.../MobileCLIP2-B_finetuned.pt
```

QAT checkpoints are standard `.pt` files with extra fields (`qat_enabled`, `qat_encodings`) and are auto-detected on `--resume`.

## MLP Reconstruction (GELU → ReLU)

Replaces all MLP GELU activations with ReLU via layer-by-layer knowledge distillation, without quantization. Based on APHQ-ViT (CVPR 2025). Speeds up inference ~10–20% with <0.5% accuracy loss.

```bash
# Full reconstruction on pretrained B (~2 hours, 32 blocks)
# Output auto-generated: checkpoints/{model}__{config}__{timestamp}/mlp_relu.pt
# Recall@K eval runs automatically at the end (--skip-eval to disable)
python mlp_reconstruction/run.py \
    --model-name MobileCLIP2-B --gpu-id 2 \
    --n-calib 1024 --n-iters 20000

# On top of a fine-tuned checkpoint
python mlp_reconstruction/run.py \
    --model-name MobileCLIP2-B --gpu-id 2 \
    --checkpoint-path checkpoints/.../checkpoint_latest_epoch_100.pt

# Evaluate reconstructed model separately
python pipeline/eval_local.py --model-name MobileCLIP2-B --k 10 \
    --checkpoint-path checkpoints/<run_name>/mlp_relu.pt

# Export reconstructed model to ONNX (_load_clip auto-detects relu_blocks checkpoint)
python pipeline/export_onnx.py --model-name MobileCLIP2-B \
    --checkpoint-path checkpoints/<run_name>/mlp_relu.pt

# Resume after crash
python mlp_reconstruction/run.py \
    --model-name MobileCLIP2-B --gpu-id 2 \
    --resume-from checkpoints/<run_name>/mlp_relu.pt.tmp \
    --skip-to visual[s1b0]
```

Reconstruction proceeds **block by block** (serial). Each block: collect calibration activations → replace GELU with ReLU → distill (`L_Direct + 2×L_Clamp`) → save intermediate checkpoint (`.tmp`). Calibration data is read from `build_datasets/data/vg_llm_contrastive.jsonl`.

Each run writes into the checkpoint directory: `train.log`, `run_config.json`, `metrics.csv`, `metrics.jsonl`, and `reconstruction_curves.png` (cosine similarity and final loss per block).

Key options: `--gpu-id` (single GPU), `--n-calib` (default 1024), `--n-iters` (default 20000), `--lr` (default 1e-3), `--aph-mode uniform|magnitude`, `--output-dir` (default `checkpoints`), `--output` (overrides auto-generated path).

Block counts: B = 32 (12 text + 20 visual), B = 56, B = 24.

## `eval_remote.py` Modes

| Mode  | Arguments                         | Description                                                  |
| ----- | --------------------------------- | ------------------------------------------------------------ |
| **A** | `--upload-dataset` + compiled IDs | Upload local `sample_data` to QAI Hub, then submit inference |
| **B** | compiled IDs only                 | Submit inference using existing dataset IDs                  |
| **C** | inference IDs                     | Download outputs from already-completed inference jobs       |

## Architecture

### Shared utilities (`utils/`)

- **`utils/clip_utils.py`** — `_load_clip()`: canonical model loader. Supports regular and QAT checkpoints via `qat_config` parameter (auto-detects checkpoint type). MobileCLIP2-specific `image_mean/std=(0,0,0)/(1,1,1)`.
- **`utils/qat_utils.py`** — `QATConfig` dataclass; `wrap_model_for_qat()` (reparameterize + AIMET QuantSim), `calibrate_quantsim()`, `get_qat_encodings_json()`, `extract_base_model_state_dict()`.
- **`utils/preprocess.py`** — `preprocess_image()`: resize 224×224, divide by 255, return `(3,224,224)` float32. No mean/std normalization.
- **`utils/data_utils.py`** — `_batched`, `recall_at_k`, CSV loaders, `load_ground_truth`.

### Pipeline (`pipeline/`)

- **`pipeline/export_onnx.py`** — exports image/text encoders to ONNX opset 18, simplifies, verifies. Key arg: `--max-text-len N` (default 77) — ONNX external I/O stays `(1, 77)` per competition spec, but the text encoder internally truncates tokens to `(1, N)` and attn_mask to `(N, N)` before the transformer, so the compiled model attends over only N positions (faster for short texts). All checkpoint types auto-detected via `--checkpoint-path`.
- **`pipeline/compile_and_profile.py`** — submits ONNX to QAI Hub (runtime: `qnn_dlc`, `--truncate_64bit_io`), then profile jobs. Auto-shares results.
- **`pipeline/eval_local.py`** — torch local Recall@K evaluation.
- **`pipeline/eval_remote.py`** — dataset upload, QAI Hub inference, Recall@K. Three modes: A/B/C.
- **`pipeline/dataset.py`** — `RetrievalEvalDataset` and `ImageTextRetrievalDataset`.

### Training (`train/`)

- **`train/finetune.py`** — CLI entry point. Args: model, data, optimizer, loss, DDP, QAT (`--qat-*`), resume (`--resume`), export (`--export-onnx`).
- **`train/trainer.py`** — `run_training()`: model load → dataset → QAT calibration → DDP wrap → training loop → export.
- **`train/train_step.py`** — `train_one_epoch()`.
- **`train/loss.py`** — CLIP InfoNCE + hard-negative loss; SigLIP loss.
- **`train/data.py`** — `ContrastiveRecordDataset`, `create_collate_fn()`.
- **`train/train_utils.py`** — `save_checkpoint()` (QAT-aware), logging, metrics helpers.
- **`train/optim.py`** — AdamW + cosine scheduler with warmup.
- **`train/distributed.py`** — DDP init, seed management.
- **`train/metrics.py`** — CSV/JSONL metrics and training curve plots.
- **`train/analyze_hard_negatives.py`** — similarity distribution analysis.

### Dataset builder (`build_datasets/`)

Generates fine-grained retrieval training data from images via an OpenRouter VLM API. Config and prompts in `DEFINE.py`; builder in `vlm_dataset_builder.py`. Output: `dataset_raw_contrastive.jsonl`.

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
- Text input spec: `(1, 77)` int32

## Key Gotchas

- `reparameterize_model()` from `timm.utils` must be called before ONNX export — this folds multi-branch conv structures. For QAT, it is called automatically inside `wrap_model_for_qat()` before QuantSim creation.
- For QAT checkpoints, the model must be reparameterized before QuantSim is created so the state dict key names match. `_load_clip()` handles this automatically.
- The `OpenClipTextEncoder` wrapper zeros tokens after the EOS position to ensure consistent input regardless of tokenizer padding. With `--max-text-len N < 77`, it additionally slices `token_ids[:, :N]` and truncates `attn_mask` to `(N, N)` — the ONNX graph still accepts `(1, 77)` externally but computes attention over only N positions, reducing latency.
- The tokenizer is always `open_clip.get_tokenizer("ViT-B-32")` for all MobileCLIP2 variants.

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

AIMET (for QAT) requires a separate wheel matched to your CUDA version — see comments in `requirements.txt`.

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

- **`utils/clip_utils.py`** — `_load_clip()`: canonical model loader shared by all scripts. Accepts `qat_config` parameter to wrap with AIMET QuantSim; auto-detects regular vs. QAT checkpoint via `qat_enabled` key. MobileCLIP2-specific `image_mean/std=(0,0,0)/(1,1,1)`.
- **`utils/qat_utils.py`** — QAT helpers: `QATConfig` dataclass; `wrap_model_for_qat()` (reparameterize + QuantSim); `calibrate_quantsim()`; `get_qat_encodings_json()`; `extract_base_model_state_dict()`.
- **`utils/preprocess.py`** — `preprocess_image()`: resize 224×224, divide by 255, return `(3,224,224)` float32 tensor. No mean/std normalization (competition requirement).
- **`utils/data_utils.py`** — `_batched`, `recall_at_k`, CSV loaders (`load_image_to_textnums`, `load_textnums_to_texts`), `load_ground_truth`.

### Pipeline (`pipeline/`)

- **`pipeline/dataset.py`** — `RetrievalEvalDataset` (per-image or per-text iteration; `get_image_to_text_eval_data()` returns all texts + per-image positive indices) and `ImageTextRetrievalDataset` (positive pairs).
- **`pipeline/export_onnx.py`** — wraps image/text encoders in `OpenClipVisionEncoder` / `OpenClipTextEncoder`, exports to ONNX opset 18, verifies outputs. Exposes `export_encoders_to_onnx(clip_model, output_dir)` for reuse. Handles QAT checkpoints via `--checkpoint-path` (auto-detects, extracts base weights).
- **`pipeline/compile_and_profile.py`** — submits ONNX models to QAI Hub as compile jobs (runtime: `qnn_dlc`, `--truncate_64bit_io`), then profile jobs. Auto-shares results with `lowpowervision@gmail.com`.
- **`pipeline/eval_local.py`** — torch local Recall@K evaluation.
- **`pipeline/eval_remote.py`** — dataset upload, QAI Hub inference submission, and Recall@K. Three modes: (A) upload + infer, (B) infer with existing dataset IDs, (C) reuse inference job outputs.

### Training (`train/`)

- **`train/finetune.py`** — CLI entry point (`parse_args` + `main`). Key args: `--resume` (regular or QAT checkpoint, auto-detected), `--qat-enabled`, `--qat-weight-bw`, `--qat-act-bw`, `--qat-calib-batches`, `--qat-quant-scheme`, `--export-onnx`. Supports single-GPU and multi-GPU DDP via `torchrun`.
- **`train/trainer.py`** — `run_training()`: model load → dataset → QAT calibration (before DDP wrap) → DDP wrap → training loop → optional ONNX export. `save_checkpoint()` called with `sim` when QAT active.
- **`train/train_step.py`** — `train_one_epoch()`: forward, loss, backward, grad clip, scheduler step.
- **`train/loss.py`** — `open_clip.ClipLoss` + `compute_hard_negative_loss()` (hinge/logsigmoid); `SigLipLoss`.
- **`train/data.py`** — `ContrastiveRecordDataset`, `create_collate_fn()` (tokenizes, pads hard negatives to 3D tensor).
- **`train/train_utils.py`** — `save_checkpoint(... sim=None)`: when `sim` is provided, adds `qat_enabled`, `qat_weight_bw`, `qat_act_bw`, `qat_encodings` to checkpoint. Also: logging, metrics CSV/JSONL, TensorBoard helpers.
- **`train/optim.py`** — AdamW + cosine decay scheduler with warmup.
- **`train/distributed.py`** — DDP init/cleanup, seed management, `broadcast_run_timestamp`.
- **`train/metrics.py`** — `append_metrics_row`, `append_metrics_jsonl`, `plot_training_curves`.
- **`train/analyze_hard_negatives.py`** — analyzes positive vs hard-negative similarity distributions.

### QAT pipeline

1. `_load_clip(..., qat_config=QATConfig(...))` in `trainer.py`:
   - Regular/pretrained checkpoint → load state dict → `wrap_model_for_qat()` (reparameterize + QuantSim)
   - QAT checkpoint → `wrap_model_for_qat()` (reparameterize + QuantSim) → load state dict + encodings
2. If `_qat_needs_calibration`: call `calibrate_quantsim()` before DDP wrap
3. Training loop unchanged; `save_checkpoint(sim=sim)` saves QAT state
4. ONNX export (`--export-onnx`): extracts base weights from QuantSim model into fresh reparameterized model, exports via `export_encoders_to_onnx()`

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

## Key Gotchas

- `reparameterize_model()` from `timm.utils` must be called before ONNX export. For QAT, it is called automatically inside `wrap_model_for_qat()` — do not call it again afterward (would break fake quant nodes).
- QAT checkpoint loading: the model must be reparameterized before QuantSim is created so the state dict key names match. `_load_clip()` handles this — regular checkpoints are loaded before reparameterize; QAT checkpoints are loaded after.
- The `OpenClipTextEncoder` wrapper zeros tokens after the EOS position (argmax) to ensure consistent input regardless of tokenizer padding style.
- QAI Hub auto-converts to fp16 during compilation.
- The tokenizer is always `open_clip.get_tokenizer("ViT-B-32")` regardless of which MobileCLIP2 variant is used — all variants share the same tokenizer.

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
qai-hub configure --api-token <api_token> # requires API token from QAI Hub
```

AIMET (for QAT) requires a separate wheel matched to your CUDA version — see comments in `requirements.txt`.

## Architecture

### Shared utilities (`utils/`)

- **`utils/clip_utils.py`** — `_load_clip()`: canonical model loader shared by all scripts. Accepts `qat_config` parameter to wrap with AIMET QuantSim; auto-detects regular vs. QAT checkpoint via `qat_enabled` key. MobileCLIP2-specific `image_mean/std=(0,0,0)/(1,1,1)`.
- **`utils/qat_utils.py`** — QAT helpers: `QATConfig` dataclass; `wrap_model_for_qat()` (reparameterize + QuantSim); `calibrate_quantsim()`; `get_qat_encodings_json()`; `extract_base_model_state_dict()`.
- **`utils/preprocess.py`** — `preprocess_image()`: resize 224×224, divide by 255, return `(3,224,224)` float32 tensor. No mean/std normalization (competition requirement).
- **`utils/data_utils.py`** — `_batched`, `recall_at_k`, CSV loaders (`load_image_to_textnums`, `load_textnums_to_texts`), `load_ground_truth`.

### Pipeline (`pipeline/`)

- **`pipeline/dataset.py`** — `RetrievalEvalDataset` (per-image or per-text iteration; `get_image_to_text_eval_data()` returns all texts + per-image positive indices) and `ImageTextRetrievalDataset` (positive pairs).
- **`pipeline/export_onnx.py`** — wraps image/text encoders in `OpenClipVisionEncoder` / `OpenClipTextEncoder`, exports to ONNX opset 18, simplifies, verifies. Key arg: `--max-text-len N` (default 77) — the ONNX external I/O is always `(1, 77)` as required by the competition, but internally the text encoder truncates tokens to `(1, N)` and attn_mask to `(N, N)` before the transformer, so the compiled model only attends over N positions (faster when real texts are short). All checkpoint types auto-detected via `--checkpoint-path` (`_load_clip` handles regular / QAT / relu-reconstruction).
- **`pipeline/compile_and_profile.py`** — submits ONNX models to QAI Hub as compile jobs (runtime: `qnn_dlc`, `--truncate_64bit_io`), then profile jobs. Auto-shares results with `lowpowervision@gmail.com`.
- **`pipeline/eval_local.py`** — torch local Recall@K evaluation. Also exports `eval_recall_with_model(model, tokenizer, root_dir, image_csv, text_csv, device, batch_size, k)` for in-training sample-set eval without reloading the model.
- **`pipeline/eval_remote.py`** — dataset upload, QAI Hub inference submission, and Recall@K. Three modes: (A) upload + infer, (B) infer with existing dataset IDs, (C) reuse inference job outputs.

### Training (`train/`)

- **`train/finetune.py`** — CLI entry point (`parse_args` + `main`). Key args: `--resume` (regular or QAT checkpoint, auto-detected), `--qat-enabled`, `--qat-weight-bw`, `--qat-act-bw`, `--qat-calib-batches`, `--qat-quant-scheme`, `--export-onnx`. Validation args: `--val-split-size` (default 1024), `--val-split-seed`, `--val-every-n-epochs` (default 1), `--no-val-loss`, `--no-val-recall`, `--val-k`, `--val-batch-size`. Sample-set args: `--sample-eval-every-n-epochs` (default 0), `--sample-eval-root`, `--sample-eval-image-csv`, `--sample-eval-text-csv`, `--sample-eval-k`, `--sample-eval-batch-size`. Best-checkpoint: `--best-metric` (choices: `sample_recall` [default], `val_recall`, `val_loss`). Supports single-GPU and multi-GPU DDP via `torchrun`.
- **`train/trainer.py`** — `run_training()`: model load → records load + val split → dataset → QAT calibration (before DDP wrap) → DDP wrap → training loop → per-epoch val eval → optional ONNX export. Saves `checkpoint_best.pt` when primary val metric improves (priority: val_recall > sample_recall > val_loss).
- **`train/train_step.py`** — `train_one_epoch()`: forward, loss, backward, grad clip, scheduler step.
- **`train/val_step.py`** — `eval_val_loss()` (local InfoNCE + hard-neg, no DDP gather) and `eval_val_recall()` (Recall@K from JSONL records, builds text corpus from all positives).
- **`train/loss.py`** — `open_clip.ClipLoss` + `compute_hard_negative_loss()` (hinge/logsigmoid); `SigLipLoss`.
- **`train/data.py`** — `load_jsonl_records()` (standalone loader), `split_val_records()` (tail or random split), `ContrastiveRecordDataset` (accepts `records=` list to skip file I/O), `create_collate_fn()`.
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

### MLP Reconstruction (`mlp_reconstruction/`)

Replaces MLP GELU activations with ReLU and recovers accuracy via layer-by-layer distillation (APHQ-ViT, CVPR 2025). No quantization. Entry point: `mlp_reconstruction/run.py`.

- **`mlp_reconstruction/mlp_blocks.py`** — `MLPBlockInfo` class (dataclass + methods): `set_activation(act_cls)`, `is_relu()`, `register_x_hook(callback)`, `forward(X)`, `trainable_params()`, `save_weights()`, `restore_weights(saved)`. All `conv_stem` vs standard-MLP branching is encapsulated here. `iter_mlp_blocks(model)` enumerates all text + visual MLP blocks. Module-level wrappers `replace_gelu_with_relu` / `restore_gelu` / `is_relu_active` / `apply_relu_blocks` are preserved for `clip_utils.py` compatibility.
- **`mlp_reconstruction/data.py`** — `VGCalibrationLoader(jsonl_path, project_root, tokenizer, n_calib, batch_size)` reads image–text pairs from `vg_llm_contrastive.jsonl`; `get_image_batches()` / `get_text_batches()`.
- **`mlp_reconstruction/collect.py`** — `collect_mlp_io(model, info, image_batches, text_batches, device)` captures per-block input X and output O via `info.register_x_hook` + a forward hook on `mlp`. Also computes `gelu_ub` (99th-percentile of fc2 pre-activations). Returns CPU tensors.
- **`mlp_reconstruction/aph.py`** — `compute_aph_weights(O_GELU, mode='uniform')` returns H_bar importance weights; shape (D,) for text/ViT, (C,) for FastVit.
- **`mlp_reconstruction/distill.py`** — `distill_mlp(model, info, X_all, O_all, ...)` replaces GELU with ReLU via `info.set_activation` and optimizes via `info.trainable_params()` / `info.forward()`. Loss: `L_Direct + alpha×L_Clamp` (L_Clamp only when `info.fc2 is not None` and `alpha > 0`). Supports `keep_activation=True` for GELU drift distillation.
- **`mlp_reconstruction/verify.py`** — `verify_reconstruction(info, X, O_orig, device)` returns mean cosine similarity; `verify_gelu_drift(model, info, ...)` re-collects block output under the current (partially-replaced) model and compares to O_orig, returning `(cos_sim, mse, X_cur)`.
- **`mlp_reconstruction/runner.py`** — `_get_block_inputs` / `_process_block` (distill → verify → optional revert + GELU drift distill for one block) / `cmd_train` (full orchestration: pre-collect O_orig, resume, main loop, checkpoint, eval).
- **`mlp_reconstruction/run.py`** — CLI entry point (`parse_args` + `main`). Supports `--gpu-id`, `--greedy`, `--gelu-threshold`, `--keep-gelu-blocks`, `--resume-from`, `--skip-to`, `--skip-eval`. Output path auto-generated as `{output_dir}/{model}__{config}__{timestamp}/mlp_relu.pt`.

Checkpoint format: `{'model_state_dict': ..., 'model_name': ..., 'relu_blocks': [list of labels], 'relu_image': bool, 'relu_text': bool}`. GELU and ReLU have no parameters so state_dict is identical in size — `relu_blocks` is the only structural hint needed on reload.

### Dataset builder (`build_datasets/`)

Generates fine-grained retrieval training data from images via an OpenRouter VLM API. Outputs flat contrastive JSONL records consumed directly by `train/finetune.py` and `train/analyze_hard_negatives.py`. Config and prompts are in `DEFINE.py`; the builder script is `vlm_dataset_builder.py`.

Expected flat training record format:

```json
{
  "image_path": "build_datasets/data/VG_100K/107914.jpg",
  "positives": ["..."],
  "hard_negatives": ["..."]
}
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
- Text input spec: `(1, 77)` int32

## Key Gotchas

- `reparameterize_model()` from `timm.utils` must be called before ONNX export. For QAT, it is called automatically inside `wrap_model_for_qat()` — do not call it again afterward (would break fake quant nodes).
- QAT checkpoint loading: the model must be reparameterized before QuantSim is created so the state dict key names match. `_load_clip()` handles this — regular checkpoints are loaded before reparameterize; QAT checkpoints are loaded after.
- The `OpenClipTextEncoder` wrapper zeros tokens after the EOS position (argmax) to ensure consistent input regardless of tokenizer padding style. With `--max-text-len N < 77`, it additionally slices `token_ids[:, :N]` and truncates `attn_mask` to `(N, N)` — the ONNX graph still accepts `(1, 77)` externally (fixed by competition spec) but computes attention over only N positions, reducing latency on short texts.
- QAI Hub auto-converts to fp16 during compilation.
- The tokenizer is always `open_clip.get_tokenizer("ViT-B-32")` regardless of which MobileCLIP2 variant is used — all variants share the same tokenizer.

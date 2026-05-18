# LPCV 2026 Track 1 — Efficient AI

This project targets image-to-text retrieval on Qualcomm XR2 Gen 2 devices. The main goal is to maximize Recall@K. The core model is `MobileCLIP2-B`. Local evaluation supports both PyTorch and ONNX Runtime, while the deployment path is ONNX export + QAI Hub compilation + remote device inference.

This document is organized for reproducibility from a clean machine, starting from the Conda environment.

## 1. Environment Setup

### 1.1 Requirements

- Linux
- A Conda environment with Python 3.10 or 3.11
- NVIDIA GPU + CUDA recommended for training and MLP reconstruction
- `conda` installed
- A valid QAI Hub account and API token for compilation and remote evaluation
- An OpenRouter-compatible API key if you want to build training data

### 1.2 Create a Conda Environment

Run from the repository root:

```bash
conda create -n lpcv python=3.10 
conda activate lpcv
```

### 1.3 Install PyTorch

Install the PyTorch build that matches your CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you use CPU or another CUDA version, replace the command with the appropriate official install command.

### 1.4 Install Project Dependencies

```bash
pip install -r requirements.txt
```

Notes:

- `requirements.txt` includes `aimet-torch` for the optional QAT path. QAT support is kept in the repository, but it was not part of the final solution and is not guaranteed to work cleanly.


### 1.5 Pre-download the Hugging Face Tokenizer

Several code paths in this project use:

```python
CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
```

That means the tokenizer must already exist in the local Hugging Face cache before first use.

Recommended:

```bash
hf download openai/clip-vit-base-patch32
```

### 1.6 Configure QAI Hub

Only required if you want to compile ONNX models or run remote evaluation:

```bash
qai-hub configure --api_token <YOUR_QAI_HUB_TOKEN>
```

## 2. Model Downloads

This project uses two kinds of model assets:

- the base pretrained model: `MobileCLIP2-B`
- the current best exported model: `jn12/2026LPCV-Track1-MobileCLIP2-B-Best`

### 2.1 Download the Base Pretrained MobileCLIP2-B Model

The code loads `MobileCLIP2-B` through `open_clip`. If the local cache does not exist yet, the pretrained weights will be downloaded automatically on first use.

If you want to prepare the cache explicitly, first complete Section 1.5 for the tokenizer, then run a local evaluation once:

```bash
python pipeline/eval_local.py --model-name MobileCLIP2-B --k 10
```

This will trigger the `open_clip` download and cache the required pretrained weights for `MobileCLIP2-B`.

### 2.2 Download the Current Best Model

Current best model repository:

```text
https://huggingface.co/jn12/2026LPCV-Track1-MobileCLIP2-B-Best
```

This repository currently provides exported ONNX files. The main files include:

- `image_encoder.onnx`
- `image_encoder.onnx.data`
- `text_encoder.onnx`
- `text_encoder.onnx.data`

Recommended local download path:


```bash
hf download jn12/2026LPCV-Track1-MobileCLIP2-B-Best \
  --local-dir ./pretrained/2026LPCV-Track1-MobileCLIP2-B-Best
```

The downloaded directory should look like:

```text
pretrained/2026LPCV-Track1-MobileCLIP2-B-Best/
├── image_encoder.onnx
├── image_encoder.onnx.data
├── text_encoder.onnx
└── text_encoder.onnx.data
```

You can use it directly for local ONNX evaluation:

```bash
python pipeline/eval_local.py \
  --onnx-dir ./pretrained/2026LPCV-Track1-MobileCLIP2-B-Best \
  --k 10
```

## 3. Training Data Download

The full fine-tuning dataset is available on Hugging Face Datasets:

```text
https://huggingface.co/datasets/jn12/VG100K4CL
```

Repository:


```bash
hf download jn12/VG100K4CL \
  --repo-type dataset \
  --local-dir ./data/VG100K4CL
```

`VG100K4CL` is the complete fine-tuning dataset used in this project. It includes the JSONL annotations needed for training.

If you want to unpack the image shards into a normal image folder, run:

```bash
python build_datasets/pack_dataset.py unpack \
  --src ./data/VG100K4CL \
  --dst ./build_datasets/data/VG_100K
```



## 4. Repository and Data Layout

Run all commands from the repository root by default:

```bash
cd /path/to/LPCV-Track1-EfficientAI-main
```

The sample evaluation set is already included under `sample_data/`:

```text
sample_data/
├── images/
├── img_list.csv
└── txt_list.csv
```

Where:

- `img_list.csv` columns are `Image_names`, `Text_nums`
- `txt_list.csv` columns are `Text_nums`, `Unique_Texts`

Training data uses a flat JSONL format:

```json
{
  "image_path": "build_datasets/data/VG_100K/107914.jpg",
  "positives": ["positive text 1", "positive text 2"],
  "hard_negatives": ["negative text 1", "negative text 2"]
}
```

## 5. Minimal Reproducible Pipeline

If you only want to verify that the project works end to end, run these steps first.

### 5.1 Local PyTorch Evaluation

```bash
python pipeline/eval_local.py \
  --model-name MobileCLIP2-B \
  --root-dir ./sample_data \
  --image-to-text-csv ./sample_data/img_list.csv \
  --textnums-to-texts-csv ./sample_data/txt_list.csv \
  --k 10
```

This loads pretrained `MobileCLIP2-B` and computes `image_to_text_recall@10` on the sample set locally.

### 5.2 Export ONNX

```bash
python pipeline/export_onnx.py --model-name MobileCLIP2-B
```

Output directory:

```text
exported_MobileCLIP2-B_onnx/
├── image_encoder.onnx
└── text_encoder.onnx
```

### 5.3 Local ONNX Evaluation

```bash
python pipeline/eval_local.py \
  --onnx-dir exported_MobileCLIP2-B_onnx \
  --root-dir ./sample_data \
  --image-to-text-csv ./sample_data/img_list.csv \
  --textnums-to-texts-csv ./sample_data/txt_list.csv \
  --k 10
```

This verifies that the exported ONNX path works locally.

Both local evaluation modes are supported:

- without `--onnx-dir`: PyTorch path
- with `--onnx-dir`: ONNX Runtime path

## 6. Remote Device Compilation and Evaluation

### 6.1 Compile and Submit Profiling Jobs

Make sure `qai-hub configure` is already done, then run:

```bash
python pipeline/compile_and_profile.py --model-name MobileCLIP2-B
```

The script will:

- read `exported_MobileCLIP2-B_onnx/`
- submit image and text compilation jobs to QAI Hub
- automatically submit profile jobs after compilation finishes

The target device is fixed to:

- `XR2 Gen 2 (Proxy)`

Compilation settings are fixed to:

- runtime: `qnn_dlc`
- image input: `(1, 3, 224, 224)` float32
- text input: `(1, 77)` int32

### 6.2 Remote Evaluation

`pipeline/eval_remote.py` supports three modes.

Mode A: upload local `sample_data`, then run inference

```bash
python pipeline/eval_remote.py \
  --upload-dataset \
  --model-name MobileCLIP2-B \
  --image-compiled-id <IMAGE_COMPILED_JOB_ID> \
  --text-compiled-id <TEXT_COMPILED_JOB_ID>
```

Mode B: reuse existing dataset IDs, then run inference

```bash
python pipeline/eval_remote.py \
  --model-name MobileCLIP2-B \
  --image-compiled-id <IMAGE_COMPILED_JOB_ID> \
  --text-compiled-id <TEXT_COMPILED_JOB_ID> \
  --image-dataset-id <IMAGE_DATASET_ID> \
  --text-dataset-id <TEXT_DATASET_ID>
```

Mode C: reuse completed inference jobs

```bash
python pipeline/eval_remote.py \
  --image-inference-id <IMAGE_INFERENCE_JOB_ID> \
  --text-inference-id <TEXT_INFERENCE_JOB_ID> \
  --k 10
```

Notes:

- If `--image-dataset-id` and `--text-dataset-id` are omitted, the code falls back to built-in default dataset IDs.
- Mode A is the safest choice for first-time reproduction.

## 7. Training Reproduction

### 7.1 Training Input

Default training file:

```text
./build_datasets/data/dataset_verify_contrastive.jsonl
```

In practice, the final training data used by this project came from Path B below, i.e. the VisualGenome-based pipeline. If you use your own data, keep the JSONL structure identical to the format shown above.

### 7.2 Single-GPU Training

```bash
python train/finetune.py \
  --jsonl-path ./build_datasets/data/dataset_verify_contrastive.jsonl \
  --model-name MobileCLIP2-B \
  --gpu-ids 0 \
  --batch-size 256 \
  --accum-freq 4 \
  --epochs 20 \
  --lr 1e-6 \
  --weight-decay 0.2 \
  --loss-type clip \
  --num-hard-negatives 4
```

### 7.3 Multi-GPU DDP Training

```bash
OMP_NUM_THREADS=8 torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  train/finetune.py \
  --jsonl-path ./build_datasets/data/dataset_verify_contrastive.jsonl \
  --model-name MobileCLIP2-B \
  --gpu-ids 0,1 \
  --batch-size 256 \
  --accum-freq 4 \
  --epochs 20 \
  --lr 1e-6 \
  --weight-decay 0.2 \
  --loss-type clip \
  --num-hard-negatives 4
```

Notes:

- `--batch-size` is per GPU.
- Effective global batch size = `batch_size × world_size × accum_freq`.
- If your machine has NCCL P2P issues, use `NCCL_P2P_DISABLE=1` as shown in `script_ft.sh`.

### 7.4 Resume from a Checkpoint

```bash
python train/finetune.py \
  --jsonl-path ./build_datasets/data/dataset_verify_contrastive.jsonl \
  --model-name MobileCLIP2-B \
  --gpu-ids 0 \
  --resume ./checkpoints/<run_name>/checkpoint_latest_epoch_05.pt
```

Note:

- Resume support exists, but it currently appears to have edge-case bugs.
- Treat resumed runs as best-effort only; correctness is not guaranteed without manual verification.

### 7.5 Training Outputs

Each run writes to:

```text
checkpoints/<model>__<config>__<timestamp>/
```

Typical contents include:

- `train.log`
- `metrics.csv`
- `metrics.jsonl`
- `training_curves.png`
- `run_config.json`
- `checkpoint_latest_epoch_*.pt`
- TensorBoard logs

Launch TensorBoard:

```bash
tensorboard --logdir checkpoints
```

## 8. Export and Evaluate Trained Checkpoints

### 8.1 Export a Specific Checkpoint

```bash
python pipeline/export_onnx.py \
  --model-name MobileCLIP2-B \
  --checkpoint-path ./checkpoints/<run_name>/checkpoint_epoch_020.pt \
  --output-postfix _ft
```

Example output directory:

```text
exported_MobileCLIP2-B_ft_onnx/
```

### 8.2 Local Evaluation of a Trained PyTorch Checkpoint

```bash
python pipeline/eval_local.py \
  --model-name MobileCLIP2-B \
  --checkpoint-path ./checkpoints/<run_name>/checkpoint_epoch_020.pt \
  --k 10
```

### 8.3 Local Evaluation of Exported ONNX

```bash
python pipeline/eval_local.py \
  --onnx-dir ./exported_MobileCLIP2-B_ft_onnx \
  --k 10
```

## 9. Dataset Building Reproduction

This repository contains two dataset-building paths:

- a direct image-to-JSONL builder based on `build_datasets/vlm_dataset_builder.py`
- a VisualGenome-based pipeline driven by `build_datasets/run_vg.sh`

The final dataset used in this project came from Path B, i.e. the VisualGenome-based pipeline driven by `build_datasets/run_vg.sh`.

### 9.1 Path A: Direct Builder

If you need to generate training JSONL files from images directly, use `build_datasets/vlm_dataset_builder.py`.

This pipeline produces two annotation variants:

- raw outputs
- cleaned / verified outputs

### 9.2 Environment Variable

```bash
export OPENROUTER_API_KEY=<YOUR_API_KEY>
```

### 9.3 Build Command

```bash
python build_datasets/vlm_dataset_builder.py \
  --image_dir /path/to/images \
  --output_dir ./build_datasets/data/my_build \
  --base_url https://openrouter.ai/api/v1 \
  --api_key "$OPENROUTER_API_KEY" \
  --model google/gemini-3.1-pro-preview \
  --verify \
  --verify_model google/gemini-3.1-flash-lite-preview \
  --detail high \
  --max_workers 4
```

Output files:

```text
build_datasets/data/my_build/
├── dataset_raw.jsonl
├── dataset_raw_contrastive.jsonl
├── dataset_verify.jsonl
├── dataset_verify_contrastive.jsonl
└── build.log
```

For Path A, the important files are usually:

- `dataset_verify.jsonl`
- `dataset_verify_contrastive.jsonl`

Path A can produce `dataset_verify_contrastive.jsonl`, but it was not the final dataset-building path used for the main training results in this repository.

If you only want to regenerate the contrastive JSONL from existing annotations:

```bash
python build_datasets/vlm_dataset_builder.py \
  --image_dir /path/to/images \
  --output_dir ./build_datasets/data/my_build \
  --base_url https://openrouter.ai/api/v1 \
  --api_key "$OPENROUTER_API_KEY" \
  --model google/gemini-3.1-pro-preview \
  --reconvert_raw
```

### 9.4 Path B: VisualGenome Pipeline

This is the dataset-building path used for the final training data in this project.

The pipeline is defined in:

- `build_datasets/run_vg.sh`

This path requires downloading the VisualGenome dataset first:

```bash
hf download jn12/VisualGenome \
  --repo-type dataset \
  --local-dir ./build_datasets/data/VisualGenome
```

Then run the pipeline steps referenced by `run_vg.sh`:

```bash
python build_datasets/setup_vg_data.py
python build_datasets/vg_integrate.py
python build_datasets/vg_llm_annotate.py \
  --base_url http://localhost:8000/v1 \
  --model google/gemma-4-31B-it \
  --max_workers 100
```

An alternative annotation command in the same script is:

```bash
python build_datasets/vg_llm_annotate.py \
  --base_url http://localhost:8000/v1 \
  --model google/gemma-4-26B-A4B-it \
  --max_workers 240
```

At a high level, this path does the following:

1. unpack VisualGenome into the local working directories expected by the scripts
2. integrate VisualGenome annotations into an intermediate per-image JSONL
3. run VLM-based annotation to produce contrastive training data


## 10. MLP Reconstruction Reproduction

This pipeline replaces `GELU` with `ReLU` in MLP blocks and restores accuracy with block-wise distillation.

### 10.1 Run from the Pretrained Model

```bash
python mlp_reconstruction/run.py \
  --model-name MobileCLIP2-B \
  --gpu-id 0 \
  --n-calib 1024 \
  --n-iters 20000
```

### 10.2 Run from a Fine-tuned Checkpoint

```bash
python mlp_reconstruction/run.py \
  --model-name MobileCLIP2-B \
  --gpu-id 0 \
  --checkpoint-path ./checkpoints/<ft_run>/checkpoint_latest_epoch_100.pt
```

### 10.3 Evaluate the Reconstructed Model

```bash
python pipeline/eval_local.py \
  --model-name MobileCLIP2-B \
  --checkpoint-path ./checkpoints/<mr_run>/mlp_relu.pt \
  --k 10
```

### 10.4 Export the Reconstructed Model to ONNX

```bash
python pipeline/export_onnx.py \
  --model-name MobileCLIP2-B \
  --checkpoint-path ./checkpoints/<mr_run>/mlp_relu.pt \
  --output-postfix _mr
```


## 11. Common Optional Settings

### 11.1 Shorten Internal Text Length for Lower Latency

You can restrict the internal text length during ONNX export:

```bash
python pipeline/export_onnx.py \
  --model-name MobileCLIP2-B \
  --max-text-len 40
```

Notes:

- The ONNX external input shape remains `(1, 77)`
- Internally, the model truncates attention to the first `40` tokens
- This usually reduces latency for short-text scenarios

### 11.2 Image Rings

The project supports removing ViT patch rings with `--resize` and `--crop`:

```bash
python pipeline/export_onnx.py --model-name MobileCLIP2-B --resize 1 --crop 1
python pipeline/eval_local.py --model-name MobileCLIP2-B --resize 1 --crop 1
```

The corresponding training arguments are:

```bash
--resize-rings N
--crop-rings N
```

## 12. Important Implementation Constraints

- Image preprocessing is `resize to 224x224 + divide by 255`, with no ImageNet mean/std normalization. This differs from common CLIP setups.
- The tokenizer is `CLIPTokenizer` from `openai/clip-vit-base-patch32`, and it must already exist in the local cache.
- Regular checkpoints and MLP reconstruction checkpoints are auto-detected by `utils/clip_utils.py::_load_clip()`.
- ONNX compilation targets `XR2 Gen 2 (Proxy)` with runtime `qnn_dlc`.
- QAT-related code still exists in the repository, but we did not use quantization in the final result because the relevant QAI hardware/operator path was not favorable for this model; treat that path as experimental only.

## 13. Directory Overview

- `pipeline/`: ONNX export, QAI Hub compilation, local and remote evaluation
- `train/`: fine-tuning, DDP, loss, metrics, checkpoints
- `mlp_reconstruction/`: block-wise GELU → ReLU reconstruction
- `build_datasets/`: VLM-based annotation and contrastive JSONL export
- `utils/`: model loading, preprocessing, retrieval metrics
- `sample_data/`: sample evaluation dataset included in the repo

## 14. Key Gotchas

- `eval_remote.py` also depends on a locally cached Hugging Face tokenizer for text tokenization. Even though inference runs on QAI Hub, the text inputs are still tokenized locally before upload, so “remote evaluation” does not mean the local tokenizer resources are unnecessary.
- `pipeline/eval_local.py --onnx-dir ...` uses the ONNXRuntime path, while `pipeline/eval_local.py` without `--onnx-dir` uses the default PyTorch path. Both paths use the same sample set, but the underlying encoding implementations differ. When debugging accuracy differences, first make sure you are comparing the same checkpoint, the same `--max-text-len`, and the same resize/crop rings configuration.
- The training-time `--resize-rings` / `--crop-rings` options and the export/evaluation-time `--resize` / `--crop` options change the effective visual encoder input structure. This information is stored in checkpoints, and `_load_clip()` will try to restore it automatically. If you manually override these options, you may end up with a `pos_embed` structure that no longer matches training.

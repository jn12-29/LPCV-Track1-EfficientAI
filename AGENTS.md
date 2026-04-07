# AGENTS.md

LPCVC 2026 Track 1 - Image-to-Text Retrieval competition submission. Uses MobileCLIP2 models exported to ONNX, compiled for Qualcomm XR2 Gen 2 via QAI Hub.

## Setup

```bash
pip install -r requirements.txt
qai-hub configure  # requires API token
```

Environment variables (set before running):
```bash
export CUDA_VISIBLE_DEVICES=<gpu_id>
export HF_HOME=/mnt/sada1/data
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONNOUSERSITE=1
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

## Pipeline (run in order)

```bash
# 1. Export ONNX (fp32)
python mobileclipv2.py --model-name MobileCLIP2-S0

# 2. Compile for XR2 Gen 2 + profile
python compile_and_profile.py --model-name MobileCLIP2-S0

# 3. Evaluate
python eval_local.py --model-name MobileCLIP2-S0 --k 10        # local torch
python eval_onnx_local.py --model-name MobileCLIP2-S0 --k 10   # local onnx
python eval_remote.py --image-compiled-id <id> --text-compiled-id <id> --k 10  # QAI Hub
```

Available models: `MobileCLIP2-S0`, `MobileCLIP2-S2`, `MobileCLIP2-S3`

## Key Constants

| Parameter | Value |
|-----------|-------|
| Image input | `(1, 3, 224, 224)` float32, normalized to [0,1] |
| Text tokens | `(1, 77)` int64 |
| Target device | `"XR2 Gen 2 (Proxy)"` |
| Runtime | `qnn_dlc` |
| Metric | Recall@K |

## Important Details

- **Preprocessing**: Images resized to 224x224, divided by 255. NO ImageNet mean/std normalization (differs from CLIP defaults).
- **Tokenizer**: Uses `open_clip.get_tokenizer("ViT-B-32")` for all MobileCLIP2 variants.
- **Model reparameterization**: `timm.utils.reparameterize_model()` called before export.
- **fp16**: QAI Hub auto-converts to fp16; `mobileclipv2_fp16.py` is redundant.
- **QAI Hub sharing**: Compile jobs auto-share results with `lowpowervision@gmail.com`.
- **Dataset IDs**: Hardcoded in `Token.py` and `eval_remote.py`:
  - Image dataset: `d2qe36jl2`
  - Text dataset: `d95k6jwm9`

## Dataset Layout

```
data/
├── images/          # .jpg, .png, .jpeg, .webp
├── img_list.csv     # Image_names, Text_nums (semicolon-separated)
└── txt_list.csv     # Text_nums, Unique_Texts
```

## Output Directories

- `exported_{model_name}_onnx/` - Exported ONNX models

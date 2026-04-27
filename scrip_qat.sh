# ── QAT (Quantization-Aware Training) ─────────────────────────────────────────

# QAT from pretrained weights, W8A8
python train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 2 --batch-size 64 --accum-freq 128 --epochs 100 --lr 5e-7 \
    --qat-enabled --qat-weight-bw 8 --qat-act-bw 8 --qat-calib-samples 1024

torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 2,3 --batch-size 128 --accum-freq 64 --epochs 100 --lr 1e-6 --weight-decay 0.2 \
    --loss-type clip --num-hard-negatives 4 \
    --qat-enabled --qat-weight-bw 8 --qat-act-bw 8 --qat-calib-samples 1024  --export-onnx

# QAT from regular finetune checkpoint, W8A16
CUDA_VISIBLE_DEVICES=5 python train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 5 --batch-size 32 --accum-freq 8 --epochs 5 --lr 5e-7 \
    --resume ./checkpoints/MobileCLIP2-B__bs128_ep200_lr1e-06_wd0.2_acc32_hn4_hnw1_seed0__20260420_000848/checkpoint_epoch_075.pt \
    --qat-enabled --qat-weight-bw 8 --qat-act-bw 16 --export-onnx

# QAT resume (auto-detects QAT checkpoint, skips calibration)
CUDA_VISIBLE_DEVICES=5 python train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 5 --batch-size 32 --accum-freq 8 --epochs 5 --lr 5e-7 \
    --resume ./checkpoints/<qat_run>/checkpoint_latest_epoch_03.pt \
    --qat-enabled --qat-weight-bw 8 --qat-act-bw 16

# Export ONNX from QAT checkpoint
python pipeline/export_onnx.py --model-name MobileCLIP2-B \
    --checkpoint-path ./checkpoints/<qat_run>/MobileCLIP2-B_finetuned.pt \
    --output-postfix _qat_w8a8
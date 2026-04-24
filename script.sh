# Run these commands from the repository root.
# Training JSONL is expected to use the flat contrastive format produced by build_datasets/.

# netron
npm install -g netron

netron checkpoints/MobileCLIP2-B__bs128_ep100_lr1e-06_wd0.2_acc64_hn4_hnw1_seed0__20260422_185248/onnx_epoch_005/image_encoder.onnx


# export onnx
python pipeline/export_onnx.py --model-name MobileCLIP2-S0
python pipeline/export_onnx.py --model-name MobileCLIP2-S2
python pipeline/export_onnx.py --model-name MobileCLIP2-B

python pipeline/export_onnx.py --model-name MobileCLIP2-S2 --checkpoint-path ./checkpoints/MobileCLIP2-S2__bs256_ep100_lr1e-06_wd0.2_acc32_hn4_hnw1_seed0__20260421_205417/checkpoint_latest_epoch_100.pt --output-postfix _260423  

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs128_ep200_lr1e-06_wd0.2_acc32_hn4_hnw1_seed0__20260420_000848/checkpoint_epoch_075.pt --output-postfix _260420

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs128_ep200_lr1e-06_wd0.2_acc32_hn4_hnw1_seed0__20260420_000848/checkpoint_epoch_060.pt --output-postfix _260421

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc30_hn4_hnw1_seed0__20260424_012159/checkpoint_epoch_120.pt --output-postfix _260423

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc30_hn4_hnw1_seed0__20260424_012159/checkpoint_epoch_100.pt --output-postfix _260424_0 --max-text-len 40

# compile and profile
python pipeline/compile_and_profile.py --model-name MobileCLIP2-S2
python pipeline/compile_and_profile.py --model-name MobileCLIP2-B

python pipeline/compile_and_profile.py --model-name MobileCLIP2-S2_260418
python pipeline/compile_and_profile.py --model-name MobileCLIP2-B_260424_0

# eval local (torch)
python pipeline/eval_local.py --model-name MobileCLIP2-S0 --k 10
python pipeline/eval_local.py --model-name MobileCLIP2-S2 --k 10

python pipeline/eval_local.py --model-name MobileCLIP2-S2 --k 10 --checkpoint-path ./checkpoints/MobileCLIP2-S2__bs256_ep400_lr1e-06_wd0.2_acc32_hn4_hnw0.5_seed0__20260417_184009/checkpoint_latest_epoch_126.pt

CUDA_VISIBLE_DEVICES=6 python pipeline/eval_local.py --model-name MobileCLIP2-B --k 10 --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc30_hn4_hnw1_seed0__20260424_012159/checkpoint_epoch_100.pt

# eval local (onnx)
python pipeline/eval_local.py --model-name MobileCLIP2-S2_260418

# eval remote (Mode A: upload + infer)
python pipeline/eval_remote.py --upload-dataset \
    --image-compiled-id <image_compile_job_id> --text-compiled-id <text_compile_job_id>

# eval remote (Mode B: existing dataset)
python pipeline/eval_remote.py \
    --image-compiled-id <image_compile_job_id> --text-compiled-id <text_compile_job_id>

python pipeline/eval_remote.py --model-name MobileCLIP2-S2_260418 \
    --image-compiled-id jpx7z411g --text-compiled-id j5mwlmzwp

python pipeline/eval_remote.py --model-name MobileCLIP2-S2 \
    --image-compiled-id j57je47v5 --text-compiled-id jp27rwvr5

# eval remote (Mode C: reuse inference)
python pipeline/eval_remote.py \
    --image-inference-id <image_inference_job_id> --text-inference-id <text_inference_job_id>

python pipeline/eval_remote.py \
    --image-inference-id jp4xy1xv5 --text-inference-id j57je4jl5 --k 7

python pipeline/eval_remote.py \
    --image-inference-id jper77m7g --text-inference-id jgj0rrn7p --k 7

# fine-tune
python train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-S2 --gpu-ids 5 --batch-size 256 --accum-freq 50 --epochs 20 --lr 1e-6

# fine-tune (multi-GPU DDP)
torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-S2 --gpu-ids 2,3 --batch-size 256 --accum-freq 32  --epochs 100 --lr 1e-6 --weight-decay 0.2 --loss-type clip --num-hard-negatives 4

torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29501 train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-S2 --gpu-ids 2,3,5,6 --batch-size 128 --accum-freq 32  --epochs 100 --lr 1e-6 --weight-decay 0.2 --loss-type clip --num-hard-negatives 4

torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29501 train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 1,2,3,5 --batch-size 256 --accum-freq 30 --epochs 200 --lr 1e-6 --weight-decay 0.2 \
    --loss-type clip --num-hard-negatives 4

python train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 2 --batch-size 64 --accum-freq 128 --epochs 100 --lr 1e-6 --no-amp # no amp will make train very slow from 180 samples/s to 40 samples/s per gpu

# fine-tune with SigLIP loss (hard negatives absorbed into sigmoid matrix)
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29501 train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-S2 --gpu-ids 2,3,5,6 --batch-size 256 --accum-freq 16 --epochs 100 --lr 1e-6 --weight-decay 0.2 \
    --loss-type siglip --num-hard-negatives 4

# analyze hard negatives
python train/analyze_hard_negatives.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-S2

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

# ── MLP Reconstruction (GELU → ReLU + distillation) ───────────────────────────

# Run full reconstruction for S0 pretrained (~2 hours, 32 blocks × 20000 iters)
python mlp_reconstruction/run.py train \
    --model-name MobileCLIP2-S0 \
    --n-calib 1024 --n-iters 20000 \
    --output checkpoints/MobileCLIP2-S0_mlp_relu.pt \
    --device cuda --log-every 500

# Run on top of a fine-tuned checkpoint
python mlp_reconstruction/run.py train \
    --model-name MobileCLIP2-S2 \
    --checkpoint-path ./checkpoints/MobileCLIP2-S2__bs256_ep100_lr1e-06_wd0.2_acc32_hn4_hnw1_seed0__20260421_205417/checkpoint_latest_epoch_100.pt \
    --n-calib 1024 --n-iters 20000 \
    --output checkpoints/MobileCLIP2-S2_mlp_relu.pt \
    --device cuda

# Resume after crash (load .tmp checkpoint, restart from a specific block)
python mlp_reconstruction/run.py train \
    --model-name MobileCLIP2-S0 \
    --resume-from checkpoints/MobileCLIP2-S0_mlp_relu.pt.tmp \
    --skip-to visual[s1b0] \
    --output checkpoints/MobileCLIP2-S0_mlp_relu.pt \
    --device cuda

# Eval reconstructed model on sample_data (torch)
python mlp_reconstruction/run.py eval \
    --checkpoint-path checkpoints/MobileCLIP2-S0_mlp_relu.pt --k 10

# Export reconstructed model to ONNX (first load checkpoint into a temp script,
# or use eval_local.py after patching _load_clip to call load_reconstructed_model)
# Typical approach: load via load_reconstructed_model(), then export_encoders_to_onnx()

# Smoke test (64 samples, 200 iters, only first 2 blocks)
python mlp_reconstruction/run.py train \
    --model-name MobileCLIP2-S0 \
    --n-calib 64 --n-iters 200 --log-every 50 \
    --output /tmp/smoke_test.pt --device cuda \
    --skip-to visual[s0b1]


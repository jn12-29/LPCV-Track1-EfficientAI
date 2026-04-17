# Run these commands from the repository root.
# Training JSONL is expected to use the flat contrastive format produced by build_datasets/.

# export onnx
python pipeline/export_onnx.py --model-name MobileCLIP2-S0
python pipeline/export_onnx.py --model-name MobileCLIP2-S2
python pipeline/export_onnx.py --model-name MobileCLIP2-S3

python pipeline/export_onnx.py --model-name MobileCLIP2-S2 --checkpoint-path ./checkpoints/MobileCLIP2-S2__bs256_ep20_lr1e-06_wd0.2_acc32_hn4_hnw0.5_seed0__20260417_174031/MobileCLIP2-S2_finetuned.pt --output-postfix _bs256_ep20_lr1e-06_wd0.2_acc32_hn4_hnw0.5_seed0


# compile and profile
python pipeline/compile_and_profile.py --model-name MobileCLIP2-S2
python pipeline/compile_and_profile.py --model-name MobileCLIP2-S3

python pipeline/compile_and_profile.py --model-name MobileCLIP2-S2_bs256_ep20_lr1e-06_wd0.2_acc32_hn4_hnw0.5_seed0

# eval local (torch)
python pipeline/eval_local.py --model-name MobileCLIP2-S0 --k 10
python pipeline/eval_local.py --model-name MobileCLIP2-S2 --k 10

python pipeline/eval_local.py --model-name MobileCLIP2-S2 --k 10 --checkpoint-path ./checkpoints/MobileCLIP2-S2__bs256_ep20_lr1e-06_wd0.2_acc32_hn4_hnw0.5_seed0__20260417_174031/MobileCLIP2-S2_finetuned.pt

# eval remote (Mode A: upload + infer)
python pipeline/eval_remote.py --upload-dataset \
    --image-compiled-id <image_compile_job_id> --text-compiled-id <text_compile_job_id>

# eval remote (Mode B: existing dataset)
python pipeline/eval_remote.py \
    --image-compiled-id <image_compile_job_id> --text-compiled-id <text_compile_job_id>

python pipeline/eval_remote.py --model-name MobileCLIP2-S2_bs256_ep20_lr1e-06_wd0.2_acc32_hn4_hnw0.5_seed0 \
    --image-compiled-id jgj08q2xp --text-compiled-id jpernyw1g

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
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-S2 --gpu-ids 5,7 --batch-size 256 --accum-freq 32  --epochs 400 --lr 1e-6

# analyze hard negatives
python train/analyze_hard_negatives.py \
    --jsonl-path ./build_datasets/data/VG_100K_GEMINI31FLASHLITE_NEW/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-S0

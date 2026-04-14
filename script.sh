# Run these commands from the repository root.
# Training JSONL is expected to use the flat contrastive format produced by build_datasets/.

# export onnx
python pipeline/export_onnx.py --model-name MobileCLIP2-S0
python pipeline/export_onnx.py --model-name MobileCLIP2-S2
python pipeline/export_onnx.py --model-name MobileCLIP2-S3

python pipeline/export_onnx.py --model-name MobileCLIP2-S2 --checkpoint-path checkpoints/MobileCLIP2-S2__bs256_ep20_lr1e-05_wd0.2_acc1_hn4_hnw0.5_seed0__20260415_005500/MobileCLIP2-S2_finetuned.pt --output-postfix _bs256_ep20_lr1e-05_wd0.2_acc1_hn4_hnw0.5_seed0

# compile and profile
python pipeline/compile_and_profile.py --model-name MobileCLIP2-S2
python pipeline/compile_and_profile.py --model-name MobileCLIP2-S3

python pipeline/compile_and_profile.py --model-name MobileCLIP2-S2_bs256_ep20_lr1e-05_wd0.2_acc1_hn4_hnw0.5_seed0

# eval local (torch)
python pipeline/eval_local.py --model-name MobileCLIP2-S0 --k 10
python pipeline/eval_local.py --model-name MobileCLIP2-S2 --k 10

python pipeline/eval_local.py --model-name MobileCLIP2-S2 --k 10 --checkpoint-path ./checkpoints/MobileCLIP2-S2_finetuned.pt

# eval remote (Mode A: upload + infer)
python pipeline/eval_remote.py --upload-dataset \
    --image-compiled-id <image_compile_job_id> --text-compiled-id <text_compile_job_id>

# eval remote (Mode B: existing dataset)
python pipeline/eval_remote.py \
    --image-compiled-id <image_compile_job_id> --text-compiled-id <text_compile_job_id>

python pipeline/eval_remote.py --model-name MobileCLIP2-S2_bs256_ep20_lr1e-05_wd0.2_acc1_hn4_hnw0.5_seed0 \
    --image-compiled-id j5q7ly0mg --text-compiled-id jgl0yx4lg

# eval remote (Mode C: reuse inference)
python pipeline/eval_remote.py \
    --image-inference-id <image_inference_job_id> --text-inference-id <text_inference_job_id>

# fine-tune
python train/finetune.py \
    --jsonl-path ./build_datasets/data/VG_100K_GEMINI31FLASHLITE_NEW/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-S2 --gpu-ids 0 --batch-size 256 --epochs 20

# fine-tune (multi-GPU DDP)
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29501 train/finetune.py \
    --jsonl-path ./build_datasets/data/VG_100K_GEMINI31FLASHLITE_NEW/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-S2 --gpu-ids 0,1,2,3 --batch-size 64 --epochs 20

# analyze hard negatives
python train/analyze_hard_negatives.py \
    --jsonl-path ./build_datasets/data/VG_100K_GEMINI31FLASHLITE_NEW/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-S0

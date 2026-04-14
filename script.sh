# export onnx
python pipeline/export_onnx.py --model-name MobileCLIP2-S0
python pipeline/export_onnx.py --model-name MobileCLIP2-S2
python pipeline/export_onnx.py --model-name MobileCLIP2-S3

# compile and profile
python pipeline/compile_and_profile.py --model-name MobileCLIP2-S2
python pipeline/compile_and_profile.py --model-name MobileCLIP2-S3

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

# eval remote (Mode C: reuse inference)
python pipeline/eval_remote.py \
    --image-inference-id <image_inference_job_id> --text-inference-id <text_inference_job_id>

# fine-tune
CUDA_VISIBLE_DEVICES=0 python train/finetune.py \
    --jsonl-path ./build_datasets/data/VG_100K_GEMINI31FLASHLITE_NEW/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-S2 --batch-size 256 --epochs 20

# analyze hard negatives
python train/analyze_hard_negatives.py \
    --jsonl-path ./build_datasets/data/VG_100K_GEMINI31FLASHLITE_NEW/dataset_raw_contrastive.jsonl \
    --model-name MobileCLIP2-S0

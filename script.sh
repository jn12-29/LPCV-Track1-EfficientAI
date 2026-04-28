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

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc30_hn4_hnw1_seed0__20260424_012159/checkpoint_epoch_120.pt --output-postfix _260425_0 --max-text-len 40

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc60_hn4_hnw1_seed0__20260425_203708/checkpoint_epoch_065.pt --output-postfix _260425_1 --max-text-len 40

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs512_ep200_lr1e-06_wd0.2_acc30_hn4_hnw1_seed0__20260426_074645/checkpoint_epoch_110.pt --output-postfix _260426_0 --max-text-len 40

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__lr0.001_nit40000_bs32_nc4096_uniform_img_nostem__20260426_210359/mlp_relu.pt --output-postfix _260427_0 --max-text-len 40

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc27_hn4_hnw1_seed0__20260426_220300/checkpoint_epoch_130.pt --output-postfix _260427_1 --max-text-len 40

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc16_hn10_hnw1_seed0__20260427_200640/checkpoint_epoch_060.pt --output-postfix _260428_0 --max-text-len 40

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc16_hn10_hnw1_seed0__20260427_200640/checkpoint_epoch_200.pt --output-postfix _260428_1 --max-text-len 40

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc16_hn10_hnw1_seed0__20260427_200640/checkpoint_epoch_200.pt --output-postfix _260428_1_30 --max-text-len 30

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc16_hn10_hnw1_seed0__20260427_200640/checkpoint_epoch_200.pt --output-postfix _260428_1_40 --max-text-len 40

# compile and profile
python pipeline/compile_and_profile.py --model-name MobileCLIP2-S2
python pipeline/compile_and_profile.py --model-name MobileCLIP2-B

python pipeline/compile_and_profile.py --model-name MobileCLIP2-B_260428_1_40


# eval local (torch)
python pipeline/eval_local.py --model-name MobileCLIP2-S0 --k 10
python pipeline/eval_local.py --model-name MobileCLIP2-S2 --k 10

CUDA_VISIBLE_DEVICES=4 python pipeline/eval_local.py --model-name MobileCLIP2-B --k 10 --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc16_hn10_hnw1_seed0__20260427_200640/checkpoint_epoch_060.pt
CUDA_VISIBLE_DEVICES=4 python pipeline/eval_local.py --model-name MobileCLIP2-B --k 10 --checkpoint-path ./checkpoints/MobileCLIP2-B__lr0.001_nit20000_bs32_nc1024_magnitude_all__20260425_225309/mlp_relu.pt

# eval local (onnx)
python pipeline/eval_local.py --onnx-dir exported_MobileCLIP2-B_260428_1_30_onnx

# eval remote (Mode A: upload + infer)
python pipeline/eval_remote.py --upload-dataset --model-name MobileCLIP2-B-260428_1_40 \
    --image-compiled-id jgdr9welp --text-compiled-id jp1dl0ylp

# eval remote (Mode B: existing dataset)
python pipeline/eval_remote.py --model-name MobileCLIP2-B-260428_1_40 \
    --image-compiled-id j563j6305 --text-compiled-id j5q7k874g

python pipeline/eval_remote.py --model-name MobileCLIP2-S2 \
    --image-compiled-id j57je47v5 --text-compiled-id jp27rwvr5

# eval remote (Mode C: reuse inference)
python pipeline/eval_remote.py \
    --image-inference-id <image_inference_job_id> --text-inference-id <text_inference_job_id>

python pipeline/eval_remote.py \
    --image-inference-id jp4xy1xv5 --text-inference-id j57je4jl5 --k 7


# analyze hard negatives
python train/analyze_hard_negatives.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-S2

# analyze affect of crop and resize
python pipeline/sweep_image_rings.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc16_hn10_hnw1_seed0__20260427_200640/checkpoint_epoch_100.pt



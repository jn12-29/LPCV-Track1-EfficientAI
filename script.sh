# Run these commands from the repository root.

# download model

# openai/clip-vit-base-patch32

hf download openai/clip-vit-base-patch32

# Training JSONL is expected to use the flat contrastive format produced by build_datasets/.

# netron
npm install -g netron

netron checkpoints/MobileCLIP2-B__bs128_ep100_lr1e-06_wd0.2_acc64_hn4_hnw1_seed0__20260422_185248/onnx_epoch_005/image_encoder.onnx


# export onnx
python pipeline/export_onnx.py --model-name MobileCLIP2-B
python pipeline/export_onnx.py --model-name MobileCLIP2-B
python pipeline/export_onnx.py --model-name MobileCLIP2-B

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep100_lr1e-06_wd0.2_acc32_hn4_hnw1_seed0__20260421_205417/checkpoint_latest_epoch_100.pt --output-postfix _260423  

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs128_ep200_lr1e-06_wd0.2_acc32_hn4_hnw1_seed0__20260420_000848/checkpoint_epoch_075.pt --output-postfix _260420

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs128_ep200_lr1e-06_wd0.2_acc32_hn4_hnw1_seed0__20260420_000848/checkpoint_epoch_060.pt --output-postfix _260421

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc30_hn4_hnw1_seed0__20260424_012159/checkpoint_epoch_120.pt --output-postfix _260423

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc30_hn4_hnw1_seed0__20260424_012159/checkpoint_epoch_120.pt --output-postfix _260425_0 --max-text-len 40

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc60_hn4_hnw1_seed0__20260425_203708/checkpoint_epoch_065.pt --output-postfix _260425_1 --max-text-len 40

# 4/25/2026 23:39:52	****1do8g	****mvd1p	0.5681097764	14348	11589	2759
python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs512_ep200_lr1e-06_wd0.2_acc30_hn4_hnw1_seed0__20260426_074645/checkpoint_epoch_110.pt --output-postfix _260426_0 --max-text-len 40

# 4/26/2026 11:10:55	****vw46g	****9nzlp	0.6199234949	15183	12389	2794
python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__lr0.001_nit40000_bs32_nc4096_uniform_img_nostem__20260426_210359/mlp_relu.pt --output-postfix _260427_0 --max-text-len 40

# 4/26/2026 18:27:05	****vwomg	****92ex5	0.6087218171	15197	12417	2780
python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc27_hn4_hnw1_seed0__20260426_220300/checkpoint_epoch_130.pt --output-postfix _260427_1 --max-text-len 40

# 4/27/2026 11:17:38	****m7zvp	****6wxr5	0.6223890557	16424	13642	2782
python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc16_hn10_hnw1_seed0__20260427_200640/checkpoint_epoch_060.pt --output-postfix _260428_0 --max-text-len 40

# 4/27/2026 23:57:54	****j6305	****k874g	0.6247616664	16372	13668	2704
python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc16_hn10_hnw1_seed0__20260427_200640/checkpoint_epoch_200.pt --output-postfix _260428_1_40 --max-text-len 40

# 4/28/2026 11:41:01	****x0nkp	****qkwq5	0.6226595643	16342	13632	2710
python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc16_hn4_hnw1_seed0__20260428_082447/checkpoint_epoch_090.pt --output-postfix _260429_0 --max-text-len 40

# 4/28/2026 20:30:23	****m3jw5	****vzlkg	0.6079269746	13128	10437	2691
python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep100_lr1e-06_wd0.2_acc4_hn4_hnw1_seed0__20260429_024529/checkpoint_epoch_100.pt --output-postfix _260429_1 --max-text-len 40

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep100_lr1e-06_wd0.2_acc4_hn4_hnw1_seed0__20260429_075545/checkpoint_epoch_080.pt --output-postfix _260430_0 --max-text-len 40

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc16_hn4_hnw1_seed0__20260428_154224/checkpoint_epoch_180.pt --output-postfix _260430_1 --max-text-len 40 

python pipeline/export_onnx.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-05_wd0.2_acc4_hn4_hnw1_seed0__20260430_090832/checkpoint_epoch_110.pt --output-postfix _260501_0 --max-text-len 40

# compile and profile
python pipeline/compile_and_profile.py --model-name MobileCLIP2-B
python pipeline/compile_and_profile.py --model-name MobileCLIP2-B

python pipeline/compile_and_profile.py --model-name MobileCLIP2-B_260430_0


# eval local (torch)
python pipeline/eval_local.py --model-name MobileCLIP2-B --k 10
python pipeline/eval_local.py --model-name MobileCLIP2-B --k 10

CUDA_VISIBLE_DEVICES=4 python pipeline/eval_local.py --model-name MobileCLIP2-B --k 10 --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc16_hn10_hnw1_seed0__20260427_200640/checkpoint_epoch_060.pt
CUDA_VISIBLE_DEVICES=3 python pipeline/eval_local.py --model-name MobileCLIP2-B --k 10 --checkpoint-path ./checkpoints/MobileCLIP2-B__lr0.001_nit40000_bs32_nc4096_uniform_img_nostem__20260427_194021/mlp_relu.pt

# eval local (onnx)
python pipeline/eval_local.py --onnx-dir exported_MobileCLIP2-B_260430_0_onnx

# eval remote (Mode A: upload + infer)
python pipeline/eval_remote.py --upload-dataset --model-name MobileCLIP2-B-260428_1_40 \
    --image-compiled-id jgdr9welp --text-compiled-id jp1dl0ylp

# eval remote (Mode B: existing dataset)
python pipeline/eval_remote.py --model-name MobileCLIP2-B-260429_0 \
    --image-compiled-id jgklm3jw5 --text-compiled-id jp83vzlkg

python pipeline/eval_remote.py --model-name MobileCLIP2-B \
    --image-compiled-id j57je47v5 --text-compiled-id jp27rwvr5

# eval remote (Mode C: reuse inference)
python pipeline/eval_remote.py \
    --image-inference-id <image_inference_job_id> --text-inference-id <text_inference_job_id>

python pipeline/eval_remote.py \
    --image-inference-id jp4xy1xv5 --text-inference-id j57je4jl5 --k 7


# analyze hard negatives
python train/analyze_hard_negatives.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B

# analyze affect of crop and resize
python pipeline/sweep_image_rings.py --model-name MobileCLIP2-B --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc16_hn4_hnw1_seed0__20260428_082447/checkpoint_epoch_090.pt



# ── MLP Reconstruction (GELU → ReLU + distillation) ───────────────────────────

# Keep ConvStem GELU (visual[stem]); reconstruct all other visual + text blocks
python mlp_reconstruction/run.py \
    --model-name MobileCLIP2-B --gpu-id 7 \
    --n-calib 4096 --n-iters 40000 --batch-size 32 --log-every 500 --aph-mode uniform \
    --no-relu-stem  --no-relu-text --gelu-threshold 0.99 --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr2e-06_wd0.2_acc16_hn10_hnw1_seed0__20260427_200640/checkpoint_epoch_060.pt

# Auto-revert: blocks with cos_sim < 0.97 after distillation are reverted back to GELU
# python mlp_reconstruction/run.py \
#     --model-name MobileCLIP2-B --gpu-id 3 \
#     --n-calib 4096 --n-iters 40000 --log-every 500 --aph-mode uniform \
#     --no-relu-stem --no-relu-text \
#     --gelu-threshold 0.97

# Manual GELU keep: skip distillation for specific blocks (e.g. poor-quality layers)
# python mlp_reconstruction/run.py \
#     --model-name MobileCLIP2-B --gpu-id 3 \
#     --n-calib 4096 --n-iters 40000 --log-every 500 --aph-mode uniform \
#     --no-relu-stem --no-relu-text \
#     --keep-gelu-blocks visual[b0] visual[b1]

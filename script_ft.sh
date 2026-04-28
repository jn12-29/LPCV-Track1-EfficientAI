# fine-tune
python train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 5 --batch-size 256 --accum-freq 50 --epochs 20 --lr 1e-6

# fine-tune (multi-GPU DDP)

# NCCL_P2P_DISABLE=1: PCIe P2P direct GPU memory access is broken on GPUs 0-3 on this machine.
# Disabling P2P forces NCCL to use shared memory (SHM) instead, which works correctly.
NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29501 train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 0,1,2,3 \
    --batch-size 256 --accum-freq 16 --epochs 200 --lr 2e-6 --init-lr 0 --min-lr 1e-7 --weight-decay 0.2 \
    --loss-type clip --num-hard-negatives 10 \
    --log-every-n-steps 1

NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29501 train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 0,1,2,3 \
    --batch-size 256 --accum-freq 16 --epochs 100 --lr 1e-6 --init-lr 0 --min-lr 5e-7 --weight-decay 0.2 \
    --loss-type clip --num-hard-negatives 4  --compile \
    --log-every-n-steps 1 --crop-rings 2

    # --max-hard-negatives-per-image 0


python train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 2 --batch-size 64 --accum-freq 128 --epochs 100 --lr 1e-6 --no-amp # no amp will make train very slow from 180 samples/s to 40 samples/s per gpu

# fine-tune with SigLIP loss (hard negatives absorbed into sigmoid matrix)
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29501 train/finetune.py \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
    --model-name MobileCLIP2-B --gpu-ids 2,3,5,6 --batch-size 256 --accum-freq 16 --epochs 100 --lr 1e-6 --weight-decay 0.2 \
    --loss-type siglip --num-hard-negatives 4

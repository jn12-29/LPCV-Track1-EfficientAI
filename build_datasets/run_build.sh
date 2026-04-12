# Batch-generate image-text retrieval training data from an image folder via OpenAI-compatible API.
# Example:
# export API_KEY="your_key"
# python vlm_dataset_builder.py ...

# 2026-04-12 new version
python build_datasets/vlm_dataset_builder.py \
    --image_dir build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31PRO_NEW \
    --base_url https://openrouter.ai/api/v1 --api_key $OPENROUTER_API_KEY \
    --model google/gemini-3.1-pro-preview --max_images 4 --max_workers 4 \
    --verify --verify_model google/gemini-3.1-flash-lite-preview

python build_datasets/vlm_dataset_builder.py \
    --image_dir build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31PRO_NEW \
    --base_url https://openrouter.ai/api/v1 --api_key $OPENROUTER_API_KEY \
    --model google/gemini-3.1-pro-preview --max_images 4 --max_workers 4 \
    --reconvert_raw

# visualize raw (before verify)
python build_datasets/visualize_annotations.py \
    --jsonl build_datasets/data/VG_100K_GEMINI31PRO_NEW/dataset_raw.jsonl \
    --image_dir build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31PRO_NEW/vis_raw \
    --n 4

# visualize verified
python build_datasets/visualize_annotations.py \
    --jsonl build_datasets/data/VG_100K_GEMINI31PRO_NEW/dataset_verify.jsonl \
    --image_dir build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31PRO_NEW/vis_verify \
    --n 4

# annotations 1000 samples by flash lite
python build_datasets/vlm_dataset_builder.py \
    --image_dir build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31FLASHLITE_NEW \
    --base_url https://openrouter.ai/api/v1 --api_key $OPENROUTER_API_KEY \
    --model google/gemini-3.1-flash-lite-preview --max_images 10000 --max_workers 32

python build_datasets/visualize_annotations.py \
    --jsonl build_datasets/data/VG_100K_GEMINI31FLASHLITE_NEW/dataset_raw.jsonl \
    --image_dir build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31FLASHLITE_NEW/vis_raw \
    --n 4
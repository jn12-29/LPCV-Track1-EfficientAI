# Batch-generate image-text retrieval training data from an image folder via OpenRouter.
# Example:
# export API_KEY="your_key"
# python vlm_dataset_builder.py \
#     --image_dir /path/to/images \
#     --output_dir ./out \
#     --model openai/gpt-4.1-mini \
#     --detail high \
#     --max_workers 8
# python /mnt/data/vlm_dataset_builder.py \
#   --image_dir /path/to/images \
#   --output_dir /path/to/out \
#   --model openai/gpt-4.1-mini \
#   --detail high \
#   --max_workers 8 \
#   --skip_existing
# This script:
# - walks an image folder recursively
# - sends each image to OpenRouter's OpenAI-compatible Responses API
# - requests structured JSON output using a JSON schema
# - saves raw object-centric annotations to JSONL
# - saves flattened object-centric training samples to JSONL
# - optionally skips images already present in the raw JSONL file

python build_datasets/vlm_dataset_builder.py \
    --image_dir ./build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31PRO \
    --model google/gemini-3.1-pro-preview \
    --detail high \
    --max_images 4 \
    --max_workers 8

python build_datasets/visualize_annotations.py \
    --jsonl build_datasets/data/VG_100K_GEMINI31PRO/dataset_raw.jsonl \
    --image_dir build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31PRO/vis_out \
    --n 4

## GEMini 3.1 Flash Lite
python build_datasets/vlm_dataset_builder.py \
    --image_dir ./build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31FLASHLITE \
    --model google/gemini-3.1-flash-lite-preview \
    --detail high \
    --max_images 4 \
    --max_workers 8

python build_datasets/visualize_annotations.py \
    --jsonl build_datasets/data/VG_100K_GEMINI31FLASHLITE/dataset_raw.jsonl \
    --image_dir build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31FLASHLITE/vis_out \
    --n 4

## GEMini 3.1 Flash Lite Low Resolution
python build_datasets/vlm_dataset_builder.py \
    --image_dir ./build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31FLASHLITE_low \
    --model google/gemini-3.1-flash-lite-preview \
    --detail low \
    --max_images 4 \
    --max_workers 8

python build_datasets/visualize_annotations.py \
    --jsonl build_datasets/data/VG_100K_GEMINI31FLASHLITE_low/dataset_raw.jsonl \
    --image_dir build_datasets/data/VG_100K \
    --output_dir build_datasets/data/VG_100K_GEMINI31FLASHLITE_low/vis_out \
    --n 4

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

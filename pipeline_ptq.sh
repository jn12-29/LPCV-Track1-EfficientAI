set -euo pipefail

export CUDA_VISIBLE_DEVICES=${1:-0}

export mode=${2:-"fp"}
export op_types=${3:-""}
export ptq_scheme=${4:-"w8a8_vit_hub"}
export fixed_text_compile_id=${5:-"jp01nr62g"}
if [ "${mode}" == "fp" ]; then
    compile_ids_file=fp_compile_ids.json
    output_postfix=""
else
    compile_ids_file=ptq_compile_ids.json
    if [ "${ptq_scheme}" == "w8a16_vit_hub" ]; then
        output_postfix="_ptq_w8a16"
    elif [ "${ptq_scheme}" == "w8a8_vit_hub" ]; then
        output_postfix="_ptq_w8a8"
    else
        output_postfix="_ptq_custom"
    fi
    if [ -n "${op_types}" ]; then
        output_postfix="${output_postfix}_${op_types}"
    fi
fi

# export onnx
# python pipeline/export_onnx.py \
#   --model-name MobileCLIP2-B \
#   --checkpoint-path ./checkpoints/MobileCLIP2-B__bs128_ep200_lr1e-06_wd0.2_acc32_hn4_hnw1_seed0__20260420_000848/checkpoint_epoch_085.pt \
#   --output-postfix ""

# python ptq/eval_ptq.py \
#   --onnx-dir exported_MobileCLIP2-B_onnx \
#   --image-postfix "${output_postfix}" \
#   --text-postfix "${output_postfix}" \
#   --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
#   --image-base-dir ./ \
#   --calib-size 1000 \
#   --val-size 100 \
#   --seed 42 \
#   --k 7 \
#   --batch-size 1 \
#   --model-name MobileCLIP2-B

# if [ "${mode}" == "ptq" ]; then
#     python ptq/quantize.py \
#     --onnx-dir exported_MobileCLIP2-B_onnx \
#     --output-suffix "${output_postfix}" \
#     --ptq-scheme "${ptq_scheme}" \
#     --jsonl-path ./build_datasets/data/vg_llm_contrastive_v1/vg_llm_contrastive.jsonl \
#     --image-base-dir ./ \
#     --calib-size 10 \
#     --val-size 100 \
#     --seed 42 \
#     --no-quantize-text \
#     --op-types "${op_types}" \
#     --model-name MobileCLIP2-B
# fi

if [ "${mode}" == "ptq" ]; then
  python vit/export_image.py \
    --onnx-path "exported_MobileCLIP2-B_onnx/image_encoder.onnx" \
    --model-name MobileCLIP2-B \
    --precision w8a8 \
    --target-runtime qnn_dlc \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive_v1/vg_llm_contrastive.jsonl \
    --image-base-dir ./ \
    --calib-size 10 \
    --val-size 100 \
    --seed 42 \
    --text-compile-id "${fixed_text_compile_id}" \
    --ids-file "${compile_ids_file}"
else
  python pipeline/compile_and_profile.py \
    --model-name MobileCLIP2-B \
    --postfix "${output_postfix}" \
    --ids-file "${compile_ids_file}"
fi

python pipeline/eval_remote.py \
  --model-name MobileCLIP2-B \
  --image-compiled-id "$(python - <<'PY'
import json
with open("'"${compile_ids_file}"'", "r", encoding="utf-8") as f:
    print(json.load(f)["image_compile_id"])
PY
)" \
  --text-compiled-id "${fixed_text_compile_id}" \
  --upload-dataset \
  --jsonl-path ./build_datasets/data/vg_llm_contrastive_v1/vg_llm_contrastive.jsonl  \
  --image-base-dir ./ \
  --calib-size 1000 \
  --val-size 100 \
  --seed 42 \
  --k 7

# # python pipeline/eval_remote.py \
# #   --model-name MobileCLIP2-B \
# #   --ids-file  ptq_compile_ids.json \
# #   --upload-dataset \
# #   --jsonl-path ./build_datasets/data/vg_llm_contrastive.jsonl \
# #   --image-base-dir ./ \
# #   --calib-size 1000 \
# #   --val-size 100 \
# #   --seed 42 \
# #   --k 7
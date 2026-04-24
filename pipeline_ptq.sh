
export CUDA_VISIBLE_DEVICES=${1:-0}

export mode=${2:-"fp"}
export op_types=${3:-""}
if [ "${mode}" == "fp" ]; then
    compile_ids_file=fp_compile_ids.json
    output_postfix=""
    image_layout_flag="--image-channel-last"
else
    compile_ids_file=ptq_compile_ids.json
    output_postfix="_ptq_qdq_u8s8_pct_1000"
    output_postfix="${output_postfix}_${op_types}"
    image_layout_flag="--image-channel-last"
fi

# export onnx
python pipeline/export_onnx.py \
  --model-name MobileCLIP2-B \
  --checkpoint-path ./checkpoints/MobileCLIP2-B__bs256_ep200_lr1e-06_wd0.2_acc30_hn4_hnw1_seed0__20260424_012159/checkpoint_epoch_120.pt \
  --output-postfix ""

if [ "${mode}" == "ptq" ]; then
    python ptq/quantize.py \
    --onnx-dir exported_MobileCLIP2-B_onnx \
    --output-suffix "${output_postfix}" \
    --jsonl-path ./build_datasets/data/vg_llm_contrastive_v1/vg_llm_contrastive.jsonl \
    --image-base-dir ./ \
    --calib-size 1000 \
    --val-size 100 \
    --seed 42 \
    --calib-method percentile \
    --quant-format qdq \
    --activation-type quint8 \
    --weight-type qint8 \
    --op-types "${op_types}" \
    --model-name MobileCLIP2-B 
fi

python pipeline/compile_and_profile.py \
  --model-name MobileCLIP2-B \
  --postfix "${output_postfix}" \
  --ids-file "${compile_ids_file}"

python pipeline/eval_remote.py \
  --model-name MobileCLIP2-B \
  --ids-file "${compile_ids_file}" \
  --upload-dataset \
  --jsonl-path ./build_datasets/data/vg_llm_contrastive_v1/vg_llm_contrastive.jsonl  \
  --image-base-dir ./ \
  --calib-size 1000 \
  --val-size 100 \
  --seed 42 \
  --k 7 \
  ${image_layout_flag}
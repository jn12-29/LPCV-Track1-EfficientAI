sudo rm -rf ./LPCV
sudo cp -r /home/gxz/LPCV ./

# export onnx
python mobileclipv2.py --model-name MobileCLIP2-S0
python mobileclipv2.py --model-name MobileCLIP2-S2
python mobileclipv2.py --model-name MobileCLIP2-S3
python mobileclipv2_fp16.py --model-name MobileCLIP2-S0
python mobileclipv2_fp16.py --model-name MobileCLIP2-S2
python mobileclipv2_fp16.py --model-name MobileCLIP2-S3

# eval local onnx
python eval_onnx_local.py --model-name MobileCLIP2-S0 --k 7
python eval_onnx_local.py --model-name MobileCLIP2-S0 --k 7 --postfix _fp16
python eval_onnx_local.py --model-name MobileCLIP2-S2 --k 7
python eval_onnx_local.py --model-name MobileCLIP2-S3 --k 7 --postfix _fp16

# compile and profile
python compile_and_profile.py --model-name MobileCLIP2-S0
python compile_and_profile.py --model-name MobileCLIP2-S0 --postfix _fp16
python compile_and_profile.py --model-name MobileCLIP2-S2
python compile_and_profile.py --model-name MobileCLIP2-S3 --postfix _fp16



# eval remote
python eval_remote.py --k 7 --image-compiled-id jp34ml4mg --text-compiled-id jgzx7q2o5 --model-name MobileCLIP2-S0

python eval_remote.py --k 7 --image-compiled-id j5wd90qmg --text-compiled-id jg9347w8g --model-name MobileCLIP2-S3

python eval_remote_eval_inference.py --k 7 --image-inference-id jp27mzwx5 --text-inference-id jgklyk8y5

CUDA_VISIBLE_DEVICES=1 python finetune.py
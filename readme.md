# LPCV-Track1-EfficientAI

## utils

AddShareEmail.py: share qaihub job to someone identified by email (如今已经集成到compile_and_profile.py等提交compile的脚本中， 理论上不需要单独运行)

dataset.py: 加载与处理sample data数据集

Token.py: qaihub token and dataset id

utils.py: 计算Recall@K

show_model.py: show 模型结构

## Export ONNX Model

mobileclipv2.py: export fp32 onnx model

mobileclipv2_quant.py: export quant(local quant) onnx model

## Submits ONNX Model to QAIHub

compile_and_profile.py: compile and profile, using onnx model exported above

qai_quant_compile_profile.py: from torch model to quant(remote quant), compile and profile, using torch model from website

## Evaluate Model

eval_local.py: 本地推理，评估torch模型

eval_onnx_local.py: 本地推理，评估onnx模型

eval_upload_dataset.py: 上传数据集到qaihub，记得记录dataset id，用于云端推理评估模型

eval_remote.py: 云端推理，评估模型，有inference job id，要配置compile job id 和 dataset id，应该和比赛方评测完全一致。

eval_remote_eval_inference.py: 传入inference job id（获得云端推理输出），只观察recall结果

## other

### 文件夹

LPCV：笑泽的代码

26LPCVC_Track1_Sample_Solution: 官方代码

26LPCVC_Track1_Sample_Solution-wjc: 家诚的代码

data：官方给的样例数据集

exported\_{model_name}\_onnx: 导出的onnx模型

### fp16

mobileclipv2_fp16.py: 其实qaihub会自动转换为fp16，所以不需要手动指定。该文件没有用。

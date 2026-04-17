import onnxruntime as ort


def ort_optimize(input_path: str, output_path: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = output_path
    ort.InferenceSession(input_path, so, providers=["CPUExecutionProvider"])
    print(f"Optimized model saved to: {output_path}")


BASENAME = (
    "exported_MobileCLIP2-S2_bs256_ep20_lr1e-06_wd0.2_acc32_hn4_hnw0.5_seed0_onnx"
)
ort_optimize(
    f"{BASENAME}/image_encoder_sim.onnx",
    f"{BASENAME}/image_encoder_ort.onnx",
)
import onnx
from collections import Counter


def count_ops(path: str):
    model = onnx.load(path)
    cnt = Counter(node.op_type for node in model.graph.node)
    print(f"\n== {path} ==")
    for op in [
        "Conv",
        "BatchNormalization",
        "Erf",
        "Gemm",
        "MatMul",
        "Transpose",
        "Reshape",
        "LayerNormalization",
    ]:
        print(f"{op:20s}: {cnt.get(op, 0)}")


count_ops(f"{BASENAME}/image_encoder.onnx")
count_ops(f"{BASENAME}/image_encoder_sim.onnx")
count_ops(f"{BASENAME}/image_encoder_ort.onnx")

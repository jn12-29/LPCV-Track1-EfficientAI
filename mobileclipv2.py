import torch
import torch.nn as nn
import os
import argparse
import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")

args = parser.parse_args()

model_name = args.model_name

# --- Configuration for File Saving ---
ONNX_DIR = f"exported_{model_name}_onnx"
device = torch.device("cpu")  # use CPU to export onnx model to avoid GPU device issues
# -----------------------------------

# -----------------------------
# 1. Prepare Environment
# -----------------------------
os.makedirs(ONNX_DIR, exist_ok=True)
print(f"Saving ONNX files to directory: {os.path.abspath(ONNX_DIR)}")

# -----------------------------
# 2. Dummy inputs
# -----------------------------
DUMMY_IMAGE_INPUT = torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device)
DUMMY_TEXT_INPUT = torch.randint(0, 49408, (1, 77), dtype=torch.int64, device=device)

# -----------------------------
# 3. Load OpenAIClip wrapper and define encoders
# -----------------------------
from eval_local import _load_clip
from timm.utils import reparameterize_model

clip_model, _, _ = _load_clip(model_name, device)
clip_model.eval()
clip_model = reparameterize_model(clip_model)


class OpenClipVisionEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


class OpenClipTextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 归一化 padding：将 EOS 位置之后的 token 全部清零。
        # token_ids.argmax() 返回第一个最大值（即 EOS=49407）的位置，
        # 这样无论 tokenizer 用 pad=0 还是 pad=49407，模型输入都一致。
        eot_pos = token_ids.argmax(dim=-1, keepdim=True)  # (B, 1)
        positions = torch.arange(
            token_ids.shape[-1], device=token_ids.device
        ).unsqueeze(
            0
        )  # (1, L)
        mask = (positions <= eot_pos).to(token_ids.dtype)  # 1 处保留，0 处清零
        return self.model.encode_text(token_ids * mask)


# -----------------------------
# 4. Create wrapper instances & Get PyTorch Baseline
# -----------------------------
image_encoder = OpenClipVisionEncoder(clip_model)
text_encoder = OpenClipTextEncoder(clip_model)
image_encoder.eval()
text_encoder.eval()

print("\nCalculating PyTorch baseline outputs for validation...")
with torch.no_grad():
    pt_img_feat = image_encoder(DUMMY_IMAGE_INPUT)
    pt_txt_feat = text_encoder(DUMMY_TEXT_INPUT)


def verify_onnx(onnx_path, input_dict, pt_output, rtol=1e-3, atol=1e-4):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {k: v.numpy() for k, v in input_dict.items()}
    ort_out = sess.run(None, ort_inputs)[0]
    pt_out = pt_output.numpy()
    max_diff = np.abs(ort_out - pt_out).max()
    status = (
        "✓ PASS" if np.allclose(ort_out, pt_out, rtol=rtol, atol=atol) else "✗ FAIL"
    )
    print(f"  Max abs diff (ONNX vs PyTorch): {max_diff:.6f}  {status}")
    if status == "✗ FAIL":
        raise RuntimeError(
            f"ONNX output mismatch for {onnx_path}. Max diff={max_diff:.6f}"
        )


# -----------------------------
# 5. Export & Verify Image Encoder
# -----------------------------
image_onnx_path = os.path.join(ONNX_DIR, "image_encoder.onnx")
print(f"\nExporting Image Encoder to {image_onnx_path}...")

torch.onnx.export(
    image_encoder,
    DUMMY_IMAGE_INPUT,
    image_onnx_path,
    input_names=["image"],
    output_names=["embedding"],
    opset_version=18,
    do_constant_folding=True,
    dynamic_axes=None,
    verbose=False,
    export_params=True,
    training=torch.onnx.TrainingMode.EVAL,
    dynamo=True,
)

verify_onnx(image_onnx_path, {"image": DUMMY_IMAGE_INPUT}, pt_img_feat)


# -----------------------------
# 6. Export & Verify Text Encoder
# -----------------------------
text_onnx_path = os.path.join(ONNX_DIR, "text_encoder.onnx")
print(f"\nExporting Text Encoder to {text_onnx_path}...")

torch.onnx.export(
    text_encoder,
    DUMMY_TEXT_INPUT,
    text_onnx_path,
    input_names=["text"],
    output_names=["text_embedding"],
    opset_version=18,
    do_constant_folding=True,
    dynamic_axes=None,
    verbose=False,
    export_params=True,
    training=torch.onnx.TrainingMode.EVAL,
    dynamo=True,
)

verify_onnx(text_onnx_path, {"text": DUMMY_TEXT_INPUT}, pt_txt_feat)

print("\nExport and verification complete.")

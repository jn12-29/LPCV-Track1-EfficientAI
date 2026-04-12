import os
import torch
import numpy as np
import open_clip
from onnxruntime.quantization import (
    shape_inference,
    quantize_static,
    QuantType,
    CalibrationDataReader,
    CalibrationMethod,
)

from dataset import RetrievalEvalDataset

# --- 参数配置 ---
model_name = "MobileCLIP2-S0"
ONNX_DIR = f"exported_{model_name}_onnx"
IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, "image_encoder.onnx")
TEXT_ONNX_PATH = os.path.join(ONNX_DIR, "text_encoder.onnx")

IMAGE_ONNX_W8A8_PATH = os.path.join(ONNX_DIR, "image_encoder_w8a8.onnx")
TEXT_ONNX_W8A8_PATH = os.path.join(ONNX_DIR, "text_encoder_w8a8.onnx")

# 数据集路径配置
DATA_ROOT = "./sample_data"
IMAGE_TO_TEXT_CSV = "./sample_data/img_list.csv"
TEXTNUMS_TO_TEXTS_CSV = "./sample_data/txt_list.csv"
CALIBRATION_SAMPLES = 300


# --- 1. Image 校准数据读取器 ---
class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataset: RetrievalEvalDataset, num_samples: int):
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        self.idx = 0

    def get_next(self) -> dict:
        if self.idx < self.num_samples:
            sample = self.dataset[self.idx]
            image_tensor = sample["image"]
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            image_np = image_tensor.numpy().astype(np.float32)
            self.idx += 1
            return {"image": image_np}
        return None


# --- 2. Text 校准数据读取器 ---
class TextCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataset: RetrievalEvalDataset, tokenizer, num_samples: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_samples = min(num_samples, len(dataset))
        self.idx = 0

    def get_next(self) -> dict:
        if self.idx < self.num_samples:
            sample = self.dataset[self.idx]
            text_str = sample["text"]

            text_tensor = self.tokenizer(text_str)
            text_np = text_tensor.numpy().astype(np.int64)
            self.idx += 1
            return {"text": text_np}
        return None


# --- 3. 执行量化流程 ---
def main():
    print("Loading datasets for calibration...")

    # 1. 实例化图像数据集 (mode="image")
    image_dataset = RetrievalEvalDataset(
        root_dir=DATA_ROOT,
        image_to_text_csv=IMAGE_TO_TEXT_CSV,
        textnums_to_texts_csv=TEXTNUMS_TO_TEXTS_CSV,
        mode="image",
        image_transform=None,
    )

    # 2. 实例化文本数据集 (mode="text")
    text_dataset = RetrievalEvalDataset(
        root_dir=DATA_ROOT,
        image_to_text_csv=IMAGE_TO_TEXT_CSV,
        textnums_to_texts_csv=TEXTNUMS_TO_TEXTS_CSV,
        mode="text",
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    print(
        f"Image dataset loaded: {len(image_dataset)} items. Using {min(CALIBRATION_SAMPLES, len(image_dataset))} for calibration."
    )
    print(
        f"Text dataset loaded: {len(text_dataset)} items. Using {min(CALIBRATION_SAMPLES, len(text_dataset))} for calibration."
    )

    # 分别传入对应的数据集
    image_reader = ImageCalibrationDataReader(image_dataset, CALIBRATION_SAMPLES)
    text_reader = TextCalibrationDataReader(
        text_dataset, tokenizer, CALIBRATION_SAMPLES
    )

    IMAGE_ONNX_PREP_PATH = IMAGE_ONNX_PATH.replace(".onnx", "_prep.onnx")
    TEXT_ONNX_PREP_PATH = TEXT_ONNX_PATH.replace(".onnx", "_prep.onnx")

    # 限定只量化计算密集型算子
    target_ops = ["Conv", "MatMul"]

    # --- 量化 Image Encoder ---
    if os.path.exists(IMAGE_ONNX_PATH):
        print("\nPre-processing Image Encoder...")
        shape_inference.quant_pre_process(IMAGE_ONNX_PATH, IMAGE_ONNX_PREP_PATH)

        print(f"Quantizing Image Encoder to W8A8: {IMAGE_ONNX_W8A8_PATH}...")
        quantize_static(
            model_input=IMAGE_ONNX_PREP_PATH,
            model_output=IMAGE_ONNX_W8A8_PATH,
            calibration_data_reader=image_reader,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=target_ops,
            per_channel=True,
            reduce_range=False,
            calibrate_method=CalibrationMethod.Entropy,
        )
        print("Image Encoder quantization complete.")

    # --- 量化 Text Encoder ---
    if os.path.exists(TEXT_ONNX_PATH):
        print("\nPre-processing Text Encoder...")
        shape_inference.quant_pre_process(TEXT_ONNX_PATH, TEXT_ONNX_PREP_PATH)

        print(f"Quantizing Text Encoder to W8A8: {TEXT_ONNX_W8A8_PATH}...")
        quantize_static(
            model_input=TEXT_ONNX_PREP_PATH,
            model_output=TEXT_ONNX_W8A8_PATH,
            calibration_data_reader=text_reader,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=target_ops,
            per_channel=True,
            reduce_range=False,
            calibrate_method=CalibrationMethod.Entropy,
        )
        print("Text Encoder quantization complete.")

    # 清理临时预处理文件
    if os.path.exists(IMAGE_ONNX_PREP_PATH):
        os.remove(IMAGE_ONNX_PREP_PATH)
    if os.path.exists(TEXT_ONNX_PREP_PATH):
        os.remove(TEXT_ONNX_PREP_PATH)


if __name__ == "__main__":
    main()

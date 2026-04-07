import os
import numpy as np
import qai_hub as hub
import open_clip

from dataset import RetrievalEvalDataset

# --- 参数配置 ---
model_name = "MobileCLIP2-S0"
ONNX_DIR = f"exported_{model_name}_onnx"
IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, "image_encoder.onnx")
TEXT_ONNX_PATH = os.path.join(ONNX_DIR, "text_encoder.onnx")

IMAGE_ONNX_QAI_PATH = os.path.join(ONNX_DIR, "image_encoder_qai_int8.onnx")
TEXT_ONNX_QAI_PATH = os.path.join(ONNX_DIR, "text_encoder_qai_int8.onnx")

DATA_ROOT = "./data"
IMAGE_TO_TEXT_CSV = "./data/img_list.csv"
TEXTNUMS_TO_TEXTS_CSV = "./data/txt_list.csv"
CALIBRATION_SAMPLES = 300


def run_profile(model, device):
    profile_job = hub.submit_profile_job(
        model=model, device=device, options="--max_profiler_iterations 100"
    )
    return profile_job.job_id


def compile_model(model, device, input_specs):
    compile_job = hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options="--target_runtime qnn_dlc --truncate_64bit_io",
    )
    compile_job.modify_sharing(add_emails=["lowpowervision@gmail.com"])
    print(f"作业 {compile_job.job_id} 的权限已成功共享给 lowpowervision@gmail.com！")
    return compile_job.job_id


def main():
    print("Loading datasets for QAI Hub calibration...")

    image_dataset = RetrievalEvalDataset(
        root_dir=DATA_ROOT,
        image_to_text_csv=IMAGE_TO_TEXT_CSV,
        textnums_to_texts_csv=TEXTNUMS_TO_TEXTS_CSV,
        mode="image",
        image_transform=None,
    )

    text_dataset = RetrievalEvalDataset(
        root_dir=DATA_ROOT,
        image_to_text_csv=IMAGE_TO_TEXT_CSV,
        textnums_to_texts_csv=TEXTNUMS_TO_TEXTS_CSV,
        mode="text",
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    image_samples = min(CALIBRATION_SAMPLES, len(image_dataset))
    text_samples = min(CALIBRATION_SAMPLES, len(text_dataset))

    print(f"Generating image calibration data in memory ({image_samples} samples)...")
    image_calib_list = []
    for i in range(image_samples):
        sample = image_dataset[i]
        image_tensor = sample["image"]
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_calib_list.append(image_tensor.numpy().astype(np.float32))

    # 直接构造内存字典，废弃 np.savez 本地保存逻辑
    image_calib_dict = {"image": image_calib_list}

    print(f"Generating text calibration data in memory ({text_samples} samples)...")
    text_calib_list = []
    for i in range(text_samples):
        sample = text_dataset[i]
        text_tensor = tokenizer(sample["text"])
        text_calib_list.append(text_tensor.numpy().astype(np.int64))

    # 直接构造内存字典
    text_calib_dict = {"text": text_calib_list}

    target_device = hub.Device("XR2 Gen 2 (Proxy)")

    # if os.path.exists(IMAGE_ONNX_PATH):
    #     print("\n--- Image Encoder Pipeline ---")
    #     print("1. Submitting quantization job...")
    #     image_quantize_job = hub.submit_quantize_job(
    #         model=IMAGE_ONNX_PATH,
    #         calibration_data=image_calib_dict,  # 传入内存字典
    #         weights_dtype=hub.QuantizeDtype.INT8,
    #         activations_dtype=hub.QuantizeDtype.INT8,
    #         name=f"{model_name}-Image-Quant",
    #     )
    #     image_quant_model = image_quantize_job.get_target_model()
    #     image_quant_model.download(IMAGE_ONNX_QAI_PATH)

    #     print("2. Submitting compile job...")
    #     img_compile_id = compile_model(
    #         model=image_quant_model,
    #         device=target_device,
    #         input_specs={"image": (1, 3, 224, 224)},
    #     )

    #     print("3. Submitting profile job...")
    #     run_profile(
    #         model=hub.get_job(img_compile_id).get_target_model(), device=target_device
    #     )

    if os.path.exists(TEXT_ONNX_PATH):
        print("\n--- Text Encoder Pipeline ---")
        print("1. Submitting quantization job...")
        text_quantize_job = hub.submit_quantize_job(
            model=TEXT_ONNX_PATH,
            calibration_data=text_calib_dict,  # 传入内存字典
            weights_dtype=hub.QuantizeDtype.INT8,
            name=f"{model_name}-Text-Quant",
        )
        text_quant_model = text_quantize_job.get_target_model()
        text_quant_model.download(TEXT_ONNX_QAI_PATH)

        print("2. Submitting compile job...")
        txt_compile_id = compile_model(
            model=text_quant_model,
            device=target_device,
            input_specs={"text": ((1, 77), "int64")},
        )

        print("3. Submitting profile job...")
        run_profile(
            model=hub.get_job(txt_compile_id).get_target_model(), device=target_device
        )

    print("\nAll pipeline jobs submitted successfully.")


if __name__ == "__main__":
    main()

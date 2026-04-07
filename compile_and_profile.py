import qai_hub
import onnx
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
parser.add_argument("--postfix", type=str, default="")


args = parser.parse_args()
model_name = args.model_name
postfix = args.postfix
# --- Configuration ---
ONNX_DIR = f"exported_{model_name}_onnx"
# ---------------------


def run_profile(model, name, device):
    """Submit a profile job for the model."""
    profile_job = qai_hub.submit_profile_job(
        model=model, name=name, device=device, options="--max_profiler_iterations 100"
    )
    return profile_job.job_id


def compile_model(model, name, device, input_specs):
    """Submits a compile job for the model and returns the job instance."""
    compile_job = qai_hub.submit_compile_job(
        model=model,
        name=name,
        device=device,
        input_specs=input_specs,
        options="--target_runtime qnn_dlc --truncate_64bit_io",
    )
    compile_job.modify_sharing(add_emails=["lowpowervision@gmail.com"])
    print(f"作业 {compile_job.job_id} 的权限已成功共享给 lowpowervision@gmail.com！")
    return compile_job.job_id


# Construct the full paths
IMAGE_ONNX_PATH = os.path.join(ONNX_DIR, f"image_encoder{postfix}.onnx")
TEXT_ONNX_PATH = os.path.join(ONNX_DIR, f"text_encoder{postfix}.onnx")

if not os.path.exists(ONNX_DIR):
    print(
        f"Error: Directory '{ONNX_DIR}' not found. Please run 'export_onnx.py' first."
    )
    sys.exit(1)


# Load the ONNX models from the new location

print(f"Loading ONNX Image Encoder from {IMAGE_ONNX_PATH}...")
onnx_img_model = onnx.load(IMAGE_ONNX_PATH)

# Check the model for errors
try:
    onnx.checker.check_model(onnx_img_model)
    print("Image ONNX model is valid ✅")
except onnx.checker.ValidationError as e:
    print("Image ONNX model validation failed ❌")
    print(e)

print(f"\nLoading ONNX Text Encoder from {TEXT_ONNX_PATH}...")
onnx_txt_model = onnx.load(TEXT_ONNX_PATH)

# Check the model for errors
try:
    onnx.checker.check_model(onnx_txt_model)
    print("Text ONNX model is valid ✅")
except onnx.checker.ValidationError as e:
    print("Text ONNX model validation failed ❌")
    print(e)

target_device = qai_hub.Device("XR2 Gen 2 (Proxy)")

# Submit compilation jobs
print("\nSubmitting compilation jobs to QAI Hub...")
img_id = compile_model(
    model=onnx_img_model,
    name=model_name + f"_image_encoder{postfix}",
    device=target_device,
    input_specs={"image": (1, 3, 224, 224)},
)
txt_id = compile_model(
    model=onnx_txt_model,
    name=model_name + f"_text_encoder{postfix}",
    device=target_device,
    input_specs={"text": ((1, 77), "int64")},
)

print(f"Image compilation job ID: {img_id}")
print(f"Text compilation job ID: {txt_id}")


# Submit profiling jobs
print("\nSubmitting profiling jobs to QAI Hub...")
run_profile(
    model=qai_hub.get_job(img_id).get_target_model(),
    name=model_name + f"_image_encoder{postfix}",
    device=target_device,
)
run_profile(
    model=qai_hub.get_job(txt_id).get_target_model(),
    name=model_name + f"_text_encoder{postfix}",
    device=target_device,
)
print("Profiling jobs submitted for both models.")

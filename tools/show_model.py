import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from utils.clip_utils import _load_clip

MODEL_NAMES = [
    "MobileCLIP2-S0",
    "MobileCLIP2-S2",
    "MobileCLIP2-B",
]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cpu")

for model_name in MODEL_NAMES:
    try:
        clip_model, _, _ = _load_clip(model_name, device)
        clip_model.eval()
        structure = str(clip_model)
        out_path = os.path.join(OUTPUT_DIR, f"{model_name}_structure.txt")
        with open(out_path, "w") as f:
            f.write(structure)
        print(f"[{model_name}] saved -> {out_path}")
    except Exception as e:
        print(f"[{model_name}] SKIP: {e}")

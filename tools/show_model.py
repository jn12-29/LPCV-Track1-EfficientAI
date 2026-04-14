from utils.clip_utils import _load_clip
import torch
import torch.nn as nn
import os


# argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")

args = parser.parse_args()

model_name = args.model_name

device = torch.device("cpu")
clip_model, _, _ = _load_clip(model_name, device)
clip_model.eval()
# print model structure in file
with open(f"{model_name}_structure.txt", "w") as f:
    f.write(str(clip_model))

print(clip_model)

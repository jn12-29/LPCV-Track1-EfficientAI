from __future__ import annotations

import numpy as np
from PIL import Image
import torch


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize to 224×224, divide by 255, return CHW float32 tensor.

    Shape: (3, 224, 224). No mean/std normalization — competition requirement.
    Callers add the batch dimension as needed.
    """
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))

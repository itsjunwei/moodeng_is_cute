import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.quantization
from torchvision.ops.misc import Conv2dNormActivation
import torchaudio.transforms
import numpy as np

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(r'.\predictions\vf7x4vpt\model_state_dict.pt', map_location=device)

# Check precision
for name, param in model.items():
    print(f"Model parameter {name} has dtype: {param.dtype}")
    break

# Optional: Force FP16 or FP32
# model.half()  # Convert model to FP16
# model.float()  # Convert model to FP32
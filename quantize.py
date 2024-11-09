import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.quantization
from torchvision import datasets, transforms
import os

from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

from models.baseline import get_model

import warnings

warnings.filterwarnings('ignore')

# model_path = 'model_state_dict.pt'
# quantized_model_save_path = 'quant_model_state_dict.pt'
# batch_size = 32
# num_classes = 10

# # Define your model (replace with your DCASE Task 1 model)
# class DCASEModel(nn.Module):
#     def __init__(self):
#         super(DCASEModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 3)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu1 = nn.ReLU()
#         # Add more layers as per your model architecture

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         return x

# Load your pre-trained model
model_fp32 = get_model()
model_fp32.load_state_dict(torch.load('model_state_dict.pt'))
model_fp32.eval()

# Set the quantization configuration
model_fp32.qconfig = quantization.get_default_qconfig('fbgemm')  # or 'qnnpack' for mobile

# Fuse the layers
model_fp32_fused = quantization.fuse_modules(model_fp32, [['conv1', 'bn1', 'relu1']])

# Prepare the model for static quantization
model_fp32_prepared = quantization.prepare(model_fp32_fused)

# Calibration step (use a representative dataset)
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            model(inputs)

# Create a DataLoader for calibration (replace with your dataset)
calibration_data = DataLoader(...)  # Your dataset here
calibrate(model_fp32_prepared, calibration_data)

# Convert the model to a quantized version
model_int8 = quantization.convert(model_fp32_prepared)

# Save the quantized model
torch.save(model_int8.state_dict(), 'quant_model_state_dict.pt')

# To run inference with the quantized model
model_int8.eval()
input_tensor = torch.randn(1, 1, 64, 64)  # Example input shape
output = model_int8(input_tensor)
print(output)
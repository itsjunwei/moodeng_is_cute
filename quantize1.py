import torch
from torch import nn
import torchaudio
import torchvision.transforms as transforms

# Load DCASE 2024 Task 1 baseline model
class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        # Example: Define layers (replace with actual model layers from repo)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10)  # Change layer size based on model

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = self.fc1(x)
        return x

# Initialize and load the model
model = BaselineModel()
model.load_state_dict(torch.load("path/to/your/model.pth"))
model.eval()

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Verify quantized model size reduction
torch.save(quantized_model.state_dict(), "quantized_model.pth")
print("Quantized model saved as 'quantized_model.pth'")

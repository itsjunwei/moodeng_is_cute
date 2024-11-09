# Sample PyTorch script for CNN quantization using static, dynamic quantization, and QAT
# Author: OpenAI's ChatGPT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torchvision import datasets, transforms
import os

# -----------------------------
# Section 1: Define the Model
# -----------------------------

# Define a simple CNN model suitable for the MNIST dataset
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        # Output layer
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        output = self.logsoftmax(x)
        return output

# -----------------------------
# Section 2: Prepare Data Loaders
# -----------------------------

# Transformations for the MNIST dataset
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=test_transform, download=True)

# Data loaders for training and testing
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# -----------------------------
# Section 3: Define Training and Testing Functions
# -----------------------------

def train(model, device, train_loader, optimizer, epoch):
    """Train the model for one epoch."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Zero the gradients
        output = model(data)                # Forward pass
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Update weights
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    """Evaluate the model on the test dataset."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)            # Forward pass
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)                        # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()            # Count correct predictions
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.2f}%)\n')

# -----------------------------
# Section 4: Train the Model
# -----------------------------

# Set device to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the model and move it to the device
model = SimpleCNN().to(device)

# Define optimizer (we use Adam optimizer here)
optimizer = torch.optim.Adam(model.parameters())

# Train the model for 1 epoch (increase the number of epochs for better accuracy)
for epoch in range(1, 2):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# Save the trained model's state_dict
torch.save(model.state_dict(), 'model.pth')

# -----------------------------
# Section 5: Dynamic Quantization
# -----------------------------

"""
Dynamic Quantization:
- Quantizes the weights of certain layers (e.g., nn.Linear) to int8.
- Activations are dynamically quantized at runtime.
- Best suited for models dominated by linear operations (e.g., NLP models).
- Reduces model size and can improve inference speed on CPUs.
"""

# Apply dynamic quantization to the model (specifically the Linear layers)
quantized_model_dynamic = torch.quantization.quantize_dynamic(
    model,  # Original model
    {nn.Linear},  # Layers to quantize
    dtype=torch.qint8  # Quantized data type
)

# Evaluate the dynamically quantized model
test(quantized_model_dynamic, device, test_loader)

# Save the dynamically quantized model's state_dict
torch.save(quantized_model_dynamic.state_dict(), 'quantized_model_dynamic.pth')

# Compare model sizes
f1_size = os.path.getsize('model.pth')
f2_size = os.path.getsize('quantized_model_dynamic.pth')
print(f'Size of original model: {f1_size / 1e6:.2f} MB')
print(f'Size of dynamically quantized model: {f2_size / 1e6:.2f} MB')

# -----------------------------
# Section 6: Static Quantization
# -----------------------------

"""
Static Quantization:
- Quantizes both weights and activations to int8.
- Requires calibration with a representative dataset to determine the scale and zero-point.
- Provides better performance improvements compared to dynamic quantization, especially for CNNs.
- Involves modifying the model to include QuantStub and DeQuantStub modules and fusing layers.
"""

# Define a model class with QuantStub and DeQuantStub for static quantization
class QuantizedSimpleCNN(nn.Module):
    def __init__(self):
        super(QuantizedSimpleCNN, self).__init__()
        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        # Output layer remains the same

    def forward(self, x):
        x = self.quant(x)  # Quantize the input
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.dequant(x)  # Dequantize the output
        output = F.log_softmax(x, dim=1)
        return output

    def fuse_model(self):
        # Fuse Conv+ReLU and Linear+ReLU layers for quantization
        torch.quantization.fuse_modules(self, [['conv1', 'relu1'], ['conv2', 'relu2'], ['fc1', 'relu3']], inplace=True)

# Create an instance of the quantized model and load the trained weights
model_static = QuantizedSimpleCNN().to(device)
model_static.load_state_dict(model.state_dict())

# Fuse the model modules
model_static.eval()
model_static.fuse_model()

# Set the quantization configuration for static quantization
model_static.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepare the model for static quantization
torch.quantization.prepare(model_static, inplace=True)

# Function to calibrate the model
def calibrate(model, data_loader):
    """Calibrate the model with a representative dataset."""
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            model(data)

# Calibrate the model using the training data
calibrate(model_static, train_loader)

# Convert the calibrated model to a quantized version
model_int8_static = torch.quantization.convert(model_static, inplace=False)

# Evaluate the statically quantized model
test(model_int8_static, device, test_loader)

# Save the statically quantized model's state_dict
torch.save(model_int8_static.state_dict(), 'quantized_model_static.pth')

# Compare the size of the statically quantized model
f3_size = os.path.getsize('quantized_model_static.pth')
print(f'Size of statically quantized model: {f3_size / 1e6:.2f} MB')

# -----------------------------
# Section 7: Quantization Aware Training (QAT)
# -----------------------------

"""
Quantization Aware Training (QAT):
- Simulates quantization effects during training to improve accuracy.
- Model is trained with fake quantization modules that mimic the effects of quantization.
- Often achieves higher accuracy compared to post-training quantization methods.
- Requires training the model with quantization aware layers and operations.
"""

# Create a new instance of the model for QAT and load trained weights
model_qat = QuantizedSimpleCNN().to(device)
model_qat.load_state_dict(model.state_dict())

# Fuse modules
model_qat.train()
model_qat.fuse_model()

# Set the quantization configuration for QAT
model_qat.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# Prepare the model for QAT
torch.quantization.prepare_qat(model_qat, inplace=True)

# Define an optimizer for QAT
optimizer_qat = torch.optim.Adam(model_qat.parameters(), lr=0.001)

# Fine-tune the model with QAT (reduce number of epochs for quick example)
for epoch in range(1, 2):
    train(model_qat, device, train_loader, optimizer_qat, epoch)
    test(model_qat, device, test_loader)

# Convert the trained QAT model to a quantized version
model_qat.eval()
model_int8_qat = torch.quantization.convert(model_qat.eval(), inplace=False)

# Evaluate the quantized model obtained from QAT
test(model_int8_qat, device, test_loader)

# Save the quantized model from QAT
torch.save(model_int8_qat.state_dict(), 'quantized_model_qat.pth')

# Compare the size of the QAT quantized model
f4_size = os.path.getsize('quantized_model_qat.pth')
print(f'Size of QAT quantized model: {f4_size / 1e6:.2f} MB')

# -----------------------------
# Section 8: Summary of Results
# -----------------------------

print("Model size comparison (in MB):")
print(f"Original model: {f1_size / 1e6:.2f} MB")
print(f"Dynamically quantized model: {f2_size / 1e6:.2f} MB")
print(f"Statically quantized model: {f3_size / 1e6:.2f} MB")
print(f"QAT quantized model: {f4_size / 1e6:.2f} MB")

import torch
import torch.quantization
from models.baseline import get_model  # Import the baseline model as defined by DCASE

# Load the pretrained baseline model (assuming it was trained and saved)
model = get_model()
model.load_state_dict(torch.load("model_state_dict.pt"))
model.eval()  # Set the model to evaluation mode for quantization

# Define quantization configuration for dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,                     # Model to be quantized
    {torch.nn.Linear},         # Layers to apply quantization to, typically Linear layers
    dtype=torch.qint8          # Quantization data type
)

# Save the quantized model
torch.save(quantized_model.state_dict(), "quantized_baseline_model.pth")

# Test the quantized model on a sample input
sample_input = torch.rand(1, 1)  # Replace input_shape with the actual input shape
with torch.no_grad():
    quantized_output = quantized_model(sample_input)
    print("Quantized model output:", quantized_output)

import torch
import torch.quantization
from run_training import PLModule
from dataset.dcase24 import get_training_set, get_test_set
from torch.utils.data import DataLoader

# Load the pre-trained model
model = PLModule.load_from_checkpoint("C:/Users/fenel/Documents/dcase2024_task1_baseline/DCASE24_Task1/5a0a7ud6/checkpoints/epoch=149-step=4200.ckpt")

# Set the model to evaluation mode
model.eval()

# Fuse the model layers
model.model.fuse_model()

# Prepare the model for quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibrate the model with a few batches of data
train_loader = DataLoader(get_training_set(), batch_size=32, shuffle=True)
for batch in train_loader:
    model(batch[0])

# Convert the model to a quantized version
torch.quantization.convert(model, inplace=True)

# Save the quantized model
torch.save(model.state_dict(), 'quantized_model.pth')

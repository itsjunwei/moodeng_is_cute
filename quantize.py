import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.quantization
from torchvision.ops.misc import Conv2dNormActivation
import torchaudio.transforms
import numpy as np
import librosa
import os

from models.baseline import get_model, initialize_weights, Block
from models.helpers.utils import make_divisible
from dataset.dcase24 import get_training_set, get_test_set

import warnings

warnings.filterwarnings('ignore')

# model_path = 'model_state_dict.pt'
# quantized_model_save_path = 'quant_model_state_dict.pt'
# batch_size = 256
# num_classes = 10

# # module for resampling waveforms on the fly
#         resample = torchaudio.transforms.Resample(
#             orig_freq=44100, 
#             new_freq=32000
#         )

# define mel spectrogram
mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, 
                                           n_fft=4096, 
                                           win_length=3072,
                                           hop_length=500, 
                                           n_mels=256)


def get_model_1(n_classes=10, in_channels=1, base_channels=32, channels_multiplier=1.8, expansion_rate=2.1,
              n_blocks=(3, 2, 1), strides=None):
    """
    @param n_classes: number of the classes to predict
    @param in_channels: input channels to the network, for audio it is by default 1
    @param base_channels: number of channels after in_conv
    @param channels_multiplier: controls the increase in the width of the network after each stage
    @param expansion_rate: determines the expansion rate in inverted bottleneck blocks
    @param n_blocks: number of blocks that should exist in each stage
    @param strides: default value set below
    @return: full neural network model based on the specified configs
    """

    if strides is None:
        strides = dict(
            b2=(1, 1),
            b3=(1, 2),
            b4=(2, 1)
        )

    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "expansion_rate": expansion_rate,
        "n_blocks": n_blocks,
        "strides": strides
    }

    m = Network_1(model_config)
    return m

class MelSpec(nn.Module):
    def __init__(self):
        super(MelSpec, self).__init__()
        resample = torchaudio.transforms.Resample(
            orig_freq=44100, 
            new_freq=32000
        )

        # define mel spectrogram
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, 
                                                    n_fft=4096, 
                                                    win_length=3072,
                                                    hop_length=500, 
                                                    n_mels=256)
        
        self.mel = torch.nn.Sequential(
            resample,
            mel
        )

    def forward(self,x):
        return self.mel(x)


class Network_1(nn.Module):
    def __init__(self, config):
        super(Network_1, self).__init__()
        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        # Convolutional layers
        n_classes = config['n_classes']
        in_channels = config['in_channels']
        base_channels = config['base_channels']
        channels_multiplier = config['channels_multiplier']
        expansion_rate = config['expansion_rate']
        n_blocks = config['n_blocks']
        strides = config['strides']
        n_stages = len(n_blocks)

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [make_divisible(base_channels * channels_multiplier ** stage_id, 8)
                                                for stage_id in range(n_stages)]
        self.total_block_count = 0

        self.in_c = nn.Sequential(
            Conv2dNormActivation(in_channels,
                                 channels_per_stage[0] // 4,
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 inplace=False
                                 ),
            Conv2dNormActivation(channels_per_stage[0] // 4,
                                 channels_per_stage[0],
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 inplace=False
                                 ),
        )

        self.stages = nn.Sequential()
        for stage_id in range(n_stages):
            stage = self._make_stage(channels_per_stage[stage_id],
                                     channels_per_stage[stage_id + 1],
                                     n_blocks[stage_id],
                                     strides=strides,
                                     expansion_rate=expansion_rate
                                     )
            self.stages.add_module(f"s{stage_id + 1}", stage)

        ff_list = []
        ff_list += [nn.Conv2d(
            channels_per_stage[-1],
            n_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False),
            nn.BatchNorm2d(n_classes),
        ]

        ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.feed_forward = nn.Sequential(
            *ff_list
        )

        self.apply(initialize_weights)


    def _make_stage(self,
                    in_channels,
                    out_channels,
                    n_blocks,
                    strides,
                    expansion_rate):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f'b{block_id}'
            self.total_block_count = self.total_block_count + 1
            if bname in strides:
                stride = strides[bname]
            else:
                stride = (1, 1)

            block = self._make_block(
                in_channels,
                out_channels,
                stride=stride,
                expansion_rate=expansion_rate
            )
            stage.add_module(bname, block)

            in_channels = out_channels
        return stage

    def _make_block(self,
                    in_channels,
                    out_channels,
                    stride,
                    expansion_rate
                    ):

        block = Block(in_channels,
                      out_channels,
                      expansion_rate,
                      stride
                      )
        return block

    def _forward_conv(self, x):
        x = self.in_c(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        x = self.quant(x)  # Quantize the input
        x = self._forward_conv(x)
        x = self.feed_forward(x)
        x = self.dequant(x)
        logits = x.squeeze(2).squeeze(2)
        return logits


# Load your pre-trained model
model_fp32 = get_model_1()
model_fp32.load_state_dict(torch.load(r'.\predictions\0vl52i7d\model_state_dict.pt'), 'weights_only=True')
model_fp32.eval()
model_unquantized = get_model_1()
model_unquantized.load_state_dict(torch.load(r'.\predictions\0vl52i7d\model_state_dict.pt'), 'weights_only=True')
print(model_unquantized)
# print(model_fp32)

# Set the quantization configuration
model_fp32.qconfig = quantization.get_default_qconfig('fbgemm')  # or 'qnnpack' for mobile

# # Fuse the layers
# model_fp32_fused = quantization.fuse_modules(model_fp32, [['conv1', 'bn1', 'relu1']])

# Prepare the model for static quantization
model_fp32_prepared = quantization.prepare(model_fp32, inplace=True)

MelSpecGenerator = MelSpec()

# Calibration step (use a representative dataset)
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            raw_waveform = inputs[0]
            mel_spec = MelSpecGenerator(raw_waveform)   
            model(mel_spec)
            print("{}/{}".format(idx, len(data_loader)), end='\r')
            if idx == 0: break

train_dataset = get_training_set()  # This accesses the original training dataset

# Set the number of samples for calibration (5% of training data)
num_calibration_samples = int(0.05 * len(train_dataset))
calibration_indices = np.random.choice(len(train_dataset), num_calibration_samples, replace=False)

# Create a subset of the training data for calibration
calibration_data = Subset(train_dataset, calibration_indices)            

# Create a DataLoader for calibration (replace with your dataset)
calibration_data = DataLoader(dataset=calibration_data,
                                batch_size=256,  # Use the same batch size as train_dl
                                num_workers=0,
                                shuffle=False  # No need to shuffle for calibration
                                )
calibrate(model_fp32_prepared, calibration_data)

# Convert the model to a quantized version
model_int8 = quantization.convert(model_fp32_prepared)
print(model_int8)

# Create the Mel spectrogram generator
mel_spec_transform = MelSpec()

# Evaluate the quantized model
def evaluate(model, dataloader, mel_spec_transform):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch[2]
            raw_waveform = batch[0] 
            mel_spec = mel_spec_transform(raw_waveform)  # Convert to Mel spectrogram
            model(mel_spec)

            # Forward pass through the model
            outputs = model(mel_spec)
            _, predicted = torch.max(outputs.data, dim =1)
            n_correct_per_sample = (predicted == labels)
            n_correct = n_correct_per_sample.sum()
            correct += n_correct
            total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy


# Evaluate quantized model on test data
test_loader = DataLoader(get_test_set(), batch_size=32, shuffle=True)
accuracy = evaluate(model_int8, test_loader, mel_spec_transform)
print(f"Quantized model accuracy: {accuracy:.2f}%")
accuracy = evaluate(model_unquantized, test_loader, mel_spec_transform)
print(f"Unquantized model accuracy: {accuracy:.2f}%")


def test_single_batch(model, dataloader, mel_spec_transform=None):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        # Get a single batch of data
        for batch in dataloader:
            labels = batch[2]
            labels = labels.type(torch.LongTensor)
            inputs = batch[0]   
            
            # Print shapes to verify correct input/output sizes
            print("Input shape:", inputs.shape)
            print("Label shape:", labels.shape)
            
            # Convert to Mel spectrogram if needed
            if mel_spec_transform:
                inputs = mel_spec_transform(inputs)
                print("Transformed input shape (Mel spectrogram):", inputs.shape)
            
            # Forward pass through the model
            outputs = model(inputs)
            print("Output shape:", outputs.shape)
            print("Outputs:", outputs)
            print("Labels:", labels)
            
            # Calculate predictions
            _, predicted = torch.max(outputs.data, 1)
            print("Predicted labels:", predicted)
            print("Actual labels:", labels)
            
            # Calculate accuracy for this single batch
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / labels.size(0)
            print(f"Accuracy for this batch: {accuracy:.2f}%")
            
            break  # Only process one batch for testing

# # 1. Check model parameters
# for name, param in model_unquantized():
#     if param.requires_grad:
#         print(f"Layer: {name} | Weights mean: {param.data.mean()} | Stddev: {param.data.std()}")

# model_unquantized.eval()# 2. Test model output on a single batch from test data
# with torch.no_grad():
    
#     for batch in test_loader:
#         inputs, labels = batch[:2]
#         if mel_spec_transform:
#             inputs = mel_spec_transform(inputs)
#         outputs = model_unquantized(inputs)
#         print("Raw Outputs (logits):", outputs)
#         break

# # 3. Run single batch evaluation on training data to check overfitting/generalization
# accuracy_on_train = test_single_batch(model_unquantized, train_dataset(), mel_spec_transform=mel_spec_transform)
# print(f"Accuracy on training data (single batch): {accuracy_on_train:.2f}%")

# Run this function with your unquantized model and dataloader
# Replace `unquantized_model` and `test_loader` with your actual model and dataloader
test_single_batch(model_unquantized, test_loader, mel_spec_transform=mel_spec_transform)

# Save the quantized model
torch.save(model_int8.state_dict(), 'quant_model_state_dict.pt')

# To run inference with the quantized model
model_int8.eval()
input_tensor = torch.randn((1, 1, 64, 64))  # Example input shape
output = model_int8(input_tensor)
print(output)

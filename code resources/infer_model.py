import os
# Ensure that script working directory is same directory as the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
print("Changing directory to : ", dname)
os.chdir(dname)


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

from models.baseline import get_model, initialize_weights, Block
from models.helpers.utils import make_divisible
from dataset.dcase24 import get_training_set, get_test_set

import warnings

# warnings.filterwarnings('ignore')


def get_model_1(n_classes=10, in_channels=1, base_channels=32, channels_multiplier=1.8, expansion_rate=2.1,
              n_blocks=(3, 2, 1), strides=None, quantize=False, mel_forward=False):
    """
    @param n_classes: number of the classes to predict
    @param in_channels: input channels to the network, for audio it is by default 1
    @param base_channels: number of channels after in_conv
    @param channels_multiplier: controls the increase in the width of the network after each stage
    @param expansion_rate: determines the expansion rate in inverted bottleneck blocks
    @param n_blocks: number of blocks that should exist in each stage
    @param strides: default value set below
    @param quantize: Determines the inclusion of Quant and Dequant stubs
    @param mel_forward: Determines the inclusion of the computation of Mel Specs within the Model
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

    m = Network_1(model_config, quantize=quantize, mel_forward=mel_forward)
    return m


class MelSpec(nn.Module):
    def __init__(self):
        super(MelSpec, self).__init__()
        
        # Resample the audio clip from original frequency to a new determined sampling frequency
        resample = torchaudio.transforms.Resample(
            orig_freq=44100, 
            new_freq=32000
        )

        # Generate the Mel Spectrogram with specified parameters
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, 
                                                    n_fft=4096, 
                                                    win_length=3072,
                                                    hop_length=500, 
                                                    n_mels=256)
        
        # Sequentially perform resampling then the Mel Spec generation
        self.mel = torch.nn.Sequential(
            resample,
            mel
        )

    def forward(self,x):
        x = self.mel(x)
        x = (x + 1e-5).log()
        return x


class Network_1(nn.Module):
    def __init__(self, config, quantize=False, mel_forward=False):
        super(Network_1, self).__init__()
        self.quantize = quantize
        if self.quantize:
            # Quantization stubs
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            
        self.mel_forward = mel_forward
        if self.mel_forward:
            # Resample the audio clip from original frequency to a new determined sampling frequency
            resample = torchaudio.transforms.Resample(
                orig_freq=44100, 
                new_freq=32000
            )

            # Generate the Mel Spectrogram with specified parameters
            mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, 
                                                        n_fft=4096, 
                                                        win_length=3072,
                                                        hop_length=500, 
                                                        n_mels=256)
            
            # Sequentially perform resampling then the Mel Spec generation
            self.mel = torch.nn.Sequential(
                resample,
                mel
            )
        
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
        if self.mel_forward:
            x = self.mel(x)
            x = (x + 1e-5).log()
        if self.quantize:
            x = self.quant(x)  # Quantize the input
        x = self._forward_conv(x)
        x = self.feed_forward(x)
        if self.quantize:
            x = self.dequant(x)
        logits = x.squeeze(2).squeeze(2)
        return logits


# Evaluate the quantized model
def evaluate(model, dataloader, device=torch.device("cpu"), mel_spec_transform=None):
    """
    Evaluates the model on the given dataloader.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader for evaluation data.
        device (torch.device): The device to perform evaluation on.
        mel_spec_transform (callable, optional): Transformation function to apply to raw_waveform.
                                                If None, no transformation is applied.

    Returns:
        float: The accuracy percentage of the model on the evaluation data.
    """

    model.eval()
    model.to(device)  # Move model to the specified device

    correct = 0
    total = 0

    with torch.no_grad():
        for idx , batch in enumerate(dataloader):

            # Unpack the batch
            # Assuming val_batch structure: x, files, labels, devices, cities
            x, files, labels, devices, cities = batch
            
            # Move inputs and labels to the correct device
            x = x.to(device)
            labels = labels.to(device).long()  # Ensure labels are LongTensors
            
            # Apply transformation if provided
            if mel_spec_transform is not None:
                x = mel_spec_transform(x)

            # Forward pass
            outputs = model(x)
            
            # Predictions
            _, preds = torch.max(outputs, dim=1)
            
            # Calculate correct predictions
            n_correct = (preds == labels).sum().item()
            correct += (preds == labels).sum().item()
            
            # Total number of labels
            total += labels.size(0)

            print("Batch: {}/{} -- Accuracy: {}/{}".format(idx, len(dataloader), n_correct, labels.size(0)))
            
    accuracy = 100 * correct / total
    return accuracy


# Load your pre-trained model
model_unquantized = get_model_1(quantize=False, mel_forward=True)
chkpt = torch.load(r'.\predictions\0vl52i7d\model_state_dict.pt')
model_unquantized.load_state_dict(chkpt, strict=False)
# checkpoint_path = r"DCASE24_Task1\0vl52i7d\checkpoints\last.ckpt"

# # Step 1: Load the checkpoint
# checkpoint = torch.load(checkpoint_path, map_location='cpu')
 
# # Step 2: Extract the state_dict
# state_dict = checkpoint['state_dict'] # PyTorch Lightning typically stores the model's state_dict under the 'state_dict' key
 
# # Step 3: Adjust the keys if necessary. Often, Lightning prefixes model parameters with 'model.', so we need to remove this
# new_state_dict = {}
# prefix = 'model.'  # Change this if your prefix is different or absent
 
# for key, value in state_dict.items():
#     # if "mel" not in key:
#     if key.startswith(prefix):
#         new_key = key[len(prefix):]  # Remove the prefix
#     else:
#         new_key = key
#     new_state_dict[new_key] = value

# # Load the adjusted state_dict into the model
# missing_keys, unexpected_keys = model_unquantized.load_state_dict(new_state_dict, strict=False)

# # Check what keys were not used or are missing
# if missing_keys:
#     print(f"Warning: Missing keys in state_dict: {missing_keys}")
# if unexpected_keys:
#     print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")

# Create the Mel spectrogram generator
mel_spec_transform = MelSpec()
mel_spec_transform = None

# Evaluate quantized model on test data
test_loader = DataLoader(get_test_set(), batch_size=256, shuffle=True)
accuracy = evaluate(model_unquantized, test_loader, mel_spec_transform=mel_spec_transform)
print(f"Unquantized model accuracy: {accuracy:.2f}%")
# Complexity Calculator for PyTorch models aligned with:
# https://github.com/AlbertoAncilotto/NeSsi/blob/main/nessi.py
# we only copy the complexity calculation for torch models from NeSsi to avoid
# including an additional tensorflow dependency in this code base

import torch
import torchinfo

MAX_PARAMS_MEMORY = 128_000
MAX_MACS = 30_000_000


# def get_torch_size(model, input_data):
#     model_profile = torchinfo.summary(model, input_size=input_size)
#     return model_profile.total_mult_adds, model_profile.total_params

def get_torch_size(model, input_size, use_device=True):
    """
    Computes the total multiply-add operations (MACs) and total parameters for the model
    using torchinfo.summary.

    This function automatically constructs dummy inputs for the model. It uses the provided
    input_size to create a dummy audio input, and it creates a dummy device id tensor (of type long)
    based on the batch size from input_size.

    Args:
        model (torch.nn.Module): The model to be profiled.
        input_size (tuple): The shape of the primary input (e.g., the audio input). 
                            For example, (1, 1, 224, 224).

    Returns:
        total_mult_adds (int): The total number of multiply-add operations.
        total_params (int): The total number of parameters in the model.
    """
    # Create a dummy tensor for the first input using input_size.
    dummy_x = torch.randn(*input_size)
    
    # Assume the first dimension of input_size is the batch size.
    batch_size = input_size[0]
    # Create a dummy device id tensor with shape [batch_size]. 
    # For example, for a batch of size 1, this will be a tensor of shape [1].
    dummy_device_id = torch.zeros(batch_size, dtype=torch.long)
    
    # Use the input_data argument to pass both inputs.
    if use_device:
        model_profile = torchinfo.summary(model, input_data=(dummy_x, dummy_device_id))
    else:
        model_profile = torchinfo.summary(model, input_data=dummy_x)
    return model_profile.total_mult_adds, model_profile.total_params


# class ModelWrapper(torch.nn.Module):
#     """
#     A wrapper for AcousticSceneClassifier to provide a default device_id when calling torchinfo.summary().
#     """
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, x):
#         dummy_device_id = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)  # Dummy device ID
#         return self.model(x, dummy_device_id)

# def get_torch_size(model, input_size):
#     """
#     Computes the MACs and parameters for the given model using torchinfo.summary().
#     """
#     model_wrapper = ModelWrapper(model)  
#     model_profile = torchinfo.summary(model_wrapper, input_size=input_size)
#     return model_profile.total_mult_adds, model_profile.total_params  # Return MACs & Parameters

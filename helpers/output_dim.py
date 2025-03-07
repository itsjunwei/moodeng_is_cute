import torch

def get_model_output_dim(model):
    """
    Determines the output feature dimension of the baseline model (self.model)
    by running a dummy input through it.
    """
    dummy_input = torch.randn(1, 1, 256, 65)  # Match the input shape for mel_spec
    dummy_device_id = torch.tensor([0])  # Provide a dummy device ID (assuming IDs are 0-8)
    with torch.no_grad():
        output = model(dummy_input, dummy_device_id)
    return output.shape[1]  # Extract feature dimension
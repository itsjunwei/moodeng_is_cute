import torch

def get_model_output_dim(model):
    """
    Determines the output feature dimension of the baseline model (self.model)
    by running a dummy input through it.
    """
    dummy_input = torch.randn(1, 1, 256, 65)  # Match the input shape for mel_spec
    with torch.no_grad():
        output = model(dummy_input)
    return output.shape[1]  # Extract feature dimension
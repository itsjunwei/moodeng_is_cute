import torch
import torch.nn as nn

class AcousticSceneClassifier(nn.Module):
    def __init__(self, num_classes=10, device_embedding_dim=4):
        super(AcousticSceneClassifier, self).__init__()
        
        # Convolutional layers: input shape (batch, 1, 256, 65)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: (16, 128, 32)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: (32, 64, 16)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Output: (64, 32, 8)
        )
        
        # Embedding layer for device ID (values 1-9)
        self.device_embedding = nn.Embedding(num_embeddings=9, embedding_dim=device_embedding_dim)
        
        # Calculate flattened CNN output size: 64 * 32 * 8 = 16384
        cnn_flatten_size = 64 * 32 * 8
        
        # Fully connected layers: concatenate CNN features with device embedding
        self.fc_layers = nn.Sequential(
            nn.Linear(cnn_flatten_size + device_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)  # Output logits for 10 classes
        )
    
    def forward(self, x, device_id):
        """
        x: Tensor of shape (batch_size, 1, 256, 65)
        device_id: Tensor of shape (batch_size,) with values in 0-8 (device IDs)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten CNN features
        
        # Get device embedding: shape (batch_size, device_embedding_dim)
        device_emb = self.device_embedding(device_id)
        
        # Concatenate CNN features and device embedding
        x = torch.cat([x, device_emb], dim=1)
        
        # Pass through fully connected layers
        x = self.fc_layers(x)
        return x
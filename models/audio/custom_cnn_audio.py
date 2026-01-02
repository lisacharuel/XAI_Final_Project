"""
Custom CNN for Audio Classification
Simple but effective baseline model for deepfake audio detection

Architecture:
- 3 Convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool)
- Global Average Pooling
- Fully connected classifier with dropout

Input: (batch_size, 3, 128, 128) - RGB mel-spectrogram
Output: (batch_size, num_classes) - logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNNAudio(nn.Module):
    """
    Lightweight CNN for audio deepfake detection
    
    Args:
        num_classes (int): Number of output classes (default: 2 for Real/Fake)
        dropout_rate (float): Dropout probability (default: 0.5)
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(CustomCNNAudio, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional Block 1: 3 → 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128 → 64x64
        
        # Convolutional Block 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 → 32x32
        
        # Convolutional Block 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 → 16x16
        
        # Global Average Pooling instead of flatten
        # This reduces parameters and works with any input size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
    
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 128, 128)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size, 128, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 128)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    
    def get_features(self, x):
        """
        Extract feature maps from the last convolutional layer
        Useful for visualization and XAI methods like Grad-CAM
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Feature maps from conv3 layer
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        features = self.pool3(x)
        
        return features
    
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test function
if __name__ == "__main__":
    # Test the model
    model = CustomCNNAudio(num_classes=2)
    print(f"✓ CustomCNNAudio initialized")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output}")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"  Feature map shape: {features.shape}")
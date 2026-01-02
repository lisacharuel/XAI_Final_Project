"""
AlexNet for Chest X-Ray Classification
Modified AlexNet architecture for medical imaging

This implementation uses transfer learning from ImageNet-pretrained AlexNet
and adapts it for binary classification (Normal vs. Malignant)

Architecture:
- 5 Convolutional layers
- 3 Fully connected layers
- Final layer adapted for 2 classes (Normal/Malignant)

Input: (batch_size, 3, 224, 224) - RGB chest X-ray
Output: (batch_size, num_classes) - logits
"""

import torch
import torch.nn as nn
import torchvision.models as models


class AlexNetChest(nn.Module):
    """
    AlexNet adapted for chest X-ray classification
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Use ImageNet pretrained weights (default: True)
        freeze_features (bool): Freeze convolutional layers (default: False)
    """
    
    def __init__(self, num_classes=2, pretrained=True, freeze_features=False):
        super(AlexNetChest, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained AlexNet
        if pretrained:
            # Use weights parameter for newer PyTorch versions
            try:
                alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
                print("   ✓ Using ImageNet pretrained weights")
            except:
                # Fallback for older PyTorch versions
                alexnet = models.alexnet(pretrained=True)
                print("   ✓ Using ImageNet pretrained weights (legacy)")
        else:
            alexnet = models.alexnet(pretrained=False)
            print("   ⚠️  Using randomly initialized weights")
        
        # Extract features (convolutional layers)
        self.features = alexnet.features
        
        # Optionally freeze feature extractor
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
            print("   ⚠️  Feature layers frozen")
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = alexnet.avgpool
        
        # Modify classifier for our task
        # Original AlexNet has 1000 output classes, we need num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
        # Initialize the new classifier layers
        self._initialize_classifier()
    
    
    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.features(x)
        
        # Pooling
        x = self.avgpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    
    def get_features(self, x):
        """
        Extract feature maps from the last convolutional layer
        Useful for Grad-CAM and other XAI methods
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Feature maps from last conv layer
        """
        x = self.features(x)
        return x
    
    
    def get_conv_layers(self):
        """
        Get all convolutional layers (useful for Grad-CAM)
        
        Returns:
            List of convolutional layer indices and modules
        """
        conv_layers = []
        for idx, module in enumerate(self.features):
            if isinstance(module, nn.Conv2d):
                conv_layers.append((idx, module))
        return conv_layers
    
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test function
if __name__ == "__main__":
    # Test the model
    print("\n" + "="*70)
    print("Testing AlexNetChest")
    print("="*70)
    
    model = AlexNetChest(num_classes=2, pretrained=False)
    print(f"\n✓ AlexNetChest initialized")
    print(f"  Trainable parameters: {model.count_parameters():,}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\n  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output}")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"  Feature map shape: {features.shape}")
    
    # Show conv layers
    conv_layers = model.get_conv_layers()
    print(f"\n  Convolutional layers ({len(conv_layers)}):")
    for idx, layer in conv_layers:
        print(f"    Layer {idx}: {layer}")
    
    print("\n" + "="*70)
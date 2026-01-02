"""
Model Loader
Unified system for loading audio and image classification models

This module handles:
- Model instantiation
- Weight loading (with fallback to random initialization)
- Device management (CPU/GPU/Colab)
- Model inference utilities
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional
import warnings

from config import (
    DEVICE, 
    AUDIO_CONFIG, 
    IMAGE_CONFIG, 
    get_model_path,
    AUDIO_WEIGHTS_DIR,
    IMAGE_WEIGHTS_DIR
)


class ModelLoader:
    """
    Unified model loading system for audio and image models
    """
    
    def __init__(self):
        self.device = DEVICE
        self.loaded_models = {}  # Cache loaded models
    
    
    def load_audio_model(self, model_name: str, 
                        weights_path: Optional[Path] = None,
                        use_pretrained: bool = True) -> nn.Module:
        """
        Load an audio classification model
        
        Args:
            model_name: Name of the model ('vgg16', 'mobilenet', 'resnet', 'custom_cnn')
            weights_path: Optional path to model weights (overrides default)
            use_pretrained: Whether to load pretrained weights if available
        
        Returns:
            Loaded model on the correct device
        """
        print(f"\n🎵 Loading audio model: {model_name}")
        
        # Get model architecture
        if model_name == "custom_cnn":
            from models.audio.custom_cnn_audio import CustomCNNAudio
            model = CustomCNNAudio(num_classes=AUDIO_CONFIG["num_classes"])
            print(f"   ✓ CustomCNNAudio architecture loaded")
            
        elif model_name == "vgg16":
            # TODO: Implement when ready
            raise NotImplementedError("VGG16 audio model not yet implemented")
            
        elif model_name == "mobilenet":
            # TODO: Implement when ready
            raise NotImplementedError("MobileNet audio model not yet implemented")
            
        elif model_name == "resnet":
            # TODO: Implement when ready
            raise NotImplementedError("ResNet audio model not yet implemented")
        else:
            raise ValueError(f"Unknown audio model: {model_name}")
        
        # Load weights if available
        if use_pretrained:
            if weights_path is None:
                weights_path = get_model_path("audio", model_name)
            
            if weights_path.exists():
                try:
                    state_dict = torch.load(weights_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    print(f"   ✓ Loaded pretrained weights from {weights_path}")
                except Exception as e:
                    warnings.warn(f"Failed to load weights: {e}")
                    print(f"   ⚠️  Using randomly initialized weights")
            else:
                print(f"   ⚠️  No weights found at {weights_path}")
                print(f"   ⚠️  Using randomly initialized weights (for testing only)")
        else:
            print(f"   ⚠️  Using randomly initialized weights")
        
        # Move model to device
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        
        print(f"   ✓ Model moved to {self.device}")
        print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    
    def load_image_model(self, model_name: str,
                        weights_path: Optional[Path] = None,
                        use_pretrained: bool = True) -> nn.Module:
        """
        Load an image classification model
        
        Args:
            model_name: Name of the model ('alexnet', 'densenet')
            weights_path: Optional path to model weights
            use_pretrained: Whether to load pretrained weights
        
        Returns:
            Loaded model on the correct device
        """
        print(f"\n🖼️  Loading image model: {model_name}")
        
        # Get model architecture
        if model_name == "alexnet":
            from models.image.alexnet_image import AlexNetChest
            model = AlexNetChest(num_classes=IMAGE_CONFIG["num_classes"])
            print(f"   ✓ AlexNet architecture loaded")
            
        elif model_name == "densenet":
            # TODO: Implement when ready
            raise NotImplementedError("DenseNet image model not yet implemented")
        else:
            raise ValueError(f"Unknown image model: {model_name}")
        
        # Load weights if available
        if use_pretrained:
            if weights_path is None:
                weights_path = get_model_path("image", model_name)
            
            if weights_path.exists():
                try:
                    state_dict = torch.load(weights_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    print(f"   ✓ Loaded pretrained weights from {weights_path}")
                except Exception as e:
                    warnings.warn(f"Failed to load weights: {e}")
                    print(f"   ⚠️  Using randomly initialized weights")
            else:
                print(f"   ⚠️  No weights found at {weights_path}")
                print(f"   ⚠️  Using randomly initialized weights (for testing only)")
        else:
            print(f"   ⚠️  Using randomly initialized weights")
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        print(f"   ✓ Model moved to {self.device}")
        print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    
    def predict(self, model: nn.Module, input_tensor: torch.Tensor,
                class_names: list) -> Tuple[int, float, dict]:
        """
        Make a prediction with a model
        
        Args:
            model: The model to use
            input_tensor: Preprocessed input tensor
            class_names: List of class names
        
        Returns:
            Tuple of (predicted_class_idx, confidence, all_probabilities)
        """
        with torch.no_grad():
            # Forward pass
            logits = model(input_tensor)
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=1)
            
            # Get prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = predicted_idx.item()
            confidence_score = confidence.item()
            
            # Create probability dictionary
            prob_dict = {
                class_names[i]: probabilities[0][i].item() 
                for i in range(len(class_names))
            }
        
        return predicted_class, confidence_score, prob_dict
    
    
    def get_model_info(self, model: nn.Module) -> dict:
        """
        Get information about a model
        
        Args:
            model: The model
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(next(model.parameters()).device),
            "model_class": model.__class__.__name__
        }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create a global model loader instance
model_loader = ModelLoader()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_model(model_type: str, model_name: str, 
               use_pretrained: bool = True) -> nn.Module:
    """
    Quick function to load any model
    
    Args:
        model_type: 'audio' or 'image'
        model_name: Name of the model
        use_pretrained: Whether to use pretrained weights
    
    Returns:
        Loaded model
    """
    if model_type == "audio":
        return model_loader.load_audio_model(model_name, use_pretrained=use_pretrained)
    elif model_type == "image":
        return model_loader.load_image_model(model_name, use_pretrained=use_pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def quick_predict(model: nn.Module, input_tensor: torch.Tensor,
                 model_type: str) -> dict:
    """
    Quick prediction function
    
    Args:
        model: The model
        input_tensor: Preprocessed input
        model_type: 'audio' or 'image'
    
    Returns:
        Dictionary with prediction results
    """
    if model_type == "audio":
        class_names = AUDIO_CONFIG["classes"]
    elif model_type == "image":
        class_names = IMAGE_CONFIG["classes"]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    pred_idx, confidence, prob_dict = model_loader.predict(
        model, input_tensor, class_names
    )
    
    return {
        "predicted_class": class_names[pred_idx],
        "predicted_index": pred_idx,
        "confidence": confidence,
        "all_probabilities": prob_dict
    }


# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MODEL LOADER TEST")
    print("="*70)
    
    # Test loading audio model
    try:
        audio_model = model_loader.load_audio_model("custom_cnn", use_pretrained=False)
        print("\n✅ Audio model loaded successfully!")
        
        # Test with dummy input
        dummy_audio = torch.randn(1, 3, 128, 128).to(DEVICE)
        prediction = quick_predict(audio_model, dummy_audio, "audio")
        print(f"\n📊 Test Prediction:")
        print(f"   Predicted: {prediction['predicted_class']}")
        print(f"   Confidence: {prediction['confidence']:.2%}")
        print(f"   All probabilities: {prediction['all_probabilities']}")
        
    except Exception as e:
        print(f"\n❌ Error loading audio model: {e}")
    
    print("\n" + "="*70)
    print("Model loader ready to use!")
    print("="*70)
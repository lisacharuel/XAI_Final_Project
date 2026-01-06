"""
Model Loader
Functions for loading audio (.h5 Keras) and image (.safetensors PyTorch) models
"""

from pathlib import Path
import json
import numpy as np
import keras
import torch
from safetensors.torch import load_file
from transformers import ConvNextV2ForImageClassification, ConvNextV2Config
from transformers import AutoModelForImageClassification, AutoConfig

from config import AUDIO_MODELS_DIR, IMAGE_MODELS_DIR, AUDIO_CONFIG, IMAGE_CONFIG, TORCH_DEVICE


class ModelLoader:
    """
    Manages loading of audio and image classification models
    """
    
    def __init__(self):
        self._loaded_models = {}
    
    def load_audio_model(self, model_name: str):
        """
        Load an audio classification model from a .h5 file
        
        Args:
            model_name: Name of the model (e.g., 'vgg16', 'mobilenet', 'resnet50')
        
        Returns:
            Loaded Keras model
        """
        cache_key = f"audio_{model_name}"
        
        # Check if already loaded
        if cache_key in self._loaded_models:
            print(f"✓ Using cached model: {model_name}")
            return self._loaded_models[cache_key]
        
        # Validate model name
        if model_name not in AUDIO_CONFIG["models"]:
            available = list(AUDIO_CONFIG["models"].keys())
            raise ValueError(f"Unknown audio model: {model_name}. Available: {available}")
        
        # Get model path
        h5_file = AUDIO_CONFIG["models"][model_name]["h5_file"]
        model_path = AUDIO_MODELS_DIR / h5_file
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"🎵 Loading audio model from {model_path}")
        model = keras.models.load_model(model_path)
        print(f"✓ Audio model '{model_name}' loaded successfully")
        
        # Cache the model
        self._loaded_models[cache_key] = model
        
        return model
    
    def load_image_model(self, model_name: str):
        """
        Load an image classification model from a .safetensors file
        
        Args:
            model_name: Name of the model (e.g., 'convnext', 'densenet')
        
        Returns:
            Loaded PyTorch model
        """
        cache_key = f"image_{model_name}"
        
        # Check if already loaded
        if cache_key in self._loaded_models:
            print(f"✓ Using cached model: {model_name}")
            return self._loaded_models[cache_key]
        
        # Validate model name
        if model_name not in IMAGE_CONFIG["models"]:
            available = list(IMAGE_CONFIG["models"].keys())
            raise ValueError(f"Unknown image model: {model_name}. Available: {available}")
        
        # Get model info
        model_info = IMAGE_CONFIG["models"][model_name]
        safetensors_file = model_info["safetensors_file"]
        architecture = model_info["architecture"]
        model_path = IMAGE_MODELS_DIR / safetensors_file
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"🖼️  Loading image model from {model_path}")
        
        # Load config from JSON file if it exists
        config_file = model_path.with_suffix('.json')
        if architecture == "convnext":
            # Handle convnext naming (convenext.json vs convnext.safetensors)
            config_file = IMAGE_MODELS_DIR / "convenext.json"
        
        if architecture == "convnext":
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                config = ConvNextV2Config(**config_dict)
                model = ConvNextV2ForImageClassification(config)
            else:
                # Fallback: create default ConvNextV2 tiny model
                config = ConvNextV2Config(
                    num_labels=IMAGE_CONFIG["num_classes"],
                    hidden_sizes=[96, 192, 384, 768],
                    depths=[3, 3, 9, 3]
                )
                model = ConvNextV2ForImageClassification(config)
        elif architecture == "densenet":
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                # Use AutoConfig and AutoModel for DenseNet from transformers
                # Since transformers doesn't have native DenseNet, we'll use torchvision for densenet
                import torchvision.models as tv_models
                num_labels = config_dict.get("num_labels", IMAGE_CONFIG["num_classes"])
                model = tv_models.densenet121(weights=None)
                model.classifier = torch.nn.Linear(model.classifier.in_features, num_labels)
            else:
                import torchvision.models as tv_models
                model = tv_models.densenet121(weights=None)
                model.classifier = torch.nn.Linear(model.classifier.in_features, IMAGE_CONFIG["num_classes"])
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Load weights from safetensors
        state_dict = load_file(str(model_path))
        model.load_state_dict(state_dict, strict=False)
        
        # Move to device and set to eval mode
        model = model.to(TORCH_DEVICE)
        model.eval()
        
        print(f"✓ Image model '{model_name}' loaded successfully")
        
        # Cache the model
        self._loaded_models[cache_key] = model
        
        return model
    
    def get_available_audio_models(self) -> list:
        """Get list of available audio models"""
        return list(AUDIO_CONFIG["models"].keys())
    
    def get_available_image_models(self) -> list:
        """Get list of available image models"""
        return list(IMAGE_CONFIG["models"].keys())


# Global model loader instance
model_loader = ModelLoader()


def quick_predict(model, input_data, input_type: str = 'audio', model_name: str = None) -> dict:
    """
    Quick prediction function for both audio and image models
    
    Args:
        model: Loaded model (Keras for audio, PyTorch for image)
        input_data: Preprocessed input data
        input_type: Type of input ('audio' or 'image')
        model_name: Name of the model (for getting correct class labels)
    
    Returns:
        Dictionary with prediction results
    """
    if input_type == 'audio':
        return _predict_audio(model, input_data)
    elif input_type == 'image':
        return _predict_image(model, input_data, model_name)
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def _predict_audio(model, input_data: np.ndarray) -> dict:
    """Prediction for Keras audio models (sigmoid output)"""
    # Ensure batch dimension
    if input_data.ndim == 3:
        input_data = np.expand_dims(input_data, axis=0)
    
    # Get prediction (sigmoid output for binary classification)
    prediction = model.predict(input_data, verbose=0)
    
    # For binary classification with sigmoid output
    # prediction shape is (batch_size, 1)
    prob_fake = float(prediction[0][0])
    prob_real = 1.0 - prob_fake
    
    # Determine predicted class
    if prob_fake > 0.5:
        predicted_class = "Fake"
        confidence = prob_fake
    else:
        predicted_class = "Real"
        confidence = prob_real
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'predicted_index': 1 if predicted_class == "Fake" else 0,
        'all_probabilities': {
            'Real': prob_real,
            'Fake': prob_fake
        }
    }


def _predict_image(model, input_data: torch.Tensor, model_name: str = None) -> dict:
    """Prediction for PyTorch image models (softmax output)"""
    model.eval()
    
    with torch.no_grad():
        # Ensure on correct device
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.from_numpy(input_data).float()
        
        input_data = input_data.to(TORCH_DEVICE)
        
        # Add batch dimension if needed
        if input_data.dim() == 3:
            input_data = input_data.unsqueeze(0)
        
        # Get prediction
        output = model(input_data)
        
        # Handle both Hugging Face models (with .logits) and regular PyTorch models
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
            
        probs = torch.softmax(logits, dim=1)
    
    # Get class probabilities
    probs_np = probs.cpu().numpy()[0]
    
    # Get class names - use model-specific classes if available
    if model_name and model_name in IMAGE_CONFIG["models"]:
        class_names = IMAGE_CONFIG["models"][model_name].get("classes", IMAGE_CONFIG["classes"])
    else:
        class_names = IMAGE_CONFIG["classes"]
    
    # Ensure class_names matches probs length
    if len(class_names) != len(probs_np):
        class_names = [f"Class_{i}" for i in range(len(probs_np))]
    
    # Get predicted class
    predicted_idx = int(np.argmax(probs_np))
    predicted_class = class_names[predicted_idx]
    confidence = float(probs_np[predicted_idx])
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'predicted_index': predicted_idx,
        'all_probabilities': {
            class_names[i]: float(probs_np[i]) for i in range(len(class_names))
        }
    }
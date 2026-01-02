"""
Configuration file for Unified XAI Interface
Centralizes all settings, paths, and hyperparameters
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
SAMPLE_AUDIO_DIR = DATA_DIR / "sample_audio"
SAMPLE_IMAGE_DIR = DATA_DIR / "sample_images"

# Model directories
MODELS_DIR = BASE_DIR / "models"
AUDIO_MODELS_DIR = MODELS_DIR / "audio"
IMAGE_MODELS_DIR = MODELS_DIR / "image"

# Weights directory
WEIGHTS_DIR = BASE_DIR / "weights"
AUDIO_WEIGHTS_DIR = WEIGHTS_DIR / "audio_models"
IMAGE_WEIGHTS_DIR = WEIGHTS_DIR / "image_models"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

# Create directories if they don't exist
for directory in [DATA_DIR, SAMPLE_AUDIO_DIR, SAMPLE_IMAGE_DIR, 
                   MODELS_DIR, AUDIO_MODELS_DIR, IMAGE_MODELS_DIR,
                   WEIGHTS_DIR, AUDIO_WEIGHTS_DIR, IMAGE_WEIGHTS_DIR,
                   OUTPUT_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# AUDIO CONFIGURATION
# ============================================================================

AUDIO_CONFIG = {
    # File formats
    "supported_formats": [".wav", ".mp3", ".flac"],
    
    # Preprocessing
    "sample_rate": 16000,  # Standard for speech
    "duration": 3.0,  # seconds
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    
    # Spectrogram dimensions
    "spectrogram_height": 128,
    "spectrogram_width": 128,
    
    # Available models
    "models": {
        "vgg16": {
            "name": "VGG16 Audio",
            "weight_file": "vgg16_audio_best.pth",
            "description": "Deep CNN with 16 layers, good for complex patterns"
        },
        "mobilenet": {
            "name": "MobileNet Audio",
            "weight_file": "mobilenet_audio_best.pth",
            "description": "Lightweight model, faster inference"
        },
        "resnet": {
            "name": "ResNet Audio",
            "weight_file": "resnet_audio_best.pth",
            "description": "Residual network, handles deep architectures"
        },
        "custom_cnn": {
            "name": "Custom CNN Audio",
            "weight_file": "custom_cnn_audio_best.pth",
            "description": "Lightweight custom architecture"
        }
    },
    
    # Classification
    "classes": ["Real", "Fake"],
    "num_classes": 2
}


# ============================================================================
# IMAGE CONFIGURATION
# ============================================================================

IMAGE_CONFIG = {
    # File formats
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp"],
    
    # Preprocessing
    "image_size": (224, 224),
    "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
    "std": [0.229, 0.224, 0.225],
    
    # Available models
    "models": {
        "alexnet": {
            "name": "AlexNet",
            "weight_file": "alexnet_chest_best.pth",
            "description": "Classic CNN architecture, efficient"
        },
        "densenet": {
            "name": "DenseNet",
            "weight_file": "densenet_chest_best.pth",
            "description": "Dense connections, better gradient flow"
        }
    },
    
    # Classification
    "classes": ["Normal", "Malignant"],
    "num_classes": 2
}


# ============================================================================
# XAI CONFIGURATION
# ============================================================================

XAI_CONFIG = {
    # XAI methods and their compatibility
    "methods": {
        "lime": {
            "name": "LIME",
            "description": "Local Interpretable Model-agnostic Explanations",
            "compatible_with": ["audio", "image"],
            "parameters": {
                "num_samples": 1000,
                "num_features": 10
            }
        },
        "shap": {
            "name": "SHAP",
            "description": "SHapley Additive exPlanations",
            "compatible_with": ["audio", "image"],
            "parameters": {
                "num_samples": 100
            }
        },
        "gradcam": {
            "name": "Grad-CAM",
            "description": "Gradient-weighted Class Activation Mapping",
            "compatible_with": ["image"],  # Only for images
            "parameters": {
                "target_layer": -1  # Last conv layer
            }
        }
    }
}


# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

APP_CONFIG = {
    "title": "Unified Explainable AI Interface",
    "description": "Multi-modal classification with XAI for audio and image data",
    "version": "1.0.0",
    
    # Chainlit settings
    "chainlit": {
        "port": 8000,
        "host": "0.0.0.0",
        "debug": True
    },
    
    # Upload settings
    "max_file_size": 10 * 1024 * 1024,  # 10 MB
    "allowed_extensions": {
        "audio": AUDIO_CONFIG["supported_formats"],
        "image": IMAGE_CONFIG["supported_formats"]
    },
    
    # Visualization settings
    "figure_size": (12, 8),
    "dpi": 100
}


# ============================================================================
# MODEL URLS (for downloading pre-trained weights)
# ============================================================================

MODEL_URLS = {
    # These would be actual URLs to download weights
    # For now, placeholders
    "audio": {
        "vgg16": "https://example.com/vgg16_audio.pth",
        "mobilenet": "https://example.com/mobilenet_audio.pth",
        "resnet": "https://example.com/resnet_audio.pth",
        "custom_cnn": "https://example.com/custom_cnn_audio.pth"
    },
    "image": {
        "alexnet": "https://example.com/alexnet_chest.pth",
        "densenet": "https://example.com/densenet_chest.pth"
    }
}


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {DEVICE}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_path(model_type: str, model_name: str) -> Path:
    """
    Get the full path to a model weight file
    
    Args:
        model_type: 'audio' or 'image'
        model_name: name of the model (e.g., 'vgg16', 'alexnet')
    
    Returns:
        Path object to the weight file
    """
    if model_type == "audio":
        weight_file = AUDIO_CONFIG["models"][model_name]["weight_file"]
        return AUDIO_WEIGHTS_DIR / weight_file
    elif model_type == "image":
        weight_file = IMAGE_CONFIG["models"][model_name]["weight_file"]
        return IMAGE_WEIGHTS_DIR / weight_file
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_compatible_xai_methods(input_type: str) -> list:
    """
    Get list of XAI methods compatible with the input type
    
    Args:
        input_type: 'audio' or 'image'
    
    Returns:
        List of compatible XAI method names
    """
    compatible = []
    for method_key, method_info in XAI_CONFIG["methods"].items():
        if input_type in method_info["compatible_with"]:
            compatible.append(method_key)
    return compatible


def detect_input_type(file_extension: str) -> str:
    """
    Detect if uploaded file is audio or image
    
    Args:
        file_extension: file extension (e.g., '.wav', '.jpg')
    
    Returns:
        'audio' or 'image'
    """
    if file_extension.lower() in AUDIO_CONFIG["supported_formats"]:
        return "audio"
    elif file_extension.lower() in IMAGE_CONFIG["supported_formats"]:
        return "image"
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


# ============================================================================
# EXPORT ALL CONFIGS
# ============================================================================

__all__ = [
    'BASE_DIR',
    'DATA_DIR',
    'MODELS_DIR',
    'WEIGHTS_DIR',
    'OUTPUT_DIR',
    'AUDIO_CONFIG',
    'IMAGE_CONFIG',
    'XAI_CONFIG',
    'APP_CONFIG',
    'DEVICE',
    'get_model_path',
    'get_compatible_xai_methods',
    'detect_input_type'
]

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
AUDIO_DIR = DATA_DIR / "audio"
IMAGE_DIR = DATA_DIR / "images"

# Model directories
MODELS_DIR = BASE_DIR / "models"
AUDIO_MODELS_DIR = MODELS_DIR / "audio"
IMAGE_MODELS_DIR = MODELS_DIR / "image"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

# Create output directories if they don't exist
for directory in [OUTPUT_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# AUDIO CONFIGURATION
# ============================================================================

AUDIO_CONFIG = {
    # File formats
    "supported_formats": [".wav", ".mp3"],
    
    # Spectrogram dimensions (matching h5 models)
    "spectrogram_height": 224,
    "spectrogram_width": 224,
    
    # Available models (corresponding to .h5 files in models/audio)
    "models": {
        "vgg16": {
            "name": "VGG16",
            "h5_file": "vgg16.h5",
            "description": "Deep CNN with 16 layers, good for complex patterns"
        },
        "mobilenet": {
            "name": "MobileNet",
            "h5_file": "mobilenet.h5",
            "description": "Lightweight model, faster inference"
        },
        "resnet50": {
            "name": "ResNet50",
            "h5_file": "resnet50.h5",
            "description": "Residual network with 50 layers"
        }
    },
    
    # Classification (binary: 0=Real, 1=Fake)
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
    
    # Available models (corresponding to .safetensors files in models/image)
    "models": {
        "convnext": {
            "name": "ConvNeXt",
            "safetensors_file": "convnext.safetensors",
            "config_file": "convenext.json",
            "architecture": "convnext",
            "description": "Modern CNN architecture with transformer-inspired design",
            "num_classes": 5,
            "classes": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        },
        "densenet": {
            "name": "DenseNet",
            "safetensors_file": "densenet.safetensors",
            "config_file": "densenet.json",
            "architecture": "densenet",
            "description": "Dense connections, better gradient flow",
            "num_classes": 14,
            "classes": ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
                       "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
                       "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]
        }
    },
    
    # Classification (default, overridden by model-specific config)
    "classes": ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"],
    "num_classes": 5
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
        "gradcam": {
            "name": "Grad-CAM",
            "description": "Gradient-weighted Class Activation Mapping",
            "compatible_with": ["audio", "image"],
            "parameters": {
                "layer_name": "features.11"  # Example layer name for ConvNeXt
            }
        },
        "shap": {
            "name": "SHAP",
            "description": "SHapley Additive exPlanations",
            "compatible_with": ["audio", "image"],
            "parameters": {
                "num_samples": 100
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
# DEVICE CONFIGURATION
# ============================================================================

# TensorFlow device (for audio models)
import tensorflow as tf

# PyTorch device (for image models)
import torch
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus or torch.cuda.is_available():
    DEVICE = "GPU"
    print(f"🖥️  Using device: GPU")
else:
    DEVICE = "CPU"
    print(f"🖥️  Using device: CPU")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_audio_model_path(model_name: str) -> Path:
    """
    Get the full path to an audio model .h5 file
    
    Args:
        model_name: name of the model (e.g., 'vgg16', 'mobilenet')
    
    Returns:
        Path object to the .h5 file
    """
    if model_name not in AUDIO_CONFIG["models"]:
        raise ValueError(f"Unknown audio model: {model_name}")
    
    h5_file = AUDIO_CONFIG["models"][model_name]["h5_file"]
    return AUDIO_MODELS_DIR / h5_file


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
    'OUTPUT_DIR',
    'AUDIO_CONFIG',
    'IMAGE_CONFIG',
    'XAI_CONFIG',
    'APP_CONFIG',
    'DEVICE',
    'AUDIO_MODELS_DIR',
    'get_audio_model_path',
    'get_compatible_xai_methods',
    'detect_input_type'
]

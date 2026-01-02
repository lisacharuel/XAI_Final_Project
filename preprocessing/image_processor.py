"""
Image Processor
Handles image loading, preprocessing, and transformations for chest X-rays
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from config import IMAGE_CONFIG, DEVICE


class ImageProcessor:
    """
    Processes images for model input
    """
    
    def __init__(self):
        self.image_size = IMAGE_CONFIG["image_size"]
        self.mean = IMAGE_CONFIG["mean"]
        self.std = IMAGE_CONFIG["std"]
        
        # Define transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Transform without normalization (for visualization)
        self.transform_no_norm = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
    
    
    def load_image(self, file_path: Path) -> Image.Image:
        """
        Load image file
        
        Args:
            file_path: Path to image file
        
        Returns:
            PIL Image object
        """
        try:
            image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
    
    
    def preprocess(self, file_path: Path) -> Tuple[torch.Tensor, torch.Tensor, Image.Image]:
        """
        Complete preprocessing pipeline for image file
        
        Args:
            file_path: Path to image file
        
        Returns:
            Tuple of (normalized_tensor, unnormalized_tensor, original_image)
            - normalized_tensor: Ready for model inference (batch, channels, H, W)
            - unnormalized_tensor: For visualization (no normalization)
            - original_image: PIL Image for reference
        """
        # Load image
        image = self.load_image(file_path)
        
        # Apply transformations
        normalized_tensor = self.transform(image)
        unnormalized_tensor = self.transform_no_norm(image)
        
        # Add batch dimension
        normalized_tensor = normalized_tensor.unsqueeze(0)  # (1, 3, H, W)
        unnormalized_tensor = unnormalized_tensor.unsqueeze(0)
        
        # Move to device
        normalized_tensor = normalized_tensor.to(DEVICE)
        unnormalized_tensor = unnormalized_tensor.to(DEVICE)
        
        return normalized_tensor, unnormalized_tensor, image
    
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor for visualization
        
        Args:
            tensor: Normalized tensor
        
        Returns:
            Denormalized tensor
        """
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        
        denormalized = tensor * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized
    
    
    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array for visualization
        
        Args:
            tensor: Image tensor (C, H, W) or (1, C, H, W)
        
        Returns:
            Numpy array (H, W, C) in [0, 255] range
        """
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Move to CPU and convert to numpy
        array = tensor.cpu().detach().numpy()
        
        # Transpose from (C, H, W) to (H, W, C)
        array = np.transpose(array, (1, 2, 0))
        
        # Scale to [0, 255]
        array = (array * 255).astype(np.uint8)
        
        return array
    
    
    def visualize_image(self, image: Image.Image, 
                       title: str = "Input Image",
                       save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create visualization of image
        
        Args:
            image: PIL Image to visualize
            title: Plot title
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.imshow(image, cmap='gray' if image.mode == 'L' else None)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    
    def visualize_tensor(self, tensor: torch.Tensor,
                        title: str = "Processed Image",
                        save_path: Optional[Path] = None) -> plt.Figure:
        """
        Visualize a tensor
        
        Args:
            tensor: Image tensor
            title: Plot title
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        # Convert to numpy array
        array = self.tensor_to_image(tensor)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(array)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    
    def get_image_info(self, file_path: Path) -> dict:
        """
        Get information about image file
        
        Args:
            file_path: Path to image file
        
        Returns:
            Dictionary with image information
        """
        try:
            image = Image.open(file_path)
            
            return {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "resized_to": self.image_size
            }
        except Exception as e:
            return {"error": str(e)}
    
    
    def apply_augmentation(self, image: Image.Image, 
                          augmentation: str) -> Image.Image:
        """
        Apply augmentation to image
        
        Args:
            image: PIL Image
            augmentation: Type of augmentation ('flip', 'rotate', 'brightness', etc.)
        
        Returns:
            Augmented image
        """
        if augmentation == 'flip':
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif augmentation == 'rotate_90':
            return image.rotate(90)
        elif augmentation == 'rotate_180':
            return image.rotate(180)
        elif augmentation == 'rotate_270':
            return image.rotate(270)
        else:
            return image


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create a global image processor instance
image_processor = ImageProcessor()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def process_image_file(file_path: Path) -> Tuple[torch.Tensor, torch.Tensor, Image.Image]:
    """
    Quick function to process an image file
    
    Args:
        file_path: Path to image file
    
    Returns:
        Tuple of (normalized_tensor, unnormalized_tensor, original_image)
    """
    return image_processor.preprocess(file_path)


# Test function
if __name__ == "__main__":
    print("Image Processor initialized")
    print(f"Image size: {image_processor.image_size}")
    print(f"Normalization mean: {image_processor.mean}")
    print(f"Normalization std: {image_processor.std}")

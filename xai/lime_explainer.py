"""
LIME Explainer for Audio and Images
Local Interpretable Model-agnostic Explanations

LIME works by:
1. Perturbing the input (add noise, mask regions)
2. Getting predictions for perturbed inputs
3. Fitting a simple linear model locally
4. Identifying which features are most important

Supports:
- Audio spectrograms
- Chest X-ray images
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from pathlib import Path
from typing import Tuple, Optional

from config import DEVICE


class LIMEExplainer:
    """
    LIME explainer for both audio spectrograms and medical images
    """
    
    def __init__(self, num_samples=1000, num_features=10):
        """
        Initialize LIME explainer
        
        Args:
            num_samples: Number of perturbed samples to generate
            num_features: Number of top features to show
        """
        self.num_samples = num_samples
        self.num_features = num_features
        
        # Create LIME image explainer
        self.explainer = lime_image.LimeImageExplainer()
        
        print(f"✓ LIME Explainer initialized")
        print(f"  Samples: {num_samples}")
        print(f"  Top features: {num_features}")
    
    
    def explain_image(self, model: torch.nn.Module, 
                     image_tensor: torch.Tensor,
                     original_image: np.ndarray,
                     class_names: list,
                     hide_color: int = 0) -> Tuple[np.ndarray, dict]:
        """
        Generate LIME explanation for an image
        
        Args:
            model: The classification model
            image_tensor: Preprocessed image tensor (normalized)
            original_image: Original image as numpy array (H, W, C) in [0, 255]
            class_names: List of class names
            hide_color: Color to use for hidden regions (0 for black)
        
        Returns:
            Tuple of (explanation_image, importance_scores)
        """
        print(f"\n🔍 Generating LIME explanation for image...")
        
        # Prepare prediction function
        def predict_fn(images):
            """
            Prediction function for LIME
            Takes images in [0, 1] range
            Returns probabilities
            """
            # Convert to torch tensors
            batch = []
            for img in images:
                # Normalize using ImageNet stats (same as preprocessing)
                from config import IMAGE_CONFIG
                mean = np.array(IMAGE_CONFIG["mean"]).reshape(1, 1, 3)
                std = np.array(IMAGE_CONFIG["std"]).reshape(1, 1, 3)
                
                # Normalize
                normalized = (img - mean) / std
                
                # Convert to tensor and transpose to (C, H, W)
                tensor = torch.from_numpy(normalized).float()
                tensor = tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                batch.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(batch).to(DEVICE)
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
            
            return probs.cpu().numpy()
        
        # Convert original image to [0, 1] range if needed
        if original_image.max() > 1:
            original_image = original_image / 255.0
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            original_image,
            predict_fn,
            top_labels=len(class_names),
            hide_color=hide_color,
            num_samples=self.num_samples
        )
        
        # Get the top class
        top_class = explanation.top_labels[0]
        
        # Get explanation image and mask
        temp, mask = explanation.get_image_and_mask(
            top_class,
            positive_only=False,
            num_features=self.num_features,
            hide_rest=False
        )
        
        # Create visualization with boundaries
        explanation_image = mark_boundaries(temp, mask)
        
        # Get feature importance scores
        importance_scores = dict(explanation.local_exp[top_class])
        
        print(f"   ✓ LIME explanation generated")
        print(f"   ✓ Explaining class: {class_names[top_class]}")
        print(f"   ✓ Number of superpixels: {len(importance_scores)}")
        
        return explanation_image, importance_scores
    
    
    def explain_audio_spectrogram(self, model: torch.nn.Module,
                                  spectrogram_tensor: torch.Tensor,
                                  original_spectrogram: np.ndarray,
                                  class_names: list) -> Tuple[np.ndarray, dict]:
        """
        Generate LIME explanation for audio spectrogram
        
        Args:
            model: The classification model
            spectrogram_tensor: Preprocessed spectrogram tensor
            original_spectrogram: Original spectrogram (H, W) normalized to [0, 1]
            class_names: List of class names
        
        Returns:
            Tuple of (explanation_image, importance_scores)
        """
        print(f"\n🔍 Generating LIME explanation for audio spectrogram...")
        
        # Convert grayscale spectrogram to RGB for LIME
        if original_spectrogram.ndim == 2:
            original_rgb = np.stack([original_spectrogram] * 3, axis=-1)
        else:
            original_rgb = original_spectrogram
        
        # Prepare prediction function
        def predict_fn(images):
            """Prediction function for audio spectrograms"""
            batch = []
            for img in images:
                # Convert back to single channel if needed
                if img.shape[-1] == 3:
                    # Take first channel (they're all the same)
                    single_channel = img[:, :, 0]
                else:
                    single_channel = img
                
                # Convert to tensor with 3 channels
                tensor = torch.from_numpy(single_channel).float()
                tensor = tensor.unsqueeze(0).repeat(3, 1, 1)  # (1, H, W) -> (3, H, W)
                batch.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(batch).to(DEVICE)
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
            
            return probs.cpu().numpy()
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            original_rgb,
            predict_fn,
            top_labels=len(class_names),
            hide_color=0,
            num_samples=self.num_samples
        )
        
        # Get the top class
        top_class = explanation.top_labels[0]
        
        # Get explanation image and mask
        temp, mask = explanation.get_image_and_mask(
            top_class,
            positive_only=False,
            num_features=self.num_features,
            hide_rest=False
        )
        
        # Create visualization
        explanation_image = mark_boundaries(temp, mask)
        
        # Get importance scores
        importance_scores = dict(explanation.local_exp[top_class])
        
        print(f"   ✓ LIME explanation generated for spectrogram")
        print(f"   ✓ Explaining class: {class_names[top_class]}")
        
        return explanation_image, importance_scores
    
    
    def visualize_explanation(self, original_image: np.ndarray,
                            explanation_image: np.ndarray,
                            importance_scores: dict,
                            prediction_result: dict,
                            title: str = "LIME Explanation",
                            save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive visualization of LIME explanation
        
        Args:
            original_image: Original input image
            explanation_image: LIME explanation overlay
            importance_scores: Dictionary of feature importance scores
            prediction_result: Prediction results from model
            title: Plot title
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image, cmap='gray' if original_image.ndim == 2 else None)
        axes[0].set_title('Original Input', fontweight='bold', fontsize=12)
        axes[0].axis('off')
        
        # LIME explanation
        axes[1].imshow(explanation_image)
        axes[1].set_title('LIME Explanation', fontweight='bold', fontsize=12)
        pred_text = f"Predicted: {prediction_result['predicted_class']}\n"
        pred_text += f"Confidence: {prediction_result['confidence']:.1%}"
        axes[1].text(0.5, -0.1, pred_text, transform=axes[1].transAxes,
                    ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].axis('off')
        
        # Feature importance
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: abs(x[1]), 
                               reverse=True)[:10]
        features, scores = zip(*sorted_features)
        
        colors = ['green' if s > 0 else 'red' for s in scores]
        axes[2].barh(range(len(scores)), scores, color=colors, alpha=0.6)
        axes[2].set_yticks(range(len(scores)))
        axes[2].set_yticklabels([f"Region {f}" for f in features])
        axes[2].set_xlabel('Importance Score', fontweight='bold')
        axes[2].set_title('Top Features', fontweight='bold', fontsize=12)
        axes[2].axvline(x=0, color='black', linestyle='--', alpha=0.3)
        axes[2].grid(axis='x', alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved visualization to {save_path}")
        
        return fig


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def explain_with_lime(model, input_tensor, original_data, 
                     input_type: str, class_names: list,
                     prediction_result: dict,
                     save_path: Optional[Path] = None):
    """
    Quick function to generate LIME explanation
    
    Args:
        model: Classification model
        input_tensor: Preprocessed input tensor
        original_data: Original data (numpy array)
        input_type: 'audio' or 'image'
        class_names: List of class names
        prediction_result: Prediction results
        save_path: Optional path to save visualization
    
    Returns:
        Tuple of (explanation_image, importance_scores, figure)
    """
    explainer = LIMEExplainer()
    
    if input_type == "audio":
        explanation_img, scores = explainer.explain_audio_spectrogram(
            model, input_tensor, original_data, class_names
        )
    elif input_type == "image":
        explanation_img, scores = explainer.explain_image(
            model, input_tensor, original_data, class_names
        )
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    
    # Create visualization
    fig = explainer.visualize_explanation(
        original_data, explanation_img, scores, 
        prediction_result, save_path=save_path
    )
    
    return explanation_img, scores, fig


# Test
if __name__ == "__main__":
    print("LIME Explainer initialized and ready to use!")
    print("\nExample usage:")
    print("  explainer = LIMEExplainer()")
    print("  explanation, scores = explainer.explain_image(model, tensor, image, classes)")
"""
LIME Explainer for Audio Spectrograms and Images
Local Interpretable Model-agnostic Explanations

LIME works by:
1. Perturbing the input (add noise, mask regions)
2. Getting predictions for perturbed inputs
3. Fitting a simple linear model locally
4. Identifying which features are most important

Supports:
- Audio spectrograms (Keras models with sigmoid output)
- Images (PyTorch models with softmax output)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from pathlib import Path
from typing import Tuple, Optional

from config import IMAGE_CONFIG, TORCH_DEVICE


class LIMEExplainer:
    """
    LIME explainer for audio spectrograms and images
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
    
    
    def explain_audio_spectrogram(self, model,
                                  spectrogram_tensor: np.ndarray,
                                  original_spectrogram: np.ndarray,
                                  class_names: list) -> Tuple[np.ndarray, dict]:
        """
        Generate LIME explanation for audio spectrogram
        
        Args:
            model: The Keras classification model
            spectrogram_tensor: Preprocessed spectrogram (224, 224, 3)
            original_spectrogram: Original spectrogram for visualization
            class_names: List of class names ['Real', 'Fake']
        
        Returns:
            Tuple of (explanation_image, importance_scores)
        """
        print(f"\n🔍 Generating LIME explanation for audio spectrogram...")
        
        # Normalize spectrogram_tensor to [0, 1] for LIME
        spec_min = spectrogram_tensor.min()
        spec_max = spectrogram_tensor.max()
        if spec_max > spec_min:
            normalized_spec = (spectrogram_tensor - spec_min) / (spec_max - spec_min)
        else:
            normalized_spec = np.zeros_like(spectrogram_tensor)
        
        # Prepare prediction function for LIME
        def predict_fn(images):
            """
            Prediction function for LIME
            Takes images in [0, 1] range, returns probabilities for all classes
            """
            # Denormalize back to original range for model
            batch = []
            for img in images:
                denorm = img * (spec_max - spec_min) + spec_min
                batch.append(denorm)
            
            batch_array = np.array(batch)
            
            # Get predictions (sigmoid output)
            predictions = model.predict(batch_array, verbose=0)
            
            # Convert sigmoid to two-class probabilities
            # predictions shape: (batch_size, 1)
            prob_fake = predictions.flatten()
            prob_real = 1.0 - prob_fake
            
            # Return shape: (batch_size, num_classes)
            return np.column_stack([prob_real, prob_fake])
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            normalized_spec,
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
    
    
    def explain_image(self, model,
                     image_tensor: torch.Tensor,
                     original_image: np.ndarray,
                     class_names: list) -> Tuple[np.ndarray, dict]:
        """
        Generate LIME explanation for an image
        
        Args:
            model: The PyTorch classification model
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)
            original_image: Original image as numpy array (H, W, C) in [0, 255]
            class_names: List of class names
        
        Returns:
            Tuple of (explanation_image, importance_scores)
        """
        print(f"\n🔍 Generating LIME explanation for image...")
        
        # Prepare prediction function for LIME
        def predict_fn(images):
            """
            Prediction function for LIME
            Takes images in [0, 1] range, returns probabilities for all classes
            """
            model.eval()
            batch = []
            
            for img in images:
                # Normalize using ImageNet stats
                mean = np.array(IMAGE_CONFIG["mean"]).reshape(1, 1, 3)
                std = np.array(IMAGE_CONFIG["std"]).reshape(1, 1, 3)
                
                # Normalize
                normalized = (img - mean) / std
                
                # Convert to tensor and transpose to (C, H, W)
                tensor = torch.from_numpy(normalized).float()
                tensor = tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                batch.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(batch).to(TORCH_DEVICE)
            
            # Get predictions
            with torch.no_grad():
                output = model(batch_tensor)
                # Handle both Hugging Face models (with .logits) and regular PyTorch models
                if hasattr(output, 'logits'):
                    logits = output.logits
                else:
                    logits = output
                probs = torch.softmax(logits, dim=1)
            
            return probs.cpu().numpy()
        
        # Convert original image to [0, 1] range if needed
        if original_image.max() > 1:
            original_image_norm = original_image / 255.0
        else:
            original_image_norm = original_image
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            original_image_norm,
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
        
        # Create visualization with boundaries
        explanation_image = mark_boundaries(temp, mask)
        
        # Get feature importance scores
        importance_scores = dict(explanation.local_exp[top_class])
        
        print(f"   ✓ LIME explanation generated for image")
        print(f"   ✓ Explaining class: {class_names[top_class]}")
        print(f"   ✓ Number of superpixels: {len(importance_scores)}")
        
        return explanation_image, importance_scores
    
    
    def visualize_explanation(self, original_data: np.ndarray,
                            explanation_image: np.ndarray,
                            importance_scores: dict,
                            prediction_result: dict,
                            input_type: str = "audio",
                            title: str = "LIME Explanation",
                            save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive visualization of LIME explanation
        
        Args:
            original_data: Original input data
            explanation_image: LIME explanation overlay
            importance_scores: Dictionary of feature importance scores
            prediction_result: Prediction results from model
            input_type: 'audio' or 'image'
            title: Plot title
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original data
        if input_type == "audio":
            if original_data.ndim == 3:
                axes[0].imshow(original_data[:, :, 0], cmap='viridis', aspect='auto')
            else:
                axes[0].imshow(original_data, cmap='viridis', aspect='auto')
            axes[0].set_title('Original Spectrogram', fontweight='bold', fontsize=12)
        else:
            axes[0].imshow(original_data)
            axes[0].set_title('Original Image', fontweight='bold', fontsize=12)
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
        if importance_scores:
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
        else:
            axes[2].text(0.5, 0.5, 'No features found', ha='center', va='center')
            axes[2].set_title('Top Features', fontweight='bold', fontsize=12)
        
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
        model: Classification model (Keras for audio, PyTorch for image)
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
        raise ValueError(f"Unsupported input type: {input_type}")
    
    # Create visualization
    fig = explainer.visualize_explanation(
        original_data, explanation_img, scores, 
        prediction_result, input_type=input_type, save_path=save_path
    )
    
    return explanation_img, scores, fig


# Test
if __name__ == "__main__":
    print("LIME Explainer initialized and ready to use!")
    print("\nExample usage:")
    print("  explainer = LIMEExplainer()")
    print("  explanation, scores = explainer.explain_audio_spectrogram(model, tensor, spec, classes)")
    print("  explanation, scores = explainer.explain_image(model, tensor, image, classes)")
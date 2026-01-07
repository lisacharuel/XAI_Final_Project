import numpy as np
import torch
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import Model
from typing import Optional, Tuple
from config import TORCH_DEVICE, IMAGE_CONFIG


class SHAPExplainer:
    def __init__(self, nsamples=50):  # Reduced samples for faster computation
        self.nsamples = nsamples
        print(f"✓ SHAP Explainer initialized (nsamples={nsamples})")

    def explain_audio(self, model: Model, spectrogram_tensor: np.ndarray):
        """Generate SHAP-like explanation for audio spectrogram (Keras model) using gradients"""
        import tensorflow as tf
        
        # Ensure batch dimension exists
        if spectrogram_tensor.ndim == 3:
            spectrogram_tensor = np.expand_dims(spectrogram_tensor, axis=0)
        
        # Use gradient-based attribution (similar to Integrated Gradients)
        # This is more reliable than KernelExplainer for high-dimensional data
        input_tensor = tf.convert_to_tensor(spectrogram_tensor, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = model(input_tensor)
            # For sigmoid output, use the prediction directly
            if predictions.shape[-1] == 1:
                target = predictions[:, 0]
            else:
                target = predictions[:, 1]  # Class 1 (Fake)
        
        # Get gradients
        grads = tape.gradient(target, input_tensor)
        
        # Compute attribution: input * gradient (simple gradient-based attribution)
        attributions = input_tensor.numpy() * grads.numpy()
        
        return attributions

    def explain_image(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Generate SHAP explanation for image (PyTorch model) using GradientExplainer"""
        model.eval()
        
        # Denormalize input for visualization
        mean = np.array(IMAGE_CONFIG["mean"]).reshape(1, 1, 3)
        std = np.array(IMAGE_CONFIG["std"]).reshape(1, 1, 3)
        input_np = input_tensor.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, C)
        input_display = input_np * std + mean  # Denormalize
        input_display = np.clip(input_display, 0, 1)
        
        # Use GradientExplainer for PyTorch models (works with image tensors)
        # Create a wrapper to handle Hugging Face model outputs
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                out = self.model(x)
                if hasattr(out, 'logits'):
                    return out.logits
                return out
        
        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()
        
        # Background: use black image (normalized)
        background = torch.zeros((1, 3, 224, 224)).to(TORCH_DEVICE)
        
        try:
            # Try GradientExplainer first (faster, gradient-based)
            explainer = shap.GradientExplainer(wrapped_model, background)
            shap_values = explainer.shap_values(input_tensor)
        except Exception as e:
            print(f"GradientExplainer failed: {e}, falling back to simple gradient-based attribution")
            # Fallback: compute simple gradient-based attribution
            input_tensor.requires_grad_(True)
            output = wrapped_model(input_tensor)
            target_class = output.argmax(dim=1).item()
            loss = output[0, target_class]
            loss.backward()
            
            # Get gradients as SHAP-like values
            grads = input_tensor.grad.cpu().numpy()
            shap_values = [grads]  # Wrap in list to match SHAP format
            input_tensor.requires_grad_(False)
        
        return shap_values, input_display


def explain_with_shap(model, input_tensor, input_type='image', save_path: Optional[Path] = None) -> Tuple:
    """
    Generate SHAP explanation with visualization
    
    Args:
        model: Classification model (Keras for audio, PyTorch for image)
        input_tensor: Preprocessed input tensor
        input_type: 'audio' or 'image'
        save_path: Optional path to save visualization
    
    Returns:
        Tuple of (explanation_image, importance_scores, figure)
    """
    explainer = SHAPExplainer(nsamples=50)
    
    print(f"\n🔍 Generating SHAP explanation for {input_type}...")
    
    if input_type == 'audio':
        shap_values = explainer.explain_audio(model, input_tensor)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original spectrogram
        if input_tensor.ndim == 4:
            spec_display = input_tensor[0, :, :, 0]
        elif input_tensor.ndim == 3:
            spec_display = input_tensor[:, :, 0]
        else:
            spec_display = input_tensor
        axes[0].imshow(spec_display, cmap='viridis', aspect='auto')
        axes[0].set_title('Original Spectrogram', fontweight='bold')
        axes[0].axis('off')
        
        # SHAP values heatmap - handle gradient-based attribution output
        # shap_values shape: (1, 224, 224, 3) - take first sample and average across channels
        if shap_values.ndim == 4:
            shap_display = np.abs(shap_values[0]).mean(axis=-1)  # (224, 224)
        elif shap_values.ndim == 3:
            shap_display = np.abs(shap_values).mean(axis=-1)
        else:
            shap_display = np.abs(shap_values)
        
        # Normalize for display
        shap_display = (shap_display - shap_display.min()) / (shap_display.max() - shap_display.min() + 1e-8)
        
        axes[1].imshow(shap_display, cmap='hot', aspect='auto')
        axes[1].set_title('Gradient Attribution', fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(spec_display, cmap='viridis', aspect='auto')
        axes[2].imshow(shap_display, cmap='hot', alpha=0.5, aspect='auto')
        axes[2].set_title('Attribution Overlay', fontweight='bold')
        axes[2].axis('off')
        
        explanation_img = shap_display
        scores = {'mean_importance': float(np.abs(shap_values).mean())}
        
    elif input_type == 'image':
        shap_values, input_img = explainer.explain_image(model, input_tensor)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(input_img)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # SHAP values - handle different formats from GradientExplainer
        # Format can be: list of arrays (per class), single array, or (batch, C, H, W, num_classes)
        try:
            if isinstance(shap_values, list) and len(shap_values) > 0:
                # List of arrays per class - take first class or predicted class
                sv = shap_values[0]  # Shape: (batch, C, H, W)
                if sv.ndim == 4:
                    # (batch, C, H, W) -> take first sample, sum across channels
                    shap_for_display = np.abs(sv[0]).sum(axis=0)  # (H, W)
                elif sv.ndim == 3:
                    shap_for_display = np.abs(sv).sum(axis=0)  # (H, W)
                else:
                    shap_for_display = np.abs(sv)
            elif isinstance(shap_values, np.ndarray):
                # Single array - check dimensions
                if shap_values.ndim == 5:
                    # Shape: (batch, C, H, W, num_classes) - take first sample, first class, sum channels
                    sv = shap_values[0, :, :, :, 0]  # (C, H, W)
                    shap_for_display = np.abs(sv).sum(axis=0)  # (H, W)
                elif shap_values.ndim == 4:
                    shap_for_display = np.abs(shap_values[0]).sum(axis=0)
                elif shap_values.ndim == 3:
                    shap_for_display = np.abs(shap_values).sum(axis=0)
                else:
                    shap_for_display = np.abs(shap_values)
            else:
                # Fallback
                shap_for_display = np.zeros((224, 224))
        except Exception as e:
            print(f"Error processing SHAP values: {e}")
            # Fallback: create a simple heatmap
            shap_for_display = np.zeros((224, 224))
        
        # Ensure 2D for display
        if shap_for_display.ndim != 2:
            print(f"Warning: shap_for_display has shape {shap_for_display.shape}, forcing to 2D")
            shap_for_display = shap_for_display.reshape(224, 224) if shap_for_display.size == 224*224 else np.zeros((224, 224))
        
        # Normalize for display
        shap_for_display = (shap_for_display - shap_for_display.min()) / (shap_for_display.max() - shap_for_display.min() + 1e-8)
        
        axes[1].imshow(shap_for_display, cmap='hot')
        axes[1].set_title('SHAP Importance Heatmap', fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(input_img)
        axes[2].imshow(shap_for_display, cmap='hot', alpha=0.5)
        axes[2].set_title('SHAP Overlay', fontweight='bold')
        axes[2].axis('off')
        
        explanation_img = shap_for_display
        try:
            if isinstance(shap_values, list):
                scores = {'mean_importance': float(np.abs(shap_values[0]).mean())}
            else:
                scores = {'mean_importance': float(np.abs(shap_values).mean())}
        except:
            scores = {'mean_importance': 0.0}
    else:
        raise ValueError(f"Unsupported input type: {input_type}")
    
    plt.suptitle('SHAP Explanation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # Close figure to free memory and flush file
        print(f"   ✓ SHAP visualization saved to {save_path}")
    
    print(f"   ✓ SHAP explanation generated")
    
    return explanation_img, scores, fig


if __name__ == "__main__":
    print("SHAP Explainer ready for images and audio")

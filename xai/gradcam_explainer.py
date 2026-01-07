"""
Grad-CAM Explainer for Audio Spectrograms and Images
Supports:
- Audio: Keras models (sigmoid output)
- Image: PyTorch models (softmax output)
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
from torchvision import models

# For audio (Keras)
from tensorflow.keras.models import Model

from config import TORCH_DEVICE, IMAGE_CONFIG

# ============================================================================

class GradCAMExplainer:
    
    def __init__(self):
        print("✓ Grad-CAM Explainer initialized")

    # ------------------- AUDIO (Keras) ------------------- #
    def explain_audio(self, model: Model, spectrogram_tensor: np.ndarray, class_index: int = 1,
                      layer_name: Optional[str] = None) -> np.ndarray:
        
        import tensorflow as tf
        
        # Ensure batch dimension exists
        if spectrogram_tensor.ndim == 3:
            spectrogram_tensor = np.expand_dims(spectrogram_tensor, axis=0)
        
        # Convert to tensor early
        input_tensor = tf.convert_to_tensor(spectrogram_tensor, dtype=tf.float32)
        
        # Build the model if it hasn't been called yet (needed for Sequential models)
        try:
            _ = model.inputs
        except AttributeError:
            # Model hasn't been built, call it once to build
            _ = model(input_tensor)
        
        if layer_name is None:
            # Find last Conv2D layer - check for various naming conventions
            conv_layer = None
            for l in reversed(model.layers):
                layer_type = l.__class__.__name__
                # Check if it's a convolutional layer by type or name
                if layer_type in ['Conv2D', 'DepthwiseConv2D', 'SeparableConv2D']:
                    conv_layer = l
                    layer_name = l.name
                    break
                # Also check by name patterns (some models use different naming)
                if any(pattern in l.name.lower() for pattern in ['conv', 'conv2d', 'depthwise']):
                    conv_layer = l
                    layer_name = l.name
                    break
            
            if layer_name is None:
                # Fallback: find any layer with 4D output (likely a conv layer)
                for l in reversed(model.layers):
                    try:
                        if hasattr(l, 'output') and len(l.output.shape) == 4 and l.output.shape[-1] is not None:
                            if l.output.shape[-1] > 1:  # Has multiple channels
                                conv_layer = l
                                layer_name = l.name
                                break
                    except:
                        continue
            
            if layer_name is None:
                raise ValueError("Could not find a convolutional layer in the model. Available layers: " + 
                               str([l.name for l in model.layers]))
        
        conv_layer = model.get_layer(layer_name)
        print(f"   Using layer '{layer_name}' for Grad-CAM")
        
        # Create gradient model - handle both Functional and Sequential models
        try:
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[conv_layer.output, model.output]
            )
        except Exception as e:
            print(f"   Warning: Could not create grad_model directly: {e}")
            # Fallback for Sequential models: use GradientTape with intermediate outputs
            return self._explain_audio_fallback(model, conv_layer, input_tensor, class_index)

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            conv_outputs, predictions = grad_model(input_tensor)
            # Handle sigmoid output (single value per sample)
            if predictions.shape[-1] == 1:
                loss = predictions[:, 0] if class_index == 1 else (1 - predictions[:, 0])
            else:
                loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Use TensorFlow operations instead of item assignment (tensors are immutable)
        # Multiply each channel by its corresponding weight
        conv_outputs_weighted = conv_outputs[0] * pooled_grads  # Broadcasting: (H, W, C) * (C,)

        heatmap = tf.reduce_sum(conv_outputs_weighted, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    def _explain_audio_fallback(self, model, conv_layer, input_tensor, class_index: int = 1) -> np.ndarray:
        """Fallback Grad-CAM for Sequential models using layer-by-layer execution"""
        import tensorflow as tf
        
        print(f"   Using fallback Grad-CAM method for Sequential model")
        
        # Create intermediate model to get conv layer output
        intermediate_outputs = []
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(input_tensor)
            
            # Forward pass through the model, capturing intermediate output
            x = input_tensor
            conv_output = None
            
            for layer in model.layers:
                x = layer(x)
                if layer.name == conv_layer.name:
                    conv_output = x
                    tape.watch(conv_output)
            
            predictions = x
            
            # Handle sigmoid output (single value per sample)
            if predictions.shape[-1] == 1:
                loss = predictions[:, 0] if class_index == 1 else (1 - predictions[:, 0])
            else:
                loss = predictions[:, class_index]
        
        if conv_output is None:
            raise ValueError(f"Could not capture output from layer {conv_layer.name}")
        
        # Get gradients
        grads = tape.gradient(loss, conv_output)
        del tape  # Clean up persistent tape
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_weighted = conv_output[0] * pooled_grads
        
        heatmap = tf.reduce_sum(conv_outputs_weighted, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    # ------------------- IMAGE (PyTorch) ------------------- #
    def explain_image(self, model: torch.nn.Module, input_tensor: torch.Tensor,
                      target_class: Optional[int] = None, target_layer_name: Optional[str] = None
                      ) -> np.ndarray:
       
        model.eval()
        input_tensor = input_tensor.to(TORCH_DEVICE)
        # ---------------- find target layer ---------------- #
        target_layer = None
        if target_layer_name:
            for name, module in model.named_modules():
                if name == target_layer_name:
                    target_layer = module
                    break
        else:
            # Last Conv2d
            for module in reversed(list(model.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break
        if target_layer is None:
            raise ValueError("No convolutional layer found for Grad-CAM")

        # ---------------- forward hook ---------------- #
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_backward_hook(backward_hook)

        # Forward pass
        output = model(input_tensor)
        
        # Handle Hugging Face model outputs (have .logits attribute)
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        loss = logits[0, target_class]
        model.zero_grad()
        loss.backward()

        # Grad-CAM calculation
        grads = gradients[0][0]  # C, H, W
        acts = activations[0][0]  # C, H, W

        weights = grads.mean(dim=(1, 2))  # channel-wise mean
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32).to(TORCH_DEVICE)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam_np = cam.cpu().detach().numpy()

        handle_forward.remove()
        handle_backward.remove()

        return cam_np

    # ------------------- VISUALIZATION ------------------- #
    def overlay_heatmap(self, heatmap: np.ndarray, original: np.ndarray,
                        alpha: float = 0.5, colormap='jet') -> np.ndarray:
       
        import cv2
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        if original.ndim == 2:
            original_rgb = np.stack([original]*3, axis=-1)
        else:
            original_rgb = original
        # Ensure original is uint8
        if original_rgb.max() <= 1.0:
            original_rgb = np.uint8(original_rgb * 255)
        else:
            original_rgb = np.uint8(original_rgb)
        overlay = cv2.addWeighted(heatmap_img, alpha, original_rgb, 1 - alpha, 0)
        return overlay

    def explain(self, model, input_tensor, target_layer=None, save_path: Optional[Path] = None):
        """
        Unified explain method for Flask app - handles PyTorch image models
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor (1, C, H, W)
            target_layer: Optional target layer name
            save_path: Optional path to save visualization
        
        Returns:
            Tuple of (overlay_image, heatmap, figure)
        """
        # Generate heatmap
        heatmap = self.explain_image(model, input_tensor, target_class=None,
                                     target_layer_name=target_layer)
        
        # Get original image from tensor for overlay
        # Denormalize the input tensor
        mean = np.array(IMAGE_CONFIG["mean"]).reshape(1, 1, 3)
        std = np.array(IMAGE_CONFIG["std"]).reshape(1, 1, 3)
        
        img_np = input_tensor.cpu().numpy()[0]  # (C, H, W)
        img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
        img_np = img_np * std + mean  # Denormalize
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        # Create overlay
        overlay = self.overlay_heatmap(heatmap, img_np)
        
        # Create visualization figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image', fontweight='bold', fontsize=12)
        axes[0].axis('off')
        
        # Heatmap
        import cv2
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontweight='bold', fontsize=12)
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Grad-CAM Overlay', fontweight='bold', fontsize=12)
        axes[2].axis('off')
        
        plt.suptitle('Grad-CAM Explanation', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close figure to free memory and flush file
            print(f"✓ Grad-CAM visualization saved to {save_path}")
        
        # Return scores as dict with heatmap stats
        scores = {
            'max_activation': float(heatmap.max()),
            'mean_activation': float(heatmap.mean()),
            'min_activation': float(heatmap.min())
        }
        
        return overlay, scores, fig

# ------------------- CONVENIENCE FUNCTION ------------------- #
def explain_with_gradcam(model, input_tensor, original_data, input_type='image',
                         target_class=None, target_layer=None, save_path: Optional[Path] = None):
  
    explainer = GradCAMExplainer()

    if input_type == 'audio':
        heatmap = explainer.explain_audio(model, input_tensor, class_index=target_class if target_class else 1)
    elif input_type == 'image':
        heatmap = explainer.explain_image(model, input_tensor, target_class=target_class,
                                          target_layer_name=target_layer)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    # Convert original_data to numpy if needed
    if hasattr(original_data, 'numpy'):
        original_np = original_data.numpy()
    elif hasattr(original_data, '__array__'):
        original_np = np.array(original_data)
    else:
        original_np = original_data
    
    # Handle PIL Image
    from PIL import Image
    if isinstance(original_data, Image.Image):
        original_np = np.array(original_data.resize((224, 224)))

    overlay = explainer.overlay_heatmap(heatmap, original_np)

    if save_path:
        plt.imsave(save_path, overlay)
        print(f"✓ Grad-CAM visualization saved to {save_path}")

    return overlay, heatmap

# ============================================================================

if __name__ == "__main__":
    print("Grad-CAM Explainer ready for images and audio!")

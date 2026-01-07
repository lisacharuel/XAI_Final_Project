"""
Grad-CAM Explainer for Audio Spectrograms and Images
Supports:
- Audio: Keras models (sigmoid output)
- Image: PyTorch models (softmax output)
"""

import numpy as np
import torch
import torch.nn.functional as F
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
        
        if layer_name is None:
            # Find last Conv2D layer
            for l in reversed(model.layers):
                if 'conv' in l.name:
                    layer_name = l.name
                    break
        conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(
            [model.inputs], [conv_layer.output, model.output]
        )

        spectrogram_tensor = tf.convert_to_tensor(spectrogram_tensor, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(spectrogram_tensor)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = tf.reduce_sum(conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-8
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
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        loss = output[0, target_class]
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
        heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        if original.ndim == 2:
            original_rgb = np.stack([original]*3, axis=-1)
        else:
            original_rgb = original
        overlay = cv2.addWeighted(heatmap_img, alpha, np.uint8(original_rgb), 1 - alpha, 0)
        return overlay

# ------------------- CONVENIENCE FUNCTION ------------------- #
def explain_with_gradcam(model, input_tensor, original_data, input_type='image',
                         target_class=None, target_layer=None, save_path: Optional[Path] = None):
  
    explainer = GradCAMExplainer()

    if input_type == 'audio':
        heatmap = explainer.explain_audio(model, input_tensor, class_index=target_class)
    elif input_type == 'image':
        heatmap = explainer.explain_image(model, input_tensor, target_class=target_class,
                                          target_layer_name=target_layer)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

    overlay = explainer.overlay_heatmap(heatmap, original_data)

    if save_path:
        plt.imsave(save_path, overlay)
        print(f"✓ Grad-CAM visualization saved to {save_path}")

    return overlay, heatmap

# ============================================================================

if __name__ == "__main__":
    print("Grad-CAM Explainer ready for images and audio!")

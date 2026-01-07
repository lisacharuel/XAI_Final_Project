import numpy as np
import torch
import shap
from pathlib import Path
from tensorflow.keras.models import Model
from config import TORCH_DEVICE, IMAGE_CONFIG

class SHAPExplainer:
    def __init__(self, nsamples=100):
        self.nsamples = nsamples

    def explain_audio(self, model: Model, spectrogram_tensor: np.ndarray):
        background = np.zeros_like(spectrogram_tensor)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(spectrogram_tensor, nsamples=self.nsamples)
        return np.array(shap_values)

    def explain_image(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        model.eval()
        background = torch.zeros_like(input_tensor).to(TORCH_DEVICE)
        def f(x):
            x_t = torch.tensor(x, dtype=torch.float32).permute(0,3,1,2).to(TORCH_DEVICE)
            with torch.no_grad():
                out = model(x_t)
                if hasattr(out, "logits"):
                    out = out.logits
                return torch.softmax(out, dim=1).cpu().numpy()
        input_np = input_tensor.permute(0,2,3,1).cpu().numpy()
        explainer = shap.KernelExplainer(f, background.permute(0,2,3,1).cpu().numpy())
        shap_values = explainer.shap_values(input_np, nsamples=self.nsamples)
        return shap_values

def explain_with_shap(model, input_tensor, input_type='image', save_path: Path = None):
    explainer = SHAPExplainer()
    if input_type == 'audio':
        values = explainer.explain_audio(model, input_tensor)
    elif input_type == 'image':
        values = explainer.explain_image(model, input_tensor)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")
    if save_path is not None:
        np.save(save_path, values)
    return values

if __name__ == "__main__":
    print("SHAP Explainer ready for images and audio")

"""
Audio Processor
Handles audio loading, preprocessing, and conversion to mel-spectrograms
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import torch

from config import AUDIO_CONFIG, DEVICE


class AudioProcessor:
    """
    Processes audio files for model input
    """
    
    def __init__(self):
        self.sample_rate = AUDIO_CONFIG["sample_rate"]
        self.duration = AUDIO_CONFIG["duration"]
        self.n_mels = AUDIO_CONFIG["n_mels"]
        self.n_fft = AUDIO_CONFIG["n_fft"]
        self.hop_length = AUDIO_CONFIG["hop_length"]
        self.target_height = AUDIO_CONFIG["spectrogram_height"]
        self.target_width = AUDIO_CONFIG["spectrogram_width"]
    
    
    def load_audio(self, file_path: Path) -> np.ndarray:
        """
        Load audio file and resample if necessary
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Audio waveform as numpy array
        """
        try:
            # Load audio file
            audio, sr = librosa.load(
                file_path, 
                sr=self.sample_rate,
                duration=self.duration
            )
            
            # Pad or trim to exact duration
            target_length = int(self.sample_rate * self.duration)
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                # Trim to target length
                audio = audio[:target_length]
            
            return audio
            
        except Exception as e:
            raise ValueError(f"Error loading audio file: {e}")
    
    
    def audio_to_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio waveform to mel-spectrogram
        
        Args:
            audio: Audio waveform (1D numpy array)
        
        Returns:
            Mel-spectrogram (2D numpy array)
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    
    def resize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Resize spectrogram to target dimensions
        
        Args:
            spectrogram: Input spectrogram
        
        Returns:
            Resized spectrogram
        """
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = (
            self.target_height / spectrogram.shape[0],
            self.target_width / spectrogram.shape[1]
        )
        
        # Resize using zoom
        resized = zoom(spectrogram, zoom_factors, order=1)
        
        return resized
    
    
    def normalize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Normalize spectrogram to [0, 1] range
        
        Args:
            spectrogram: Input spectrogram
        
        Returns:
            Normalized spectrogram
        """
        # Min-max normalization
        spec_min = spectrogram.min()
        spec_max = spectrogram.max()
        
        if spec_max - spec_min == 0:
            return np.zeros_like(spectrogram)
        
        normalized = (spectrogram - spec_min) / (spec_max - spec_min)
        return normalized
    
    
    def preprocess(self, file_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Complete preprocessing pipeline for audio file
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Tuple of (model_input_tensor, original_spectrogram)
            - model_input_tensor: Ready for model inference (batch, channels, H, W)
            - original_spectrogram: For visualization
        """
        # Load audio
        audio = self.load_audio(file_path)
        
        # Convert to mel-spectrogram
        mel_spec = self.audio_to_melspectrogram(audio)
        
        # Resize to target dimensions
        mel_spec_resized = self.resize_spectrogram(mel_spec)
        
        # Normalize
        mel_spec_norm = self.normalize_spectrogram(mel_spec_resized)
        
        # Convert to tensor
        # Add batch and channel dimensions: (1, 1, H, W) for grayscale
        # or (1, 3, H, W) for RGB (duplicate channels)
        tensor = torch.from_numpy(mel_spec_norm).float()
        
        # Duplicate to 3 channels (RGB) for models expecting RGB input
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        tensor = tensor.unsqueeze(0)  # (1, 3, H, W)
        
        # Move to device
        tensor = tensor.to(DEVICE)
        
        return tensor, mel_spec_norm
    
    
    def visualize_spectrogram(self, spectrogram: np.ndarray, 
                             title: str = "Mel-Spectrogram",
                             save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create visualization of mel-spectrogram
        
        Args:
            spectrogram: Mel-spectrogram to visualize
            title: Plot title
            save_path: Optional path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Display spectrogram
        img = librosa.display.specshow(
            spectrogram,
            x_axis='time',
            y_axis='mel',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            ax=ax,
            cmap='viridis'
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Mel Frequency', fontsize=12)
        
        # Add colorbar
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.set_label('Amplitude (dB)', fontsize=12)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    
    def get_audio_info(self, file_path: Path) -> dict:
        """
        Get information about audio file
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Dictionary with audio information
        """
        try:
            audio, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            return {
                "duration_seconds": round(duration, 2),
                "sample_rate": sr,
                "num_samples": len(audio),
                "resampled_to": self.sample_rate,
                "processed_duration": self.duration
            }
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create a global audio processor instance
audio_processor = AudioProcessor()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def process_audio_file(file_path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Quick function to process an audio file
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Tuple of (model_input, spectrogram)
    """
    return audio_processor.preprocess(file_path)


# Test function
if __name__ == "__main__":
    # Test with a dummy audio file
    print("Audio Processor initialized")
    print(f"Sample rate: {audio_processor.sample_rate}")
    print(f"Duration: {audio_processor.duration}s")
    print(f"Mel bands: {audio_processor.n_mels}")
    print(f"Target dimensions: {audio_processor.target_height}x{audio_processor.target_width}")

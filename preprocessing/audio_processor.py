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

from keras.preprocessing import image

from config import AUDIO_CONFIG


class AudioProcessor:
    """
    Processes audio files for model input
    """
    
    def __init__(self):
        self.spectrogram_height = AUDIO_CONFIG["spectrogram_height"]
        self.spectrogram_width = AUDIO_CONFIG["spectrogram_width"]
    
    def preprocess(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for audio file
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Tuple of:
                - Preprocessed mel-spectrogram as 3-channel image (np.ndarray)
                - Original mel-spectrogram for visualization (np.ndarray)
        """
        # Load audio
        audio, sr = librosa.load(file_path, sr=None)
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Store original for visualization
        original_spec = mel_spec_db.copy()
        
        # Add channel dimension: (128, 153) -> (128, 153, 1)
        mel_spec_3d = np.expand_dims(mel_spec_db, axis=-1)
        
        # Resize for model input
        img_resized = image.smart_resize(
            mel_spec_3d,
            size=(self.spectrogram_height, self.spectrogram_width)
        )
        
        # Convert to 3 channels: (224, 224, 1) -> (224, 224, 3)
        img_rgb = np.concatenate([img_resized, img_resized, img_resized], axis=-1)

        return img_rgb, original_spec
    
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
                "channels": 1  # librosa loads as mono
            }
        except Exception as e:
            return {"error": str(e)}


# Global audio processor instance
audio_processor = AudioProcessor()


def preprocess(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for preprocessing audio files
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Tuple of (preprocessed_tensor, original_spectrogram)
    """
    return audio_processor.preprocess(file_path)
"""
File Handler Utility
Manages file uploads, validation, and temporary storage
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Optional
import tempfile

from config import AUDIO_CONFIG, IMAGE_CONFIG, APP_CONFIG


class FileHandler:
    """
    Handles all file operations for the application
    """
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "unified_xai_temp"
        self.temp_dir.mkdir(exist_ok=True)
    
    
    def validate_file(self, file_path: Path) -> Tuple[bool, str, Optional[str]]:
        """
        Validate uploaded file
        
        Args:
            file_path: Path to the uploaded file
        
        Returns:
            Tuple of (is_valid, file_type, error_message)
            - is_valid: True if file is valid
            - file_type: 'audio' or 'image' if valid, None otherwise
            - error_message: Error description if invalid, None otherwise
        """
        
        # Check if file exists
        if not file_path.exists():
            return False, None, "File does not exist"
        
        # Check file size
        file_size = file_path.stat().st_size
        max_size = APP_CONFIG["max_file_size"]
        if file_size > max_size:
            max_mb = max_size / (1024 * 1024)
            return False, None, f"File size exceeds {max_mb}MB limit"
        
        # Check file extension
        file_ext = file_path.suffix.lower()
        
        # Determine file type
        if file_ext in AUDIO_CONFIG["supported_formats"]:
            file_type = "audio"
        elif file_ext in IMAGE_CONFIG["supported_formats"]:
            file_type = "image"
        else:
            supported = (AUDIO_CONFIG["supported_formats"] + 
                        IMAGE_CONFIG["supported_formats"])
            return False, None, f"Unsupported format. Supported: {supported}"
        
        return True, file_type, None
    
    
    def save_upload(self, uploaded_file, original_filename: str) -> Path:
        """
        Save uploaded file to temporary directory
        
        Args:
            uploaded_file: File object from web framework
            original_filename: Original name of the file
        
        Returns:
            Path to saved file
        """
        # Create unique filename to avoid conflicts
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        file_ext = Path(original_filename).suffix
        new_filename = f"{unique_id}_{original_filename}"
        
        save_path = self.temp_dir / new_filename
        
        # Save file
        with open(save_path, 'wb') as f:
            if hasattr(uploaded_file, 'read'):
                # File-like object
                shutil.copyfileobj(uploaded_file, f)
            else:
                # Bytes
                f.write(uploaded_file)
        
        return save_path
    
    
    def get_file_info(self, file_path: Path) -> dict:
        """
        Get information about a file
        
        Args:
            file_path: Path to the file
        
        Returns:
            Dictionary with file information
        """
        stat = file_path.stat()
        
        return {
            "filename": file_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "extension": file_path.suffix,
            "path": str(file_path)
        }
    
    
    def cleanup_temp_files(self, older_than_hours: int = 24):
        """
        Clean up old temporary files
        
        Args:
            older_than_hours: Remove files older than this many hours
        """
        import time
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        
        for file_path in self.temp_dir.iterdir():
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
    
    
    def delete_file(self, file_path: Path):
        """
        Delete a specific file
        
        Args:
            file_path: Path to file to delete
        """
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def is_audio_file(file_path: Path) -> bool:
    """Check if file is an audio file"""
    return file_path.suffix.lower() in AUDIO_CONFIG["supported_formats"]


def is_image_file(file_path: Path) -> bool:
    """Check if file is an image file"""
    return file_path.suffix.lower() in IMAGE_CONFIG["supported_formats"]


def get_file_type(file_path: Path) -> str:
    """
    Get the type of file ('audio' or 'image')
    
    Args:
        file_path: Path to the file
    
    Returns:
        'audio' or 'image'
    
    Raises:
        ValueError if file type cannot be determined
    """
    if is_audio_file(file_path):
        return "audio"
    elif is_image_file(file_path):
        return "image"
    else:
        raise ValueError(f"Cannot determine type for file: {file_path}")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create a global file handler instance
file_handler = FileHandler()

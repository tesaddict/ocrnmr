"""OCR engine for extracting text from image frames using EasyOCR."""

import io
import sys
import warnings
from typing import Optional, List

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import easyocr
    import numpy as np
    import torch
    has_easyocr = True
    
    # Suppress PyTorch pin_memory warnings on MPS (Apple Silicon)
    # These warnings are harmless but distracting
    # The warning message contains "pin_memory" and "MPS" or "not supported on MPS"
    warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*not supported on MPS.*', category=UserWarning)
except ImportError:
    easyocr = None
    np = None
    torch = None
    has_easyocr = False

# Global reader instance (initialized lazily)
EASYOCR_READER = None


def initialize_reader(gpu: Optional[bool] = None) -> None:
    """
    Initialize EasyOCR reader with GPU acceleration if available.
    
    Args:
        gpu: If True, use GPU; if False, use CPU; if None, auto-detect (default: None)
    
    Raises:
        RuntimeError: If EasyOCR is not available
    """
    global EASYOCR_READER
    
    if not has_easyocr:
        raise RuntimeError(
            "easyocr is not installed. Please install it with: pip install easyocr"
        )
    
    if EASYOCR_READER is not None:
        return  # Already initialized
    
    # Auto-detect GPU if not specified
    if gpu is None:
        # Check for CUDA (NVIDIA GPU)
        if torch is not None and torch.cuda.is_available():
            gpu = True
        # Check for MPS (Apple Silicon GPU)
        elif torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu = True
        else:
            gpu = False
    
    try:
        # Initialize EasyOCR reader
        # Use GPU if available, otherwise CPU
        # Note: canvas_size might need to be set at initialization, but Reader doesn't accept it
        # We'll set it in readtext_batched calls instead
        EASYOCR_READER = easyocr.Reader(['en'], gpu=gpu)
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize EasyOCR: {str(e)}\n"
            "Make sure EasyOCR is properly installed: pip install easyocr"
        ) from e


def extract_text_from_frame(frame_bytes: bytes) -> Optional[str]:
    """
    Extract text from a frame image using EasyOCR with vanilla settings.
    
    Args:
        frame_bytes: JPEG/PNG image bytes
    
    Returns:
        Extracted text string, or None if no text found or on error
    
    Raises:
        RuntimeError: If EasyOCR or Pillow is not available
    """
    global EASYOCR_READER
    
    if Image is None:
        raise RuntimeError(
            "Pillow is not installed. Please install it with: pip install Pillow"
        )
    
    if not has_easyocr:
        raise RuntimeError(
            "easyocr is not installed. Please install it with: pip install easyocr"
        )
    
    try:
        # Initialize reader if not already done
        if EASYOCR_READER is None:
            initialize_reader(gpu=None)  # Auto-detect GPU
        
        # Optimize image loading: use BytesIO efficiently
        # For JPEG/MJPEG frames (most common), PIL can decode directly without full image object creation
        image_bytes_io = io.BytesIO(frame_bytes)
        image = Image.open(image_bytes_io)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL Image to numpy array for EasyOCR
        img_array = np.array(image)
        
        # Run EasyOCR with vanilla settings (no custom parameters)
        # EasyOCR returns a list of tuples: (bbox, text, confidence)
        results = EASYOCR_READER.readtext(img_array)
        
        # Extract text with confidence filtering
        # Use a lower threshold (0.3) to catch more text, but still filter noise
        # This helps catch title cards that might have lower confidence due to stylization
        words = []
        for bbox, text, confidence in results:
            # Filter by confidence - lower threshold for stylized text
            if confidence >= 0.3:  # 30% confidence threshold
                text_clean = text.strip()
                if text_clean and len(text_clean) > 1:  # Filter out single characters
                    words.append(text_clean)
        
        # Combine all detected text
        if words:
            text = ' '.join(words)
            # Clean up the text
            text = ' '.join(text.split())  # Normalize whitespace
            if text.strip():
                return text.strip()
        
        return None
        
    except Exception as e:
        # Return None on error - errors are handled at higher levels
        # No need to print here as it will be noisy and errors are shown via rich display
        return None

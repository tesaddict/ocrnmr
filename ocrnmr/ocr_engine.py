"""OCR engine for extracting text from image frames using EasyOCR."""

import io
import sys
import warnings
from typing import Optional, List, Tuple, Union

try:
    from PIL import Image, ImageEnhance, ImageOps
except ImportError:
    Image = None
    ImageEnhance = None
    ImageOps = None

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


def _run_ocr_on_image(image) -> Tuple[Optional[str], float]:
    """Helper to run OCR on a PIL Image and return (text, confidence)."""
    try:
        img_array = np.array(image)
        results = EASYOCR_READER.readtext(img_array)
        
        current_words = []
        current_conf_sum = 0.0
        word_count = 0
        
        for bbox, text, confidence in results:
            if confidence >= 0.3:
                text_clean = text.strip()
                if len(text_clean) > 1:
                    current_words.append(text_clean)
                    current_conf_sum += confidence
                    word_count += 1
        
        if word_count > 0:
            avg_conf = current_conf_sum / word_count
            full_text = ' '.join(current_words)
            return full_text, avg_conf
            
        return None, 0.0
    except Exception:
        return None, 0.0


def extract_text_from_frame(frame: Union[bytes, "Image.Image"]) -> Optional[str]:
    """
    Extract text from a frame image using EasyOCR with smart multi-pass preprocessing.
    
    Optimization:
    1. Runs on original image first.
    2. If no text found, returns None immediately (fastest for empty frames).
    3. If text found with high confidence (> 0.8), returns immediately.
    4. Only if text is found but low confidence (< 0.8), tries other strategies.
    
    Args:
        frame: JPEG/PNG image bytes or a pre-loaded PIL Image
    
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
        
        # Optimize image loading: accept bytes or pre-constructed PIL Images
        if Image is not None and isinstance(frame, Image.Image):
            original_image = frame.copy()
        else:
            image_bytes_io = io.BytesIO(frame)
            original_image = Image.open(image_bytes_io)
        
        # Ensure RGB mode for consistency
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # 1. Run on Original Image
        best_text, best_confidence = _run_ocr_on_image(original_image)
        
        # Optimization: If no text found in original, assume empty frame and stop.
        # This prevents running expensive preprocessing on 90% of video frames.
        if best_text is None:
            return None
            
        # Optimization: If confidence is high enough, stop.
        if best_confidence > 0.8:
            return best_text
            
        # 2. If we are here, we found text but confidence is low (< 0.8).
        # Try to improve it with preprocessing strategies.
        
        strategies = []
        
        if ImageEnhance:
            strategies.append(
                lambda img: ImageEnhance.Contrast(img.convert('L')).enhance(2.0).convert('RGB')
            )
            
        if ImageEnhance and ImageOps:
            strategies.append(
                lambda img: ImageOps.invert(ImageEnhance.Contrast(img.convert('L')).enhance(2.0)).convert('RGB')
            )
            
        # Add thresholding strategies
        strategies.append(
            lambda img: img.convert('L').point(lambda x: 0 if x < 128 else 255).convert('RGB')
        )
        
        if ImageOps:
             strategies.append(
                lambda img: ImageOps.invert(img.convert('L').point(lambda x: 0 if x < 128 else 255)).convert('RGB')
            )
        
        for strategy in strategies:
            try:
                processed_image = strategy(original_image)
                text, conf = _run_ocr_on_image(processed_image)
                
                if text and conf > best_confidence:
                    best_confidence = conf
                    best_text = text
                    
                    # Early exit if we improved enough
                    if best_confidence > 0.8:
                        break
            except Exception:
                continue
        
        return best_text
        
    except Exception as e:
        # Return None on error - errors are handled at higher levels
        return None

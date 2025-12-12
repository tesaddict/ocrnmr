"""OCR engine for extracting text from image frames using EasyOCR."""

import io
import sys
import warnings
import logging
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
    warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*not supported on MPS.*', category=UserWarning)
except ImportError:
    easyocr = None
    np = None
    torch = None
    has_easyocr = False

# Global reader instance (initialized lazily)
EASYOCR_READER = None
logger = logging.getLogger(__name__)


def initialize_reader(gpu: Optional[bool] = None, quantize: bool = True, verbose: bool = True) -> None:
    """
    Initialize EasyOCR reader with GPU acceleration if available.
    
    Args:
        gpu: Use GPU acceleration (default: auto-detect)
        quantize: Use dynamic quantization (default: True)
        verbose: Print initialization status (default: True)
    """
    global EASYOCR_READER
    
    if not has_easyocr:
        raise RuntimeError("easyocr is not installed. Please install it with: pip install easyocr")
    
    if EASYOCR_READER is not None:
        return
    
    # Auto-detect GPU if not specified
    if gpu is None:
        if torch is not None and torch.cuda.is_available():
            gpu = True
            if verbose: logger.info("EasyOCR: Using CUDA (NVIDIA/ROCm GPU)")
        elif torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu = True
            if verbose: logger.info("EasyOCR: Using MPS (Apple Silicon GPU)")
        else:
            gpu = False
            if verbose: logger.info("EasyOCR: Using CPU (No GPU detected)")
    else:
        if verbose: logger.info(f"EasyOCR: Using {'GPU' if gpu else 'CPU'} (Manual override)")
    
    try:
        # quantize default is True in easyocr, but we expose it here
        EASYOCR_READER = easyocr.Reader(['en'], gpu=gpu, quantize=quantize)
    except Exception as e:
        # Fallback to CPU if GPU initialization fails
        if gpu:
            if verbose: logger.warning(f"EasyOCR: GPU initialization failed ({e}), falling back to CPU")
            try:
                EASYOCR_READER = easyocr.Reader(['en'], gpu=False, quantize=quantize)
                return
            except Exception as e2:
                raise RuntimeError(f"Failed to initialize EasyOCR (CPU fallback also failed): {str(e2)}") from e2
        raise RuntimeError(f"Failed to initialize EasyOCR: {str(e)}") from e


def _process_ocr_results(results, min_confidence: float = 0.3) -> Tuple[Optional[str], float]:
    """Helper to process raw EasyOCR results into text and confidence."""
    current_words = []
    current_conf_sum = 0.0
    word_count = 0
    
    for bbox, text, confidence in results:
        if confidence >= min_confidence:
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


def extract_text_from_batch(
    frames: List[Union[bytes, "Image.Image"]],
    decoder: str = 'greedy',
    beamWidth: int = 5,
    canvas_size: int = 2560,
    mag_ratio: float = 1.0,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    min_confidence: float = 0.3,
) -> List[Optional[str]]:
    """
    Extract text from a batch of frames using EasyOCR batch processing.
    
    Args:
        frames: List of image bytes or PIL Images
        decoder: 'greedy' or 'beamsearch' (default: 'greedy')
        beamWidth: Beam width for beam search (default: 5)
        canvas_size: Maximum image dimension (default: 2560)
        mag_ratio: Image magnification ratio (default: 1.0)
        text_threshold: Text confidence threshold (default: 0.7)
        link_threshold: Link confidence threshold (default: 0.4)
        low_text: Low text confidence threshold (default: 0.4)
        min_confidence: Minimum confidence to include a word in result (default: 0.3)
        
    Returns:
        List of extracted text strings (or None if no text found)
    """
    global EASYOCR_READER
    
    if not has_easyocr:
        raise RuntimeError("easyocr is not installed")
        
    if not frames:
        return []
        
    try:
        if EASYOCR_READER is None:
            initialize_reader(gpu=None)
            
        # Convert all frames to numpy arrays (RGB)
        images_np = []
        for frame in frames:
            if isinstance(frame, bytes):
                img = Image.open(io.BytesIO(frame))
            else:
                img = frame
                
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images_np.append(np.array(img))
            
        results_list = []
        
        for img_np in images_np:
            # Pass parameters to readtext
            results = EASYOCR_READER.readtext(
                img_np,
                decoder=decoder,
                beamWidth=beamWidth,
                canvas_size=canvas_size,
                mag_ratio=mag_ratio,
                text_threshold=text_threshold,
                link_threshold=link_threshold,
                low_text=low_text
            )
            text, _ = _process_ocr_results(results, min_confidence=min_confidence)
            results_list.append(text)
            
        return results_list
        
    except Exception:
        return [None] * len(frames)



def clear_gpu_memory():
    """Clear GPU memory if using CUDA or MPS."""
    if not has_easyocr or torch is None:
        return
        
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             # MPS doesn't have an explicit empty_cache equivalent in stable, but we can try collecting garbage
             pass
    except Exception:
        pass


def extract_text_from_frame(frame: Union[bytes, "Image.Image"]) -> Optional[str]:
    """Extract text from a single frame."""
    results = extract_text_from_batch([frame])
    return results[0] if results else None
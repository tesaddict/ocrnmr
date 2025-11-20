"""Main orchestrator for scanning video files for title cards and matching episodes."""

import io
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    from PIL import Image
except ImportError:
    Image = None

from ocrnmr.frame_extractor import extract_frames_batch
from ocrnmr.ocr_engine import extract_text_from_batch, extract_text_from_frame, initialize_reader
from ocrnmr.episode_matcher import match_episode

# Set up logger
logger = logging.getLogger(__name__)


def _load_image_from_bytes(frame_bytes: bytes) -> "Image.Image":
    """Load a PIL Image from raw frame bytes."""
    if Image is None:
        raise RuntimeError("Pillow is required for title card scanning. Please install it with: pip install Pillow")
    image = Image.open(io.BytesIO(frame_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def _resize_image(image: "Image.Image", target_dimension: int) -> "Image.Image":
    """Resize image to fit within target_dimension while preserving aspect ratio."""
    if target_dimension is None or target_dimension <= 0:
        return image
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= target_dimension:
        return image
    scale = target_dimension / float(longest_side)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.BILINEAR)


def find_episode_by_title_card(
    media_file: Path,
    episode_names: List[str],
    match_threshold: float = 0.65,
    return_details: bool = False,
    duration: Optional[float] = None,
    frame_interval: float = 1.0,
    enable_profiling: bool = True,
    hwaccel: Optional[str] = None,
    start_time: Optional[float] = None,
    max_dimension: Optional[int] = None
) -> Union[Optional[Tuple[Path, str]], Optional[Tuple[Path, str, float, str]]]:
    """
    Scan a video file for title cards and match against episode names.
    
    Extracts all frames using batch extraction, then processes them sequentially.
    
    Args:
        media_file: Path to the video file to scan
        episode_names: List of episode names to match against
        match_threshold: Minimum similarity ratio for episode matching (default: 0.65)
        return_details: If True, return (media_file, episode_name, timestamp, extracted_text) (default: False)
        duration: Maximum duration to process in seconds. If set, only processes the first N seconds of the video.
                 Useful when title cards are known to appear early (e.g., duration=600 for first 10 minutes).
                 If None, processes entire video (default: None)
        frame_interval: Interval between frames in seconds (default: 1.0)
        enable_profiling: If True, log FFMPEG and OCR timing information at INFO level (default: True)
        hwaccel: Hardware acceleration for decoding (None for software, 'videotoolbox' for macOS, 'vaapi' for Linux, 'd3d11va'/'dxva2' for Windows)
        start_time: Optional start time (seconds) for extraction. Used when falling back to synchronous extraction.
        max_dimension: Maximum width/height for OCR. If set, frames are resized by ffmpeg.
    
    Returns:
        If return_details=False: Tuple of (media_file, episode_name) if match found, else None
        If return_details=True: Tuple of (media_file, episode_name, timestamp, extracted_text) if match found, else None
    """
    if not media_file.exists():
        return None
    
    if not episode_names:
        return None
    
    # Verify EasyOCR is available upfront
    try:
        initialize_reader(gpu=None)  # Auto-detect GPU
    except Exception:
        pass
    
    try:
        result = _find_episode_sequential(
            media_file, episode_names, match_threshold, return_details,
            duration, frame_interval, enable_profiling, hwaccel,
            start_time=start_time,
            max_dimension=max_dimension
        )
        
        # If return_details is False and result has details, strip them
        if result and not return_details and len(result) > 2:
            return (result[0], result[1])
        return result
    except Exception:
        return None


def _find_episode_sequential(
    media_file: Path,
    episode_names: List[str],
    match_threshold: float,
    return_details: bool = False,
    duration: Optional[float] = None,
    frame_interval: float = 1.0,
    enable_profiling: bool = True,
    hwaccel: Optional[str] = None,
    start_time: Optional[float] = None,
    max_dimension: Optional[int] = None
) -> Optional[Tuple]:
    """
    Sequential processing: extract all frames at once using batch extraction, then process them.
    """
    try:
        frame_count = 0
        
        # Extract all frames using batch extraction
        ffmpeg_start = time.time()
        
        if duration:
            logger.info(f"Extracting frames from ffmpeg (first {duration:.1f}s / {duration/60:.1f} min)...")
        else:
            logger.info("Extracting frames from ffmpeg...")
            
        frames = extract_frames_batch(
            media_file,
            interval_seconds=frame_interval,
            max_dimension=max_dimension,  # Use ffmpeg scaling if provided
            duration=duration,
            hwaccel=hwaccel,
            start_time=start_time
        )
        
        ffmpeg_time = time.time() - ffmpeg_start
        if enable_profiling:
            logger.info(f"FFMPEG extraction completed in {ffmpeg_time:.2f}s ({len(frames)} frames)")
        
        # Process frames in batches
        ocr_start = time.time()
        frame_count = 0
        
        # Determine if we need Python-side resizing
        use_heuristic_downscale = (max_dimension is None)
        target_dimension = None
        
        # Batch size for OCR
        BATCH_SIZE = 16
        
        # Process frames in chunks
        for i in range(0, len(frames), BATCH_SIZE):
            batch_frames = frames[i:i+BATCH_SIZE]
            batch_images = []
            batch_timestamps = []
            
            # Load and preprocess images for this batch
            for timestamp, frame_bytes in batch_frames:
                frame_count += 1
                if frame_count % 50 == 0:
                    logger.info(f"Processed {frame_count} frames, current timestamp: {int(timestamp // 60):02d}:{int(timestamp % 60):02d}")
                
                try:
                    image = _load_image_from_bytes(frame_bytes)
                    
                    # Apply heuristic downscaling if needed
                    if use_heuristic_downscale:
                        if target_dimension is None:
                            width, height = image.size
                            longest_side = max(width, height)
                            computed = int(longest_side / 2)
                            if computed > 0 and computed < longest_side:
                                target_dimension = computed
                                logger.info(f"Heuristic downscale enabled: {longest_side}px → {target_dimension}px")
                        
                        if target_dimension:
                            image = _resize_image(image, target_dimension)
                            
                    batch_images.append(image)
                    batch_timestamps.append(timestamp)
                except Exception as exc:
                    logger.debug(f"Skipping frame at {timestamp:.1f}s: could not load image ({exc})")
                    continue
            
            if not batch_images:
                continue
                
            # Run batch OCR
            batch_texts = extract_text_from_batch(batch_images)
            
            # Check results
            for j, extracted_text in enumerate(batch_texts):
                if not extracted_text:
                    continue
                    
                timestamp = batch_timestamps[j]
                
                matched_episode = match_episode(
                    extracted_text,
                    episode_names,
                    threshold=match_threshold
                )
                
                if matched_episode:
                    match_min = int(timestamp // 60)
                    match_sec = int(timestamp % 60)
                    logger.info(
                        f"Match found at {match_min:02d}:{match_sec:02d}: {matched_episode} (after {frame_count} frames)"
                    )
                    ocr_time = time.time() - ocr_start
                    if enable_profiling:
                        logger.info(f"Total FFMPEG time: {ffmpeg_time:.2f}s, Total OCR processing time: {ocr_time:.2f}s")
                    if return_details:
                        return (media_file, matched_episode, timestamp, extracted_text)
                    else:
                        return (media_file, matched_episode)

        ocr_time = time.time() - ocr_start
        if enable_profiling:
            logger.info(f"Total FFMPEG time: {ffmpeg_time:.2f}s, Total OCR processing time: {ocr_time:.2f}s")
        logger.info(f"No match found in entire video (processed {frame_count} frames)")
        return None
        
    except Exception as e:
        logger.error(f"Error in find_episode_by_title_card: {e}", exc_info=True)
        return None

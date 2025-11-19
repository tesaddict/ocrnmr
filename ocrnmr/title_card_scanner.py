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
from ocrnmr.ocr_engine import extract_text_from_frame, initialize_reader
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
    pre_extracted_frames: Optional[List[Tuple[float, bytes]]] = None,
    start_time: Optional[float] = None
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
        pre_extracted_frames: Optional pre-extracted frames list. If provided, skips FFMPEG extraction and uses these frames directly.
                            Useful for pipelined extraction where frames are extracted in background (default: None)
        start_time: Optional start time (seconds) for extraction. Used when falling back to synchronous extraction.
    
    Returns:
        If return_details=False: Tuple of (media_file, episode_name) if match found, else None
        If return_details=True: Tuple of (media_file, episode_name, timestamp, extracted_text) if match found, else None
    
    Example:
        >>> result = find_episode_by_title_card(
        ...     Path("episode.mkv"),
        ...     ["Pilot", "The Second Episode", "Finale"]
        ... )
        >>> if result:
        ...     file_path, episode_name = result
        ...     print(f"Found: {episode_name}")
    """
    if not media_file.exists():
        return None
    
    if not episode_names:
        return None
    
    # Verify EasyOCR is available upfront
    # Initialize with GPU auto-detection for better performance
    try:
        initialize_reader(gpu=None)  # Auto-detect GPU (MPS for Apple Silicon, CUDA for NVIDIA)
    except Exception:
        # If initialization fails, let extract_text_from_frame handle it
        pass
    
    try:
        result = _find_episode_sequential(
            media_file, episode_names, match_threshold, return_details,
            duration, frame_interval, enable_profiling, hwaccel,
            pre_extracted_frames=pre_extracted_frames,
            start_time=start_time
        )
        
        # If return_details is False and result has details, strip them
        if result and not return_details and len(result) > 2:
            return (result[0], result[1])
        return result
    except Exception:
        # Return None on any error
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
    pre_extracted_frames: Optional[List[Tuple[float, bytes]]] = None,
    start_time: Optional[float] = None
) -> Optional[Tuple]:
    """
    Sequential processing: extract all frames at once using batch extraction, then process them.
    
    Uses batch extraction - all frames are extracted first, then processed sequentially.
    This allows for accurate timing measurements of FFMPEG vs OCR processing time.
    
    Args:
        pre_extracted_frames: Optional pre-extracted frames list. If provided, skips FFMPEG extraction
                             and uses these frames directly. FFMPEG time will be reported as 0.0.
    """
    try:
        frame_count = 0
        target_dimension = None
        
        # Extract all frames using batch extraction (or use pre-extracted frames)
        ffmpeg_start = time.time()
        if pre_extracted_frames is not None:
            frames = pre_extracted_frames
            ffmpeg_time = 0.0  # Already done in background
            if enable_profiling:
                logger.info(f"Using pre-extracted frames ({len(frames)} frames, FFMPEG time already accounted)")
        else:
            if duration:
                logger.info(f"Extracting frames from ffmpeg (first {duration:.1f}s / {duration/60:.1f} min)...")
            else:
                logger.info("Extracting frames from ffmpeg...")
            frames = extract_frames_batch(
                media_file,
                interval_seconds=frame_interval,
                max_dimension=None,  # Always extract at full resolution
                duration=duration,
                hwaccel=hwaccel,
                start_time=start_time
            )
        
        ffmpeg_time = time.time() - ffmpeg_start
        if enable_profiling:
            logger.info(f"FFMPEG extraction completed in {ffmpeg_time:.2f}s ({len(frames)} frames)")
        
        # Process frames individually to avoid memory issues
        ocr_start = time.time()
        frame_count = 0
        
        candidate_frames: List[Tuple[float, bytes]] = []
        
        for timestamp, frame_bytes in frames:
            frame_count += 1
            if frame_count % 50 == 0:
                logger.info(f"Processed {frame_count} frames, current timestamp: {int(timestamp // 60):02d}:{int(timestamp % 60):02d}")
            
            try:
                image = _load_image_from_bytes(frame_bytes)
            except Exception as exc:
                logger.debug(f"Skipping frame {frame_count}: could not load image ({exc})")
                continue
            
            if target_dimension is None:
                width, height = image.size
                longest_side = max(width, height)
                computed = int(longest_side / 2)
                if computed > 0 and computed < longest_side:
                    target_dimension = computed
                    logger.info(f"Heuristic downscale enabled: {longest_side}px → {target_dimension}px")
            
            extracted_text = None
            if target_dimension:
                resized_image = _resize_image(image, target_dimension)
                extracted_text = extract_text_from_frame(resized_image)
                if extracted_text:
                    logger.info(f"Downscaled OCR at {timestamp:.1f}s: '{extracted_text}'")
            else:
                extracted_text = extract_text_from_frame(image)
            
            if not extracted_text:
                continue
            
            matched_episode = match_episode(
                extracted_text,
                episode_names,
                threshold=match_threshold
            )
            
            if matched_episode:
                match_min = int(timestamp // 60)
                match_sec = int(timestamp % 60)
                logger.info(
                    f"Match found (downscaled) at {match_min:02d}:{match_sec:02d}: {matched_episode} (after {frame_count} frames)"
                )
                ocr_time = time.time() - ocr_start
                if enable_profiling:
                    logger.info(f"Total FFMPEG time: {ffmpeg_time:.2f}s, Total OCR processing time: {ocr_time:.2f}s")
                if return_details:
                    return (media_file, matched_episode, timestamp, extracted_text)
                else:
                    return (media_file, matched_episode)
            
            # Store for later full-res retry if we were operating in downscaled mode
            if target_dimension:
                candidate_frames.append((timestamp, frame_bytes))
        
        # Full-resolution retry pass for frames where text was detected but no match
        if target_dimension and candidate_frames:
            logger.info(f"No match after downscaled pass. Retrying {len(candidate_frames)} candidate frames at full resolution...")
            for timestamp, frame_bytes in candidate_frames:
                try:
                    image = _load_image_from_bytes(frame_bytes)
                except Exception as exc:
                    logger.debug(f"Skipping candidate frame at {timestamp:.1f}s: could not load image ({exc})")
                    continue
                
                full_res_text = extract_text_from_frame(image)
                if not full_res_text:
                    continue
                
                logger.info(f"Full-res OCR (retry) at {timestamp:.1f}s: '{full_res_text}'")
                
                matched_episode = match_episode(
                    full_res_text,
                    episode_names,
                    threshold=match_threshold
                )
                
                if matched_episode:
                    match_min = int(timestamp // 60)
                    match_sec = int(timestamp % 60)
                    logger.info(
                        f"Match found (full_res) at {match_min:02d}:{match_sec:02d}: {matched_episode}"
                    )
                    ocr_time = time.time() - ocr_start
                    if enable_profiling:
                        logger.info(f"Total FFMPEG time: {ffmpeg_time:.2f}s, Total OCR processing time: {ocr_time:.2f}s")
                    if return_details:
                        return (media_file, matched_episode, timestamp, full_res_text)
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

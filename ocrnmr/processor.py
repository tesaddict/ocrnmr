"""Sequential OCR processor with true streaming."""

import logging
import time
import os
import threading
import queue
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

from ocrnmr.title_card_scanner import _load_image_from_bytes, _resize_image
from ocrnmr.ocr_engine import extract_text_from_batch
from ocrnmr.episode_matcher import match_episode
from ocrnmr.display import OCRProgressDisplay
from ocrnmr.filename import sanitize_filename
from ocrnmr.frame_extractor import extract_frames_generator
from ocrnmr.profiler import profiler

logger = logging.getLogger(__name__)


class StatusCallback:
    """Status callback for video processing events."""
    
    def __init__(
        self,
        display: OCRProgressDisplay,
        video_files: List[Path],
        episode_info: Optional[Dict[str, Tuple[int, int]]],
        show_name: Optional[str]
    ):
        self.display = display
        self.video_files = video_files
        self.episode_info = episode_info
        self.show_name = show_name
        self.total_files = len(video_files)
    
    def update_status(self, file_index: int, file_path: str, status: str, message: str, match_timestamp: Optional[float] = None):
        """Update status in display."""
        self.display.update_status(
            file=file_path,
            index=file_index,
            total=self.total_files,
            status=status,
            message=message
        )
        
        if message:
            self.display.add_log(message)
            
    def match_found(self, file_index: int, file_path: str, matched_episode: str, match_timestamp: Optional[float]):
        """Handle match found event."""
        new_filename = None
        if matched_episode and self.episode_info and matched_episode in self.episode_info:
            ep_season, ep_num = self.episode_info[matched_episode]
            file_path_obj = Path(file_path)
            ext = file_path_obj.suffix
            new_filename = f"{self.show_name} - S{ep_season:02d}E{ep_num:02d} - {matched_episode}{ext}"
            new_filename = sanitize_filename(new_filename)
        
        message = f"Match found: {matched_episode}"
        self.display.add_log(message)
        self.display.add_completed_file(file_path, matched_episode, True, new_filename=new_filename, match_timestamp=match_timestamp)

    def no_match(self, file_index: int, file_path: str):
        """Handle no match event."""
        message = f"No match found for {Path(file_path).name}"
        self.display.add_log(message)
        self.display.add_completed_file(file_path, None, False, new_filename=None, match_timestamp=None)

    def error(self, file_index: int, file_path: str, error_msg: str):
        """Handle error event."""
        message = f"Error: {error_msg}"
        self.display.add_log(message)
        self.display.add_completed_file(file_path, None, False, new_filename=None, match_timestamp=None)


def _process_batch_two_pass(
    batch_frames: List[Tuple[float, bytes]],
    episode_titles: List[str],
    ocr_config: Dict,
    use_heuristic_downscale: bool,
    target_dimension: Optional[int]
) -> Tuple[Optional[str], Optional[float]]:
    """
    Process a batch of frames using two-pass OCR strategy.
    
    Pass 1: Low resolution (min_dimension) to detect text.
    Pass 2: High resolution (max_dimension) on candidates to recognize text.
    """
    min_dimension = ocr_config.get('min_dimension', 320)
    max_dimension = ocr_config.get('max_dimension', 800)
    match_threshold = ocr_config.get('match_threshold', 0.6)
    
    # --- PASS 1: Low Resolution Filter ---
    pass1_images = []
    pass1_indices = []
    
    profiler.start_timer("load_and_resize_pass1")
    for i, (ts, fb) in enumerate(batch_frames):
        try:
            image = _load_image_from_bytes(fb)
            # Resize to min_dimension for fast detection
            image_small = _resize_image(image, min_dimension)
            pass1_images.append(image_small)
            pass1_indices.append(i)
        except Exception:
            continue
    profiler.stop_timer("load_and_resize_pass1")
            
    if not pass1_images:
        return None, None
        
    # Run OCR on small images - use very low confidence (0.1) to just detect *any* text
    profiler.start_timer("ocr_pass1")
    pass1_texts = extract_text_from_batch(pass1_images, decoder='greedy', min_confidence=0.1)
    profiler.stop_timer("ocr_pass1")
    
    # Check for immediate matches or candidates
    candidate_indices = []
    
    profiler.start_timer("match_pass1")
    for i, text in enumerate(pass1_texts):
        if text:
            # Check if we got lucky and matched on low-res (unlikely but possible)
            matched = match_episode(text, episode_titles, threshold=match_threshold)
            if matched:
                profiler.stop_timer("match_pass1")
                original_idx = pass1_indices[i]
                return matched, batch_frames[original_idx][0]
            
            # If text detected but no match, mark as candidate
            candidate_indices.append(pass1_indices[i])
    profiler.stop_timer("match_pass1")
            
    if not candidate_indices:
        return None, None
        
    # --- PASS 2: High Resolution Refinement ---
    pass2_images = []
    pass2_timestamps = []
    
    profiler.start_timer("load_and_resize_pass2")
    for idx in candidate_indices:
        ts, fb = batch_frames[idx]
        try:
            image = _load_image_from_bytes(fb)
            # Resize to max_dimension (or heuristic) for accurate recognition
            if use_heuristic_downscale and target_dimension:
                image = _resize_image(image, target_dimension)
            elif max_dimension:
                image = _resize_image(image, max_dimension)
                
            pass2_images.append(image)
            pass2_timestamps.append(ts)
        except Exception:
            continue
    profiler.stop_timer("load_and_resize_pass2")
            
    if not pass2_images:
        return None, None
        
    # Run OCR on candidate images (high res)
    profiler.start_timer("ocr_pass2")
    pass2_texts = extract_text_from_batch(pass2_images, decoder='greedy', min_confidence=0.3)
    profiler.stop_timer("ocr_pass2")
    
    profiler.start_timer("match_pass2")
    for i, text in enumerate(pass2_texts):
        if text:
            matched = match_episode(text, episode_titles, threshold=match_threshold)
            if matched:
                profiler.stop_timer("match_pass2")
                return matched, pass2_timestamps[i]
    profiler.stop_timer("match_pass2")
                
    return None, None


def process_video_files(
    input_directory: Path,
    episode_titles: List[str],
    ocr_config: Dict,
    display: OCRProgressDisplay,
    episode_info: Optional[Dict[str, Tuple[int, int]]] = None,
    show_name: Optional[str] = None
) -> List[Tuple[Path, Optional[str]]]:
    """
    Process video files and match episodes using true streaming.
    """
    # Get video files
    extensions = ['*.mkv', '*.mp4', '*.avi', '*.m4v', '*.mov']
    video_files = []
    for ext in extensions:
        video_files.extend(input_directory.glob(ext))
    # Add uppercase versions just in case (Linux is case sensitive)
    for ext in extensions:
        video_files.extend(input_directory.glob(ext.upper()))
        
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        display.add_log(f"No video files found in {input_directory} (checked mkv, mp4, avi, m4v, mov)")
        return []
    
    total_files = len(video_files)
    display.add_log(f"Found {total_files} video files")
    display.total_files = total_files
    
    # Create status callback
    callback = StatusCallback(display, video_files, episode_info, show_name)
    
    # Mark OCR start time
    display.ocr_start_time = time.time()
    
    matches = []
    
    try:
        for idx, video_file in enumerate(video_files):
            # Check for early exit
            if display.early_exit_requested:
                display.add_log("Early exit requested, stopping processing")
                break
            
            file_path_str = str(video_file)
            callback.update_status(idx, file_path_str, "processing", f"Processing {video_file.name}")
            
            try:
                matched_episode = None
                match_timestamp = None
                
                # Batch size for OCR
                BATCH_SIZE = ocr_config.get('batch_size', 32)
                
                frame_count = 0
                use_heuristic_downscale = (ocr_config.get('max_dimension') is None)
                target_dimension = None
                
                # Start generator
                frame_generator = extract_frames_generator(
                    video_file,
                    interval_seconds=ocr_config.get('frame_interval', 2.0),
                    max_dimension=ocr_config.get('max_dimension'),
                    duration=ocr_config.get('duration'),
                    hwaccel=ocr_config.get('hwaccel'),
                    start_time=ocr_config.get('start_time')
                )
                
                # Process frames in batches as they arrive
                current_batch_frames = []
                
                for timestamp, frame_bytes in frame_generator:
                    current_batch_frames.append((timestamp, frame_bytes))
                    
                    # Determine target dimension once if using heuristic
                    if use_heuristic_downscale and target_dimension is None:
                        try:
                            image = _load_image_from_bytes(frame_bytes)
                            width, height = image.size
                            longest_side = max(width, height)
                            computed = int(longest_side / 2)
                            if computed > 0 and computed < longest_side:
                                target_dimension = computed
                        except Exception:
                            pass

                    if len(current_batch_frames) >= BATCH_SIZE:
                        # Process batch with two-pass strategy
                        matched_episode, match_timestamp = _process_batch_two_pass(
                            current_batch_frames,
                            episode_titles,
                            ocr_config,
                            use_heuristic_downscale,
                            target_dimension
                        )
                        
                        frame_count += len(current_batch_frames)
                        current_batch_frames = []
                        
                        if matched_episode:
                            # Stop generator (will kill FFMPEG process on garbage collection or exit)
                            break
                
                # Process remaining frames in last batch
                if not matched_episode and current_batch_frames:
                    matched_episode, match_timestamp = _process_batch_two_pass(
                        current_batch_frames,
                        episode_titles,
                        ocr_config,
                        use_heuristic_downscale,
                        target_dimension
                    )
                    frame_count += len(current_batch_frames)
                
                if matched_episode:
                    matches.append((video_file, matched_episode))
                    callback.match_found(idx, file_path_str, matched_episode, match_timestamp)
                else:
                    matches.append((video_file, None))
                    callback.no_match(idx, file_path_str)
                    
            except Exception as e:
                logger.error(f"Error processing {video_file.name}: {e}")
                matches.append((video_file, None))
                callback.error(idx, file_path_str, str(e))
                
    except KeyboardInterrupt:
        display.add_log("Processing interrupted by user")
        raise
    finally:
        display.ocr_end_time = time.time()
        display.update_status(total_files, "", "finished", "Processing complete")
    
    return matches

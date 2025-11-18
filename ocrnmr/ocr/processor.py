"""Pipelined OCR processor with background frame extraction."""

import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

from .frame_extractor import extract_frames_batch
from .title_card_scanner import find_episode_by_title_card

logger = logging.getLogger(__name__)

# Import FORCE_EXIT flag from exit_flag module
from ocrnmr.exit_flag import FORCE_EXIT


class FrameCache:
    """Manages pipelined FFMPEG frame extraction with memory limits."""
    
    def __init__(self, memory_limit_bytes: int = 1073741824):  # 1GB default
        self.memory_limit = memory_limit_bytes
        self.cache: Dict[Path, Tuple[Future, int]] = {}  # {file_path: (future, memory_bytes)}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self.currently_extracting: Optional[Path] = None  # Track which file is actually being extracted right now
        self.currently_extracting_index: Optional[int] = None
        self.logger.info(f"FrameCache initialized with {memory_limit_bytes / 1024 / 1024:.0f}MB memory limit")
    
    def estimate_memory(self, frames: List[Tuple[float, bytes]]) -> int:
        """Estimate memory usage of frame list in bytes."""
        if not frames:
            return 0
        # Sum of frame bytes + ~10% overhead for list/tuple overhead
        base_memory = sum(len(frame_bytes) for _, frame_bytes in frames)
        return int(base_memory * 1.1)
    
    def get_total_memory(self) -> int:
        """Get total memory used by all cached frames."""
        with self.lock:
            return sum(memory_bytes for _, memory_bytes in self.cache.values())
    
    def start_extraction(self, file_path: Path, config: Dict[str, Any], status_callback: Optional[Callable] = None, file_index: Optional[int] = None) -> None:
        """Start background frame extraction for a file."""
        with self.lock:
            if file_path in self.cache:
                # Already extracting or cached
                return
            
            # Check if we have room (we'll check again when extraction completes)
            future = self.executor.submit(self._extract_frames, file_path, config, status_callback, file_index)
            # We don't know memory yet, so estimate conservatively
            self.cache[file_path] = (future, 0)
            self.logger.info(f"Queued background FFMPEG extraction for {file_path.name}")
            # Don't show "Starting" message here - it will be shown when extraction actually starts in _extract_frames
    
    def _extract_frames(self, file_path: Path, config: Dict[str, Any], status_callback: Optional[Callable] = None, file_index: Optional[int] = None) -> List[Tuple[float, bytes]]:
        """Extract frames in background thread."""
        start_time = time.time()
        
        # Update currently extracting file
        with self.lock:
            self.currently_extracting = file_path
            self.currently_extracting_index = file_index
        
        # Show start message when extraction actually begins
        if status_callback:
            status_callback('ffmpeg_start', file_index, str(file_path), 'decoding', f"Starting frame extraction for {file_path.name}", 'ffmpeg')
        
        try:
            frames = extract_frames_batch(
                file_path,
                interval_seconds=config.get("frame_interval", 2.0),
                max_dimension=config.get("max_dimension"),
                duration=config.get("duration"),
                hwaccel=config.get("hwaccel")
            )
            extraction_time = time.time() - start_time
            
            # Update memory estimate
            memory_bytes = self.estimate_memory(frames)
            
            with self.lock:
                if file_path in self.cache:
                    self.cache[file_path] = (self.cache[file_path][0], memory_bytes)
                    total_memory = sum(mem_bytes for _, mem_bytes in self.cache.values())
                    log_msg = (
                        f"FFMPEG extraction completed for {file_path.name}: "
                        f"{len(frames)} frames, ~{memory_bytes / 1024 / 1024:.1f}MB "
                        f"(total cache: ~{total_memory / 1024 / 1024:.1f}MB)"
                    )
                    self.logger.info(log_msg)
                    if status_callback:
                        status_callback('ffmpeg_complete', file_index, str(file_path), 'finished', 
                                      f"Frame extraction completed for {file_path.name} ({len(frames)} frames)", 'ffmpeg')
            
            # Clear currently extracting when done
            with self.lock:
                if self.currently_extracting == file_path:
                    self.currently_extracting = None
                    self.currently_extracting_index = None
            
            return frames
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Background extraction failed for {file_path.name} after {elapsed:.2f}s: {e}"
            self.logger.error(error_msg)
            with self.lock:
                if file_path in self.cache:
                    del self.cache[file_path]
                if self.currently_extracting == file_path:
                    self.currently_extracting = None
                    self.currently_extracting_index = None
            if status_callback:
                status_callback('ffmpeg_error', file_index, str(file_path), 'standby', f"Error: {str(e)[:50]}", 'ffmpeg')
            raise
    
    def get_frames(self, file_path: Path, status_callback: Optional[Callable] = None) -> Optional[List[Tuple[float, bytes]]]:
        """Get frames for a file, waiting if extraction is in progress."""
        with self.lock:
            if file_path not in self.cache:
                return None
            
            future, memory_bytes = self.cache[file_path]
        
        # Check if extraction is already complete
        if future.done():
            try:
                frames = future.result()
                return frames
            except Exception as e:
                error_msg = f"Error getting frames for {file_path.name}: {e}"
                self.logger.error(error_msg)
                with self.lock:
                    if file_path in self.cache:
                        del self.cache[file_path]
                return None
        
        # Extraction still in progress - wait for it with timeout to allow interrupt checking
        self.logger.info(f"Waiting for background FFMPEG extraction to complete for {file_path.name}...")
        
        try:
            # Poll with very short timeout and check FORCE_EXIT flag
            while not future.done():
                # Check force exit flag first - if set, exit immediately
                if FORCE_EXIT.is_set():
                    try:
                        future.cancel()
                    except Exception:
                        pass
                    os._exit(130)
                
                try:
                    frames = future.result(timeout=0.01)  # Very short timeout (10ms) for quick interrupt response
                    break
                except TimeoutError:
                    # Continue polling - allows FORCE_EXIT check on each iteration
                    continue
            
            # Check force exit again before getting final result
            if FORCE_EXIT.is_set():
                try:
                    future.cancel()
                except Exception:
                    pass
                os._exit(130)
            
            # Get final result if not already got it
            if not future.done():
                frames = future.result(timeout=0.01)
            else:
                frames = future.result()
            
            # Check memory limit after extraction completes
            try:
                total_memory = self.get_total_memory()
                if total_memory > self.memory_limit:
                    self.logger.warning(
                        f"Memory limit exceeded: {total_memory / 1024 / 1024:.1f}MB > "
                        f"{self.memory_limit / 1024 / 1024:.1f}MB. "
                        f"Consider reducing duration or frame_interval."
                    )
            except Exception:
                pass
            
            return frames
        except KeyboardInterrupt:
            # Cancel the future and force exit immediately
            FORCE_EXIT.set()
            try:
                future.cancel()
            except Exception:
                pass
            with self.lock:
                if file_path in self.cache:
                    del self.cache[file_path]
            os._exit(130)
        except Exception as e:
            error_msg = f"Error waiting for frames for {file_path.name}: {e}"
            self.logger.error(error_msg)
            with self.lock:
                if file_path in self.cache:
                    del self.cache[file_path]
            return None
    
    def get_currently_extracting(self) -> Tuple[Optional[Path], Optional[int]]:
        """Get the file that's currently being extracted."""
        with self.lock:
            return (self.currently_extracting, self.currently_extracting_index)
    
    def cleanup(self, file_path: Path) -> None:
        """Remove frames from cache after processing."""
        with self.lock:
            if file_path in self.cache:
                del self.cache[file_path]
                self.logger.debug(f"Cleaned up frames for {file_path.name}")
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor and cancel pending tasks aggressively."""
        # Check force exit - if set, kill everything immediately
        if FORCE_EXIT.is_set():
            # Kill all FFMPEG processes immediately
            try:
                subprocess.run(['pkill', '-9', 'ffmpeg'], timeout=0.5, capture_output=True)
            except Exception:
                pass
            # Don't wait for executor - just exit
            os._exit(130)
        
        # Cancel all pending futures immediately
        with self.lock:
            for file_path, (future, _) in list(self.cache.items()):
                if not future.done():
                    future.cancel()
                    self.logger.info(f"Cancelled extraction for {file_path.name}")
        
        # Kill any active FFMPEG processes aggressively (no wait)
        try:
            from .frame_extractor import _active_processes
            for proc in list(_active_processes):
                try:
                    proc.kill()  # Kill immediately, don't wait
                except Exception:
                    pass
            # Also use pkill as fallback
            try:
                subprocess.run(['pkill', '-9', 'ffmpeg'], timeout=0.5, capture_output=True)
            except Exception:
                pass
        except Exception:
            pass
        
        # Shutdown executor - always use wait=False for immediate exit
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass


class PipelinedOCRProcessor:
    """Processes video files with pipelined FFMPEG extraction."""
    
    def __init__(self, memory_limit_bytes: int = 1073741824):  # 1GB default
        self.frame_cache = FrameCache(memory_limit_bytes)
        self.logger = logging.getLogger(__name__)
    
    def process_files(
        self,
        video_files: List[Path],
        episode_titles: List[str],
        config: Dict[str, Any],
        status_callback: Optional[Callable[[str, Optional[int], Optional[str], str, str, str], None]] = None,
        should_continue: Optional[Callable[[], bool]] = None
    ) -> List[Tuple[Path, Optional[str]]]:
        """
        Process video files with pipelined FFMPEG extraction.
        
        Args:
            video_files: List of video file paths to process
            episode_titles: List of episode titles to match against
            config: Configuration dict with OCR settings (frame_interval, match_threshold, max_dimension, duration, hwaccel)
            status_callback: Optional callback function(status_type, file_index, file_path, status, message)
                           status_type: 'ffmpeg_start', 'ffmpeg_complete', 'ocr_start', 'ocr_complete', 'match_found'
        
        Returns:
            List of tuples (file_path, matched_episode_name or None)
        """
        matches = []
        total_files = len(video_files)
        
        # Normalize config values
        ocr_config = {
            'frame_interval': config.get("frame_interval", 2.0),
            'match_threshold': config.get("match_threshold", 0.6),
            'max_dimension': config.get("max_dimension"),
            'duration': config.get("duration"),
            'hwaccel': config.get("hwaccel")
        }
        
        # Handle None/0 values
        if ocr_config['max_dimension'] is not None and ocr_config['max_dimension'] == 0:
            ocr_config['max_dimension'] = None
        if ocr_config['duration'] is not None and ocr_config['duration'] == 0:
            ocr_config['duration'] = None
        
        # Start extraction for all files upfront (FFMPEG runs ahead without waiting)
        for idx, video_file in enumerate(video_files):
            self.frame_cache.start_extraction(video_file, ocr_config, status_callback, idx)
        
        # Process each file - FFMPEG extractions are already running in background
        for idx, video_file in enumerate(video_files):
            # Check if we should continue processing (check before each file)
            if should_continue and not should_continue():
                self.logger.info("Early exit requested, stopping processing")
                if status_callback:
                    status_callback('ocr_complete', idx - 1 if idx > 0 else 0, None, 'finished', 
                                  "Early exit requested by user", 'ocr')
                break
            
            file_path_str = str(video_file)
            
            # Wait for current file's frames to be ready (extraction may already be complete)
            if status_callback:
                status_callback('ocr_start', idx, file_path_str, 'standby', f"Starting processing: {video_file.name}", 'ocr')
            
            pre_extracted_frames = self.frame_cache.get_frames(video_file, status_callback)
            
            if pre_extracted_frames is None:
                # Extraction failed or not started, fall back to inline extraction
                self.logger.warning(f"Could not get pre-extracted frames for {video_file.name}, using inline extraction")
                if status_callback:
                    status_callback('ocr_start', idx, file_path_str, 'standby', 
                                  f"Warning: Using inline extraction for {video_file.name}", 'ocr')
                pre_extracted_frames = None
            
            # Process OCR
            if status_callback:
                status_callback('ocr_start', idx, file_path_str, 'processing', f"Processing OCR for {video_file.name}", 'ocr')
            
            try:
                result = find_episode_by_title_card(
                    video_file,
                    episode_titles,
                    match_threshold=ocr_config['match_threshold'],
                    return_details=True,  # Return timestamp for display
                    max_dimension=ocr_config['max_dimension'],
                    duration=ocr_config['duration'],
                    frame_interval=ocr_config['frame_interval'],
                    enable_profiling=False,
                    hwaccel=ocr_config['hwaccel'],
                    pre_extracted_frames=pre_extracted_frames
                )
                
                if result:
                    # When return_details=True, result is (media_file, episode_name, timestamp, extracted_text)
                    if len(result) == 4:
                        _, matched_episode, match_timestamp, _ = result
                    else:
                        # Fallback for old format (shouldn't happen with return_details=True)
                        matched_episode = result[1] if len(result) >= 2 else None
                        match_timestamp = result[2] if len(result) >= 3 else None
                    
                    matches.append((video_file, matched_episode))
                    if status_callback:
                        status_callback('match_found', idx, file_path_str, 'finished', f"Match found: {matched_episode}", 'ocr', match_timestamp=match_timestamp)
                        status_callback('ocr_complete', idx, file_path_str, 'finished', f"Completed: {video_file.name}", 'ocr')
                else:
                    matches.append((video_file, None))
                    if status_callback:
                        status_callback('ocr_complete', idx, file_path_str, 'finished', f"No match found for {video_file.name}", 'ocr')
                        # Also signal no match for display
                        status_callback('match_found', idx, file_path_str, 'finished', f"No match found for {video_file.name}", 'ocr', match_timestamp=None)
                
            except Exception as e:
                self.logger.error(f"Error processing {video_file.name}: {e}")
                matches.append((video_file, None))
                if status_callback:
                    status_callback('ocr_complete', idx, file_path_str, 'finished', f"Error processing {video_file.name}: {str(e)[:50]}", 'ocr')
                    # Also signal error for display
                    status_callback('match_found', idx, file_path_str, 'finished', f"Error processing {video_file.name}: {str(e)[:50]}", 'ocr')
            
            # Clean up frames after processing
            self.frame_cache.cleanup(video_file)
        
        # Shutdown executor - always use wait=False for immediate exit
        try:
            self.frame_cache.shutdown(wait=False)
        except KeyboardInterrupt:
            # Force exit on interrupt
            FORCE_EXIT.set()
            os._exit(130)
        except Exception:
            # If shutdown fails, force exit
            FORCE_EXIT.set()
            os._exit(130)
        
        return matches


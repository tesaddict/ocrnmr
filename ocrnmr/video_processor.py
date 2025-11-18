"""Video processing orchestration."""

import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ocrnmr.display import OCRProgressDisplay
from ocrnmr.processor import PipelinedOCRProcessor
from ocrnmr.filename import sanitize_filename
from ocrnmr.exit_flag import FORCE_EXIT


class StatusCallback:
    """Simplified status callback for video processing events."""
    
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
    
    def ffmpeg_start(self, file_index: int, file_path: str, status: str, message: str, source: str):
        """Handle FFMPEG start event."""
        self.display.update_ffmpeg_status(
            filename=file_path,
            status=status,
            file_index=file_index,
            message=message
        )
        if source == 'ffmpeg':
            self.display.add_ffmpeg_diagnostic(message)
        else:
            self.display.add_ocr_diagnostic(message)
    
    def ffmpeg_complete(self, file_index: int, file_path: str, status: str, message: str, source: str):
        """Handle FFMPEG complete event."""
        self.display.update_ffmpeg_status(
            filename=file_path,
            status=status,
            file_index=file_index,
            message=message
        )
        if source == 'ffmpeg':
            self.display.add_ffmpeg_diagnostic(message)
        else:
            self.display.add_ocr_diagnostic(message)
    
    def ffmpeg_error(self, file_index: int, file_path: str, status: str, message: str, source: str):
        """Handle FFMPEG error event."""
        self.display.update_ffmpeg_status(
            filename=file_path,
            status=status,
            file_index=file_index,
            message=message
        )
        if source == 'ffmpeg':
            self.display.add_ffmpeg_diagnostic(message)
        else:
            self.display.add_ocr_diagnostic(message)
    
    def ocr_start(self, file_index: int, file_path: str, status: str, message: str, source: str):
        """Handle OCR start event."""
        with self.display.lock:
            if file_path and file_path not in self.display.file_start_times:
                self.display.file_start_times[file_path] = time.time()
        
        self.display.update_ocr_status(
            file=file_path,
            index=file_index,
            total=self.total_files,
            status=status
        )
        if source == 'ocr':
            self.display.add_ocr_diagnostic(message)
        else:
            self.display.add_ffmpeg_diagnostic(message)
    
    def ocr_complete(self, file_index: int, file_path: str, status: str, message: str, source: str):
        """Handle OCR complete event."""
        self.display.update_ocr_status(
            file=file_path,
            index=file_index,
            total=self.total_files,
            status=status
        )
        if source == 'ocr':
            self.display.add_ocr_diagnostic(message)
        else:
            self.display.add_ffmpeg_diagnostic(message)
    
    def match_found(self, file_index: int, file_path: str, message: str, source: str, match_timestamp: Optional[float]):
        """Handle match found event."""
        if message.startswith("Match found: "):
            matched_episode = message.replace("Match found: ", "")
            new_filename = None
            if matched_episode and self.episode_info and matched_episode in self.episode_info:
                ep_season, ep_num = self.episode_info[matched_episode]
                file_path_obj = Path(file_path)
                ext = file_path_obj.suffix
                new_filename = f"{self.show_name} - S{ep_season:02d}E{ep_num:02d} - {matched_episode}{ext}"
                new_filename = sanitize_filename(new_filename)
            
            if source == 'ocr':
                self.display.add_ocr_diagnostic(message)
            else:
                self.display.add_ffmpeg_diagnostic(message)
            self.display.add_completed_file(file_path, matched_episode, True, new_filename=new_filename, match_timestamp=match_timestamp)
        elif message.startswith("No match found"):
            if source == 'ocr':
                self.display.add_ocr_diagnostic(message)
            else:
                self.display.add_ffmpeg_diagnostic(message)
            self.display.add_completed_file(file_path, None, False, new_filename=None, match_timestamp=None)
        else:
            if source == 'ocr':
                self.display.add_ocr_diagnostic(message)
            else:
                self.display.add_ffmpeg_diagnostic(message)
    
    def create_callback(self) -> Callable:
        """Create the callback function expected by processor."""
        def callback(event_type: str, file_index: Optional[int], file_path: Optional[str], 
                    status: str, message: str, source: str = 'ocr', match_timestamp: Optional[float] = None) -> None:
            if file_path is None:
                return
            
            idx = file_index if file_index is not None else 0
            
            if event_type == 'ffmpeg_start':
                self.ffmpeg_start(idx, file_path, status, message, source)
            elif event_type == 'ffmpeg_complete':
                self.ffmpeg_complete(idx, file_path, status, message, source)
            elif event_type == 'ffmpeg_error':
                self.ffmpeg_error(idx, file_path, status, message, source)
            elif event_type == 'ocr_start':
                self.ocr_start(idx, file_path, status, message, source)
            elif event_type == 'ocr_complete':
                self.ocr_complete(idx, file_path, status, message, source)
            elif event_type == 'match_found':
                self.match_found(idx, file_path, message, source, match_timestamp)
        
        return callback


def process_video_files(
    input_directory: Path,
    episode_titles: List[str],
    ocr_config: Dict,
    display: OCRProgressDisplay,
    episode_info: Optional[Dict[str, Tuple[int, int]]] = None,
    show_name: Optional[str] = None
) -> List[Tuple[Path, Optional[str]]]:
    """
    Process video files and match episodes using pipelined FFMPEG extraction.
    
    Returns:
        List of tuples (file_path, matched_episode_name or None)
    """
    # Get video files
    video_files = sorted(list(input_directory.glob("*.mkv")))
    if not video_files:
        display.add_ocr_diagnostic(f"No .mkv files found in {input_directory}")
        return []
    
    total_files = len(video_files)
    display.add_ocr_diagnostic(f"Found {total_files} video files")
    display.total_files = total_files
    
    # Create status callback
    status_cb = StatusCallback(display, video_files, episode_info, show_name)
    callback = status_cb.create_callback()
    
    # Initialize processor
    memory_limit = ocr_config.get("memory_limit_bytes", 1073741824)
    processor = PipelinedOCRProcessor(memory_limit_bytes=memory_limit)
    
    # Check for early exit function
    def should_continue() -> bool:
        return not display.early_exit_requested
    
    # Mark OCR start time
    display.ocr_start_time = time.time()
    
    # Process files using pipelined processor
    try:
        matches = processor.process_files(
            video_files=video_files,
            episode_titles=episode_titles,
            config=ocr_config,
            status_callback=callback,
            should_continue=should_continue
        )
    except KeyboardInterrupt:
        FORCE_EXIT.set()
        os._exit(130)
    finally:
        display.ocr_end_time = time.time()
    
    # All processing complete
    display.update_ocr_status(
        file="",
        index=total_files,
        total=total_files,
        status="finished"
    )
    
    return matches


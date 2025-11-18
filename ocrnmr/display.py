"""Display module for OCR test progress using rich."""

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

# Import FORCE_EXIT flag from exit_flag module
from ocrnmr.exit_flag import FORCE_EXIT


class OCRProgressDisplay:
    """Display progress for OCR processing with split panes for OCR and FFMPEG."""
    
    def __init__(self):
        """Initialize the display."""
        self.lock = threading.Lock()
        
        # OCR state
        self.current_file: Optional[str] = None
        self.current_file_index = 0
        self.total_files = 0
        self.ocr_status = "Waiting..."
        self.ocr_diagnostics: deque = deque(maxlen=100)  # OCR activity log (larger buffer, display subset)
        self.ocr_completed: List[Dict] = []  # {file, matched, status} - similar to FFMPEG completed
        
        # FFMPEG state
        self.ffmpeg_extractions: Dict[str, Dict] = {}  # {filename: {status, progress, frames, memory}}
        self.ffmpeg_log: deque = deque(maxlen=100)  # FFMPEG activity log (larger buffer, display subset)
        self.current_ffmpeg_file: Optional[str] = None  # Currently extracting file
        self.current_ffmpeg_file_index: int = -1  # Current file index (0-based, -1 means not set)
        
        # Results
        self.completed_files: List[Dict] = []  # {file, matched, status, elapsed_time}
        
        # Timing statistics
        self.ocr_start_time: Optional[float] = None
        self.ocr_end_time: Optional[float] = None
        self.file_start_times: Dict[str, float] = {}  # Track start time per file
        
        # Error display
        self.error_message: Optional[str] = None
        
        # UI components
        self.console = Console()
        self.live: Optional[Live] = None
        self.running = False
        self.early_exit_requested = False
        
        # Activity log max lines
        self.activity_log_max_lines = 10  # Fixed number of log lines to show
    
    def _get_ocr_pane_content(self) -> Panel:
        """Get content for OCR pane as a rich Panel."""
        with self.lock:
            content_lines = []
            content_lines.append(Text(" OCR Processing ", style="bold green on dark_blue"))
            content_lines.append(Text(" (Press Ctrl+C to cancel)\n", style="dim"))
            content_lines.append(Text("\n"))
            
            # Show current processing status
            if self.current_file:
                content_lines.append(Text("Processing File ", style="dim"))
                content_lines.append(Text(f"{self.current_file_index + 1}/{self.total_files}: {Path(self.current_file).name}\n", style="bright_white"))
            else:
                content_lines.append(Text("Processing File ", style="dim"))
                content_lines.append(Text(f"-/{self.total_files}: -\n", style="bright_white"))
            
            content_lines.append(Text("\n"))
            content_lines.append(Text("Status: ", style="dim"))
            # Normalize status display - use clearer status values
            status_display = self.ocr_status.lower()
            if status_display in ['processing', 'extracting', 'ocr']:
                status_display = "processing"
            elif status_display in ['matching', 'match']:
                status_display = "matching"
            elif status_display in ['finished', 'completed', 'done']:
                status_display = "finished"
            else:
                status_display = "standby"
            content_lines.append(Text(f"{status_display}\n", style="bright_white"))
            
            # Show error message prominently if present
            if self.error_message:
                content_lines.append(Text("\n"))
                content_lines.append(Text("─" * 40 + "\n", style="red"))
                content_lines.append(Text("ERROR:\n", style="bold red"))
                # Split error message into lines and display
                for line in self.error_message.split('\n'):
                    if line.strip():
                        content_lines.append(Text(f"  {line}\n", style="red"))
            
            # Show activity log - newest at top (bottom-up scrolling)
            if self.ocr_diagnostics:
                content_lines.append(Text("\n"))
                content_lines.append(Text("─" * 40 + "\n", style="dim"))
                content_lines.append(Text("Activity Log:\n", style="dim"))
                # Get most recent messages and reverse them (newest first)
                all_messages = list(self.ocr_diagnostics)
                messages_to_show = all_messages[-self.activity_log_max_lines:] if len(all_messages) > self.activity_log_max_lines else all_messages
                # Reverse so newest is at top
                for msg in reversed(messages_to_show):
                    content_lines.append(Text(f"  {msg}\n", style="dim"))
            
            # Combine all text lines
            combined_text = Text()
            for line in content_lines:
                combined_text.append(line)
            
            return Panel(
                combined_text,
                title="OCR Processing",
                border_style="green",
                width=None  # Use full available width from layout
            )
    
    def _get_ffmpeg_pane_content(self) -> Panel:
        """Get content for FFMPEG pane as a rich Panel."""
        with self.lock:
            content_lines = []
            content_lines.append(Text(" FFMPEG Extraction ", style="bold green on dark_blue"))
            content_lines.append(Text(" (Press Ctrl+C to cancel)\n", style="dim"))
            content_lines.append(Text("\n"))
            
            # Show current file being extracted
            # Prioritize current_ffmpeg_file if it has 'decoding' status (it's the one actually being extracted)
            currently_extracting = None
            current_index = -1
            
            # First check if current_ffmpeg_file has 'decoding' status (this is the one actually being extracted)
            if self.current_ffmpeg_file and self.current_ffmpeg_file in self.ffmpeg_extractions:
                info = self.ffmpeg_extractions[self.current_ffmpeg_file]
                status = info.get('status', '').lower()
                if status in ['extracting', 'decoding']:
                    currently_extracting = self.current_ffmpeg_file
                    current_index = self.current_ffmpeg_file_index
            
            # If not found, search for any file with 'decoding' status
            if not currently_extracting:
                for filename, info in self.ffmpeg_extractions.items():
                    status = info.get('status', '').lower()
                    if status in ['extracting', 'decoding']:
                        currently_extracting = filename
                        current_index = info.get('file_index', -1)
                        break
            
            # Check if all extractions are completed first
            all_completed = len(self.ffmpeg_extractions) > 0 and all(
                info.get('status', '').lower() in ['completed', 'finished'] 
                for info in self.ffmpeg_extractions.values()
            )
            
            # Display the file
            if all_completed:
                # All done - show total files completed
                content_lines.append(Text("Processing File ", style="dim"))
                content_lines.append(Text(f"{self.total_files}/{self.total_files}: All files\n", style="bright_white"))
            elif currently_extracting:
                file_index = current_index + 1 if current_index >= 0 else 0
                content_lines.append(Text("Processing File ", style="dim"))
                content_lines.append(Text(f"{file_index}/{self.total_files}: {Path(currently_extracting).name}\n", style="bright_white"))
            elif self.current_ffmpeg_file and self.current_ffmpeg_file_index >= 0:
                # Show last processed file if we have an index
                file_index = self.current_ffmpeg_file_index + 1
                filename = self.current_ffmpeg_file
                content_lines.append(Text("Processing File ", style="dim"))
                content_lines.append(Text(f"{file_index}/{self.total_files}: {Path(filename).name}\n", style="bright_white"))
            else:
                content_lines.append(Text("Processing File ", style="dim"))
                content_lines.append(Text(f"-/{self.total_files}: -\n", style="bright_white"))
            
            content_lines.append(Text("\n"))
            
            # Show status - use clear status values: decoding, standby, finished
            currently_extracting = None
            for filename, info in self.ffmpeg_extractions.items():
                status = info.get('status', '').lower()
                if status in ['extracting', 'decoding']:
                    currently_extracting = filename
                    break
            
            if all_completed:
                content_lines.append(Text("Status: ", style="dim"))
                content_lines.append(Text("finished\n", style="green"))
            elif currently_extracting:
                content_lines.append(Text("Status: ", style="dim"))
                content_lines.append(Text("decoding\n", style="bright_white"))
            else:
                content_lines.append(Text("Status: ", style="dim"))
                content_lines.append(Text("standby\n", style="bright_white"))
            
            # Show activity log - newest at top (bottom-up scrolling)
            if self.ffmpeg_log:
                content_lines.append(Text("\n"))
                content_lines.append(Text("─" * 40 + "\n", style="dim"))
                content_lines.append(Text("Activity Log:\n", style="dim"))
                # Get most recent messages and reverse them (newest first)
                all_messages = list(self.ffmpeg_log)
                messages_to_show = all_messages[-self.activity_log_max_lines:] if len(all_messages) > self.activity_log_max_lines else all_messages
                # Reverse so newest is at top
                for msg in reversed(messages_to_show):
                    content_lines.append(Text(f"  {msg}\n", style="dim"))
            
            # Combine all text lines
            combined_text = Text()
            for line in content_lines:
                combined_text.append(line)
            
            return Panel(
                combined_text,
                title="FFMPEG Extraction",
                border_style="green",
                width=None  # Use full available width from layout
            )
    
    def _get_summary_pane_content(self) -> Panel:
        """Get content for summary pane showing all matches as a rich Panel."""
        with self.lock:
            content_lines = []
            content_lines.append(Text(" Matches ", style="bold green on dark_blue"))
            content_lines.append(Text("\n"))
            
            completed = len(self.completed_files)
            matched = sum(1 for f in self.completed_files if f.get('matched'))
            
            content_lines.append(Text("Completed: ", style="dim"))
            content_lines.append(Text(f"{completed}/{self.total_files}", style="bright_white"))
            content_lines.append(Text("  ", style="dim"))
            content_lines.append(Text("Matched: ", style="dim"))
            content_lines.append(Text(f"{matched}\n", style="green"))
            
            # Show elapsed time if processing has started
            if self.ocr_start_time is not None:
                if self.ocr_end_time is not None:
                    elapsed = self.ocr_end_time - self.ocr_start_time
                else:
                    elapsed = time.time() - self.ocr_start_time
                
                # Format elapsed time nicely
                if elapsed < 60:
                    time_str = f"{elapsed:.1f}s"
                elif elapsed < 3600:
                    minutes = int(elapsed // 60)
                    seconds = int(elapsed % 60)
                    time_str = f"{minutes}m {seconds}s"
                else:
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    time_str = f"{hours}h {minutes}m"
                
                content_lines.append(Text("  ", style="dim"))
                content_lines.append(Text("Elapsed: ", style="dim"))
                content_lines.append(Text(f"{time_str}\n", style="bright_white"))
            else:
                content_lines.append(Text("\n"))
            
            content_lines.append(Text("\n"))
            
            if self.completed_files:
                content_lines.append(Text("─" * 40 + "\n", style="dim"))
                # Show ALL matches - newest at top (reverse the list)
                for result in reversed(self.completed_files):
                    file = result.get('file', 'unknown')
                    matched_ep = result.get('matched')
                    elapsed_time = result.get('elapsed_time')
                    new_filename = result.get('new_filename')
                    match_timestamp = result.get('match_timestamp')
                    
                    # Format elapsed time if available
                    elapsed_str = ""
                    if elapsed_time is not None:
                        if elapsed_time < 60:
                            elapsed_str = f" ({elapsed_time:.1f}s elapsed)"
                        elif elapsed_time < 3600:
                            minutes = int(elapsed_time // 60)
                            seconds = int(elapsed_time % 60)
                            elapsed_str = f" ({minutes}m {seconds}s elapsed)"
                        else:
                            hours = int(elapsed_time // 3600)
                            minutes = int((elapsed_time % 3600) // 60)
                            elapsed_str = f" ({hours}h {minutes}m elapsed)"
                    
                    # Format match timestamp if available
                    timestamp_str = ""
                    if match_timestamp is not None:
                        minutes = int(match_timestamp // 60)
                        seconds = int(match_timestamp % 60)
                        timestamp_str = f" (found at {minutes}:{seconds:02d})"
                    
                    if matched_ep:
                        # Show filename with checkmark
                        content_lines.append(Text("✓ ", style="green"))
                        content_lines.append(Text(f"{Path(file).name}", style="bright_white"))
                        if elapsed_str:
                            content_lines.append(Text(elapsed_str, style="dim"))
                        if timestamp_str:
                            content_lines.append(Text(timestamp_str, style="dim"))
                        content_lines.append(Text("\n"))
                        # Show new filename (fully qualified rename)
                        if new_filename:
                            new_name_display = new_filename[:80] + "..." if len(new_filename) > 80 else new_filename
                            content_lines.append(Text(f"  → {new_name_display}\n", style="dim"))
                        else:
                            # Fallback: show episode name if new_filename not available
                            ep_name = matched_ep[:70] + "..." if len(matched_ep) > 70 else matched_ep
                            content_lines.append(Text(f"  → {ep_name}\n", style="dim"))
                        content_lines.append(Text("\n"))
                    else:
                        # Show filename with X for no match
                        content_lines.append(Text("✗ ", style="red"))
                        content_lines.append(Text(f"{Path(file).name}", style="bright_white"))
                        if elapsed_str:
                            content_lines.append(Text(elapsed_str, style="dim"))
                        content_lines.append(Text("\n"))
                        content_lines.append(Text("\n"))
            else:
                content_lines.append(Text("No matches yet...\n", style="dim"))
            
            # Combine all text lines
            combined_text = Text()
            for line in content_lines:
                combined_text.append(line)
            
            return Panel(
                combined_text,
                title="Matches",
                border_style="cyan",
                width=None  # Full width
            )
    
    def _create_layout(self) -> Layout:
        """Create the rich layout with three panes."""
        # Create top section with two side-by-side panels
        top_layout = Layout()
        top_layout.split_row(
            self._get_ocr_pane_content(),
            self._get_ffmpeg_pane_content()
        )
        
        # Create main layout with top and bottom sections
        main_layout = Layout()
        main_layout.split_column(
            top_layout,
            self._get_summary_pane_content()
        )
        
        return main_layout
    
    def start(self):
        """Start the display in a separate thread."""
        def run_display():
            self.running = True
            try:
                # Clear screen before starting
                self.console.clear()
                # Use Live without screen mode for better compatibility with threads
                # We'll manually clear and redraw to simulate full-screen behavior
                initial_layout = self._create_layout()
                with Live(initial_layout, refresh_per_second=1, screen=False, console=self.console) as live:
                    self.live = live
                    while self.running and not FORCE_EXIT.is_set():
                        # Update the layout with fresh content (1Hz refresh rate)
                        try:
                            live.update(self._create_layout())
                        except Exception as e:
                            # If update fails, silently continue (display errors shouldn't crash the app)
                            # Errors are already shown in the display itself
                            pass
                        time.sleep(1.0)
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                # Display errors shouldn't crash the app - just stop the display
                self.running = False
            finally:
                self.running = False
        
        self.display_thread = threading.Thread(target=run_display, daemon=True)
        self.display_thread.start()
        
        # Give it a moment to start and render
        time.sleep(0.5)
    
    def stop(self):
        """Stop the display."""
        # Check if force exit is requested - if so, exit immediately
        if FORCE_EXIT.is_set():
            os._exit(130)
        
        self.running = False
        # The Live context manager will clean up automatically when the thread exits
    
    def update_ocr_status(self, file: str, index: int, total: int, status: str, 
                         frames_processed: int = 0, total_frames: int = 0, 
                         timestamp: str = "00:00", matched: Optional[str] = None):
        """Update OCR processing status."""
        with self.lock:
            # Track start time when processing begins (first file)
            if self.ocr_start_time is None and status.lower() in ['processing', 'standby'] and index == 0:
                self.ocr_start_time = time.time()
            
            # Track end time when processing finishes (last file)
            if status.lower() in ['finished', 'completed'] and index == total - 1:
                self.ocr_end_time = time.time()
            
            self.current_file = file
            self.current_file_index = index
            self.total_files = total
            self.ocr_status = status
    
    def add_ocr_completed(self, file: str, matched: Optional[str], status: str = ""):
        """Add a completed OCR file to the OCR pane."""
        with self.lock:
            self.ocr_completed.append({
                'file': file,
                'matched': matched,
                'status': status
            })
    
    def update_ffmpeg_status(self, filename: str, status: str, frames: int = 0, 
                            memory_mb: float = 0.0, message: str = "", file_index: Optional[int] = None):
        """Update FFMPEG extraction status."""
        with self.lock:
            if filename not in self.ffmpeg_extractions:
                self.ffmpeg_extractions[filename] = {}
            
            self.ffmpeg_extractions[filename].update({
                'status': status,
                'frames': frames,
                'memory_mb': memory_mb,
                'file_index': file_index  # Store file_index with each extraction
            })
            
            # Update current file if extracting/decoding
            if status.lower() in ['extracting', 'decoding']:
                self.current_ffmpeg_file = filename
                if file_index is not None:
                    self.current_ffmpeg_file_index = file_index
            elif status.lower() in ['completed', 'finished'] and self.current_ffmpeg_file == filename:
                # Keep file name and index visible, just clear extracting status
                # Don't clear current_ffmpeg_file or current_ffmpeg_file_index
                pass
    
    def add_completed_file(self, file: str, matched: Optional[str], success: bool, new_filename: Optional[str] = None, match_timestamp: Optional[float] = None):
        """Add a completed file to the summary."""
        with self.lock:
            # Calculate elapsed time for this file
            elapsed_time = None
            if file in self.file_start_times:
                start_time = self.file_start_times[file]
                elapsed_time = time.time() - start_time
                # Remove from tracking dict to free memory
                del self.file_start_times[file]
            
            self.completed_files.append({
                'file': file,
                'matched': matched,
                'success': success,
                'elapsed_time': elapsed_time,
                'new_filename': new_filename,
                'match_timestamp': match_timestamp
            })
    
    def add_ocr_diagnostic(self, message: str):
        """Add a diagnostic message to OCR pane."""
        with self.lock:
            self.ocr_diagnostics.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def add_ffmpeg_diagnostic(self, message: str):
        """Add a diagnostic message to FFMPEG pane."""
        with self.lock:
            self.ffmpeg_log.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def show_error(self, error_message: str):
        """Display an error message prominently in the display."""
        with self.lock:
            self.error_message = error_message
            self.ocr_status = "ERROR"
        
        # Force an immediate update
        if self.live:
            try:
                self.live.update(self._create_layout())
            except Exception:
                pass
